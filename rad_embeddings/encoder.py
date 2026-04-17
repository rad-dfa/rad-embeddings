import os
import re
import jax
import jraph
import wandb
import distrax
from dfa import DFA
import jax.numpy as jnp
import flax.linen as nn
from .ppo import make_train
from typing import Sequence
from dfa_gym import DFABisimEnv
from .wrappers import LogWrapper
from dfax.samplers import RADSampler
import flax.serialization as serialization
from flax.traverse_util import flatten_dict
from dfax import DFAx, dfa2dfax, list2batch, batch2graph
from flax.linen.initializers import constant, orthogonal


class GATv2Conv(nn.Module):
    out_dim: int
    num_heads: int

    def setup(self):
        self.W_a = nn.Dense(self.num_heads * self.out_dim, use_bias=False)
        self.W_m = nn.Dense(self.num_heads * self.out_dim, use_bias=False)
        self.a = nn.Dense(1, use_bias=False)

    def __call__(self, node_features: jnp.ndarray, edge_features: jnp.ndarray, edge_index: jnp.ndarray, attn_mask: jnp.ndarray) -> jnp.ndarray:
        n_nodes = node_features.shape[0]

        src, tgt = edge_index
        src_features = node_features[src]
        tgt_features = node_features[tgt]

        h_a = self.W_a(
            jnp.concatenate([src_features, edge_features, tgt_features], axis=-1)
        ).reshape(-1, self.num_heads, self.out_dim)

        h_m = self.W_m(
            jnp.concatenate([edge_features, tgt_features], axis=-1)
        ).reshape(-1, self.num_heads, self.out_dim)

        logits = self.a(nn.leaky_relu(h_a, negative_slope=0.2))
        logits = jnp.where(
            attn_mask[:, None, None],
            logits,
            -jnp.inf
        )
        max_per_node = jraph.segment_max(logits.reshape(logits.shape[0], -1),
                                         src,
                                         num_segments=n_nodes)
        dead_nodes = jnp.isneginf(max_per_node[:, 0])
        dead_mask = dead_nodes[src]
        safe_logits = jnp.where(
            dead_mask[:, None, None],
            jnp.zeros_like(logits),
            logits
        )
        attn = jraph.segment_softmax(safe_logits, src, n_nodes)
        msgs = attn * h_m
        h = jraph.segment_sum(msgs, src, num_segments=n_nodes)

        return h


class EncoderModule(nn.Module):
    encoder_dim: int = 32
    max_size: int = 10
    n_heads: int = 4

    def setup(self):
        hidden_dim = self.encoder_dim * 2
        self.linear_h = nn.Dense(hidden_dim, use_bias=False)
        self.linear_e = nn.Dense(hidden_dim, use_bias=False)
        self.gatv2 = GATv2Conv(out_dim=hidden_dim, num_heads=self.n_heads)
        self.g_embed = nn.Dense(self.encoder_dim, use_bias=False)

    def __call__(
        self,
        graph
    ) -> jnp.ndarray:

        h0 = self.linear_h(graph["node_features"].astype(jnp.float32))
        e = self.linear_e(graph["edge_features"].astype(jnp.float32))
        edge_index = graph["edge_index"]
        attn_mask = jnp.any(graph["edge_features"] != 0, axis=-1)

        h = h0
        n_states = graph["n_states"]

        n_iters = self.max_size if self.max_size is not None else n_states.max()

        for i in range(n_iters):
            _h = nn.tanh(
                self.gatv2(
                    node_features=jnp.concatenate([h, h0], axis=-1),
                    edge_features=e,
                    edge_index=edge_index,
                    attn_mask=attn_mask
                ).sum(axis=1)
            )
            h = jnp.where((i < n_states)[:, None], _h, h)

        return self.g_embed(h[graph["current_state"]])

    @classmethod
    def train(
        cls,
        # Configuration
        seed: int = 42,
        save_dir: str | None = None,
        max_size: int = 10,
        n_tokens: int = 10,
        binary_reward: bool = False,
        enable_wandb: bool = False,
        wandb_entity: str = "beyazit-y-berkeley-eecs",
        wandb_project: str = "rad-jax",
        debug: bool = False,
        log: str = "log.csv",
        # Training hyperparameters
        lr: float = 1e-3,
        num_envs: int = 16,
        num_steps: int = 512,
        total_timesteps: int = 1e6,
        update_epochs: int = 2,
        num_minibatches: int = 8,
        gamma: float = 0.9,
        gae_lambda: float = 0.0,
        clip_eps: float = 0.1,
        ent_coef: float = 0.00,
        vf_coef: float = 1.0,
        max_grad_norm: float = 0.5,
        anneal_lr: bool = False,
    ):

        config = {
            "LR": lr,
            "NUM_ENVS": num_envs,
            "NUM_STEPS": num_steps,
            "TOTAL_TIMESTEPS": total_timesteps,
            "UPDATE_EPOCHS": update_epochs,
            "NUM_MINIBATCHES": num_minibatches,
            "GAMMA": gamma,
            "GAE_LAMBDA": gae_lambda,
            "CLIP_EPS": clip_eps,
            "ENT_COEF": ent_coef,
            "VF_COEF": vf_coef,
            "MAX_GRAD_NORM": max_grad_norm,
            "ANNEAL_LR": anneal_lr,
        }

        config["DEBUG"] = debug
        config["WANDB"] = enable_wandb
        config["LOG"] = log

        if config["WANDB"]:
            wandb.init(
                entity=wandb_entity,
                project=wandb_project,
                config=config
            )

        key = jax.random.PRNGKey(seed)

        sampler = RADSampler(max_size=max_size, n_tokens=n_tokens)
        env = DFABisimEnv(sampler=sampler, binary_reward=binary_reward)
        env = LogWrapper(env=env, config=config)

        encoder = cls(max_size=max_size)

        network = ActorCritic(
            action_dim=env.action_space(env.agents[0]).n,
            encoder=encoder
        )

        if config["DEBUG"]:
            key, subkey = jax.random.split(key)
            init_x = env.observation_space(env.agents[0]).sample(subkey)
            key, subkey = jax.random.split(key)
            params = network.init(subkey, init_x)
            flat = flatten_dict(params, sep="/")
            total = 0
            for k, v in flat.items():
                count = v.size
                total += count
                print(f"{k:60} {v.shape} {v.dtype} ({count:,} params)")
            print(f"\nTotal parameters: {total:,}")

        train_jit = jax.jit(make_train(config, env, network))
        out = train_jit(key)

        if save_dir is None:
            save_dir = os.path.join(os.path.dirname(__file__), "storage")

        os.makedirs(save_dir, exist_ok=True)

        trained_params = out["runner_state"][0].params
        with open(f"{save_dir}/encoder_params_max_size_{max_size}_n_tokens_{n_tokens}_seed_{seed}_binary_reward_{binary_reward}.msgpack", "wb") as f:
            f.write(serialization.to_bytes(trained_params))

        if config["WANDB"]:
            wandb.finish()


def distance(feat_l, feat_r):
    safe_l2_norm = lambda x: jnp.sqrt(jnp.sum(x ** 2, axis=-1, keepdims=True) + jnp.finfo(jnp.float32).eps)
    feat_l_normalized = feat_l / safe_l2_norm(feat_l)
    feat_r_normalized = feat_r / safe_l2_norm(feat_r)
    return safe_l2_norm(feat_l_normalized - feat_r_normalized)


class ActorCritic(nn.Module):
    action_dim: int
    encoder: nn.Module
    deterministic: bool = False

    def setup(self):
        self.policy_head = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0))

    def __call__(self, batch):
        
        graph_l = batch2graph(batch["graph_l"])
        graph_r = batch2graph(batch["graph_r"])

        batch = {
            "node_features": jnp.stack(jnp.array([graph_l["node_features"], graph_r["node_features"]])),
            "edge_features": jnp.stack(jnp.array([graph_l["edge_features"], graph_r["edge_features"]])),
            "edge_index": jnp.stack(jnp.array([graph_l["edge_index"], graph_r["edge_index"]])),
            "current_state": jnp.concatenate(jnp.array([graph_l["current_state"], graph_r["current_state"]])),
            "n_states": jnp.stack(jnp.array([graph_l["n_states"], graph_r["n_states"]]))
        }

        graph = batch2graph(batch)

        feat = self.encoder(graph)
        feat_l, feat_r = jnp.array_split(feat, 2)

        value = distance(feat_l, feat_r)
        logits = self.policy_head(feat_l - feat_r)

        if self.deterministic:
            action = jnp.argmax(logits, axis=-1)
            return action, jnp.squeeze(value, axis=-1)
        else:
            pi = distrax.Categorical(logits=logits)
            return pi, jnp.squeeze(value, axis=-1)


class Encoder:
    def __init__(
        self,
        max_size: int = 10,
        n_tokens: int = 10,
        seed: int = 42,
        storage_dir: str | None = None,
        binary_reward: bool = False,
    ):
        key = jax.random.PRNGKey(seed)
        self.encoder = EncoderModule(max_size=max_size)
        sampler = RADSampler()
        dfa = sampler.sample(key)
        dfa_graph = dfa.to_graph()
        self.encoder_ac = ActorCritic(action_dim=n_tokens, encoder=self.encoder, deterministic=True)
        params = self.encoder_ac.init(key, {"graph_l": dfa_graph, "graph_r": dfa_graph})

        if storage_dir is None:
            storage_dir = os.path.join(os.path.dirname(__file__), "storage")

        pattern = re.compile(
            r"encoder_params_max_size_(\d+)_n_tokens_(\d+)_seed_(\d+)_binary_reward_(True|False)\.msgpack"
        )

        candidates = []
        for fname in os.listdir(storage_dir):
            m = pattern.match(fname)
            if m:
                f_max_size, f_n_tokens, f_seed, f_binary_reward = m.groups()
                f_max_size = int(f_max_size)
                f_n_tokens = int(f_n_tokens)
                f_seed = int(f_seed)
                f_binary_reward = f_binary_reward == "True"
                if f_n_tokens == n_tokens and f_binary_reward == binary_reward:
                    candidates.append((f_max_size, f_seed, fname))

        if not candidates:
            raise FileNotFoundError(
                f"No pretrained encoder found with n_tokens == {n_tokens} and binary_reward == {binary_reward}"
            )

        candidates.sort(key=lambda x: x[0])
        chosen = None

        for c in candidates:
            if c[1] == seed:
                chosen = c
                break
        if chosen is None:
            for c in candidates:
                if c[1] == 42:
                    chosen = c
                    break

        if chosen is None:
            raise FileNotFoundError(
                f"No pretrained encoder found for seed {seed} or fallback seed 42 "
                f"with n_tokens == {n_tokens}"
            )

        params_file = os.path.join(storage_dir, chosen[2])
        with open(params_file, "rb") as f:
            self.encoder_ac_params = serialization.from_bytes(params, f.read())

        self.encoder_params = {"params": self.encoder_ac_params["params"]["encoder"]}

    def input2graph(self, inp: DFAx | DFA | dict):
        def _input2graph(x):
            if isinstance(x, DFAx):
                return x.to_graph()
            elif isinstance(x, DFA):
                return dfa2dfax(x).to_graph()
            elif isinstance(x, dict):
                return x
            else:
                raise TypeError(type(x))
        graph = None
        if isinstance(inp, DFAx) or isinstance(inp, DFA) or isinstance(inp, dict):
            graph = _input2graph(inp)
        elif isinstance(inp, Sequence) and not isinstance(inp, (str, bytes)):
            batch = list2batch([_input2graph(x) for x in inp])
            graph = batch2graph(batch)
        else:
            raise TypeError(type(inp))
        assert graph is not None
        return graph

    def __call__(
        self,
        inp: DFAx | DFA | dict | Sequence
    ):
        graph = self.input2graph(inp)
        return jax.lax.stop_gradient(self.encoder.apply(self.encoder_params, graph))

    def solve(
        self,
        inp_l: DFAx | DFA | dict | Sequence,
        inp_r: DFAx | DFA | dict | Sequence
    ):
        graph_l = self.input2graph(inp_l)
        graph_r = self.input2graph(inp_r)
        problem = {"graph_l": graph_l, "graph_r": graph_r}
        action, _ = jax.lax.stop_gradient(self.encoder_ac.apply(self.encoder_ac_params, problem))
        return action

    def distance(self, feat_l, feat_r):
        return distance(feat_l, feat_r)

