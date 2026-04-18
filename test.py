import jax
import argparse
from rad_embeddings.encoder import Encoder
from dfa_gym import DFABisimEnv
from dfax.samplers import RADSampler


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Test DFA encoder")
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed used for PRNGKey (default: 42)"
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="storage",
        help="Directory for saving the trained encoder (default: storage)"
    )
    parser.add_argument(
        "--n",
        type=int,
        default=100,
        help="Number of samples (default: 100)"
    )
    parser.add_argument(
        "--max-size",
        type=int,
        default=None,
        help="Max DFA size, i.e., number of states, to be passed to the encoder (default: None, i.e., unbounded)"
    )
    parser.add_argument(
        "--n-states",
        type=int,
        default=10,
        help="Number of DFA states (default: 10)"
    )
    parser.add_argument(
        "--n-tokens",
        type=int,
        default=10,
        help="Number tokens (default: 10)"
    )
    parser.add_argument(
        "--binary-reward",
        action="store_true",
        help="Use binary rewards in the environment and encoder (default: False)"
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.9,
        help="Gamma (default: 0.9)"
    )
    args = parser.parse_args()

    key = jax.random.PRNGKey(args.seed)

    sampler = RADSampler(max_size=args.n_states, n_tokens=args.n_tokens)
    encoder = Encoder(max_size=args.max_size, n_tokens=args.n_tokens, seed=args.seed, binary_reward=args.binary_reward, gamma=args.gamma, debug=True)
    env = DFABisimEnv(sampler=sampler, binary_reward=args.binary_reward)

    total_reward = 0
    accept_count = 0
    reject_count = 0
    undecide_count = 0
    for i in range(args.n):
        key, subkey = jax.random.split(key)
        obs, state = env.reset(subkey)
        init_state = state
        generated_str = []
        done = False
        for j in range(args.n_states):
            problem = obs[env.agents[0]]
            action = encoder.solve(problem["graph_l"], problem["graph_r"])
            feat_l = encoder(problem["graph_l"])
            feat_r = encoder(problem["graph_r"])
            distance = encoder.distance(feat_l, feat_r)
            action = {agent: action[i] for i, agent in enumerate(env.agents)}
            generated_str.append(action["agent_0"])
            key, subkey = jax.random.split(key)
            obs, state, reward, done, info = env.step(subkey, state, action)
            done = done["__all__"]
            total_reward += reward["agent_0"]
            if done or (j == (args.n_states - 1)):
                if reward["agent_0"] > 0:
                    accept_count += 1
                elif reward["agent_0"] < 0:
                    reject_count += 1
                else:
                    undecide_count += 1
                break

        print(f"Test completed for {i + 1} samples.", end="\r")

    print(f"Test completed for {args.n} samples.")
    print(f"Mean reward:", total_reward/args.n)
    print(f"Accepted rate:", accept_count/args.n)
    print(f"Rejected rate:", reject_count/args.n)
    print(f"Undecided rate:", undecide_count/args.n)

