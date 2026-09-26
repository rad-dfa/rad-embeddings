import inspect
import argparse
from rad_embeddings import EncoderModule
from rad_embeddings.paths import checkpoint_path, default_storage_dir, log_path

# Single source of truth for defaults; training hyperparameters are intentionally not exposed.
defaults = {name: p.default for name, p in inspect.signature(EncoderModule.train).parameters.items()}

parser = argparse.ArgumentParser(
    description="Pretrain a RAD encoder. Training hyperparameters are fixed to EncoderModule.train's defaults.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument("--seed", type=int, default=defaults["seed"], help="Random seed for reproducibility")
parser.add_argument("--max-size", "--n-states", dest="max_size", type=int, default=defaults["max_size"], help="Maximum number of DFA states")
parser.add_argument("--n-tokens", type=int, default=defaults["n_tokens"], help="Number of tokens in the DFA alphabet")
parser.add_argument("--binary-reward", action=argparse.BooleanOptionalAction, default=defaults["binary_reward"], help="Use binary rewards in the bisimulation game")
parser.add_argument("--gamma", type=float, default=defaults["gamma"], help="Discount factor (part of the run name, used by Encoder to select checkpoints)")
parser.add_argument("--save-dir", type=str, default=default_storage_dir(), help="Directory for the checkpoint and CSV log")
parser.add_argument("--no-log", action="store_true", help="Don't write the CSV training log")
parser.add_argument("--wandb", dest="enable_wandb", action=argparse.BooleanOptionalAction, default=defaults["enable_wandb"], help="Log to Weights & Biases")
parser.add_argument("--wandb-entity", type=str, default=defaults["wandb_entity"], help="W&B entity")
parser.add_argument("--wandb-project", type=str, default=defaults["wandb_project"], help="W&B project")
parser.add_argument("--debug", action="store_true", default=defaults["debug"], help="Print parameter shapes and per-update metrics")
parser.add_argument("--overwrite", action="store_true", default=defaults["overwrite"], help="Replace an existing checkpoint and log for this run")
parser.add_argument("--experimental", action="store_true", default=defaults["experimental"], help="Use the antisymmetric MLP policy head: logits = mlp(feat_l - feat_r) - mlp(feat_r - feat_l)")

args = parser.parse_args()

run = dict(
    max_size=args.max_size,
    n_tokens=args.n_tokens,
    seed=args.seed,
    binary_reward=args.binary_reward,
    gamma=args.gamma,
    experimental=args.experimental,
)

print(f"Checkpoint: {checkpoint_path(args.save_dir, **run)}")
if not args.no_log:
    print(f"Log: {log_path(args.save_dir, **run)}")

EncoderModule.train(
    **run,
    save_dir=args.save_dir,
    log=False if args.no_log else None,
    enable_wandb=args.enable_wandb,
    wandb_entity=args.wandb_entity,
    wandb_project=args.wandb_project,
    debug=args.debug,
    overwrite=args.overwrite,
)
