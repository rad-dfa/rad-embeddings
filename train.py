import os
import argparse
from rad_embeddings import EncoderModule

parser = argparse.ArgumentParser(description="Run RAD embedding experiment with configurable parameters.")
parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
parser.add_argument("--gamma", type=float, default=0.9, help="Discount factor for rewards")
parser.add_argument("--n-states", type=int, default=10, help="Number of states in the chain DFA")
parser.add_argument("--n-tokens", type=int, default=10, help="Number of tokens in the chain DFA")
parser.add_argument("--save-dir", type=str, default="rad_embeddings/storage", help="Directory to save training results")
parser.add_argument("--binary-reward", action="store_true", help="Use binary rewards in the environment and encoder")
parser.add_argument("--debug", action="store_true", help="Enable debug mode for training.")

args = parser.parse_args()

os.makedirs(args.save_dir, exist_ok=True)

EncoderModule.train(
    seed=args.seed,
    max_size=args.n_states,
    n_tokens=args.n_tokens,
    save_dir=args.save_dir,
    log=f"{args.save_dir}/log_n_states_{args.n_states}_n_tokens_{args.n_tokens}_binary_reward_{args.binary_reward}_gamma_{args.gamma}_seed_{args.seed}.csv",
    gamma=args.gamma,
    binary_reward=args.binary_reward,
    debug=args.debug
)

