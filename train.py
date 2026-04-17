import os
import argparse
from rad_embeddings import EncoderModule

parser = argparse.ArgumentParser(description="Run RAD embedding experiment with configurable parameters.")
parser.add_argument('--n-states', type=int, default=10, help='Number of states in the chain DFA')
parser.add_argument('--n-tokens', type=int, default=10, help='Number of tokens in the chain DFA')
parser.add_argument('--save-dir', type=str, default="storage", help='Directory to save training results')

args = parser.parse_args()

os.makedirs(args.save_dir, exist_ok=True)

seeds = [42, 0, 1, 2, 3, 4]
gammas = [0.99, 0.9, 0.5, 0.1, 0.01]

for seed in seeds:
    for gamma in gammas:
        EncoderModule.train(
            seed=seed,
            max_size=args.n_states,
            n_tokens=args.n_tokens,
            save_dir=args.save_dir,
            log=f"{args.save_dir}/log_n_states_{args.n_states}_n_tokens_{args.n_tokens}_seed_{seed}_gamma_{gamma}.csv",
            gamma=gamma
        )

