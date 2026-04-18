import os
import re
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def parse_filename(filename):
    """
    Extract gamma and seed from filenames like:
    log_n_states_10_n_tokens_10_binary_reward_True_gamma_0.99_seed_42.csv
    """
    pattern = r"gamma_([0-9.]+)_seed_([0-9]+)\.csv"
    match = re.search(pattern, os.path.basename(filename))
    if match:
        gamma = float(match.group(1))
        seed = int(match.group(2))
        return gamma, seed
    return None, None


def load_runs(log_dir):
    """
    Load all CSVs and group by gamma.
    Returns:
        runs_by_gamma = {
            gamma1: [df_seed1, df_seed2, ...],
            gamma2: [...]
        }
    """
    runs_by_gamma = {}

    for filepath in glob.glob(os.path.join(log_dir, "*.csv")):
        gamma, seed = parse_filename(filepath)
        if gamma is None:
            print(f"Skipping unmatched file: {filepath}")
            continue

        df = pd.read_csv(filepath)
        runs_by_gamma.setdefault(gamma, []).append(df)

    return runs_by_gamma


def aggregate_runs(runs_by_gamma):
    """
    Keep only completed runs (same final timestep as the longest run for each gamma),
    then compute mean/std across seeds.
    """
    aggregated = {}

    for gamma, runs in runs_by_gamma.items():
        # Find the maximum final timestep
        max_timestep = max(run["timestep"].iloc[-1] for run in runs)

        # Keep only runs that reached max timestep
        completed_runs = [
            run for run in runs
            if run["timestep"].iloc[-1] == max_timestep
        ]

        dropped = len(runs) - len(completed_runs)
        if dropped > 0:
            print(f"Gamma {gamma}: dropped {dropped} incomplete runs")

        if len(completed_runs) == 0:
            print(f"Gamma {gamma}: no completed runs found, skipping")
            continue

        base = completed_runs[0][["timestep"]].copy()
        metrics = [c for c in completed_runs[0].columns if c != "timestep"]

        for metric in metrics:
            values = np.stack([run[metric].values for run in completed_runs], axis=0)
            base[f"{metric}_mean"] = values.mean(axis=0)
            base[f"{metric}_std"] = values.std(axis=0)

        aggregated[gamma] = base

    return aggregated


def plot_metrics(aggregated, output_dir="plots"):
    """
    Plot all metrics with mean ± std shading for each gamma.
    """
    os.makedirs(output_dir, exist_ok=True)

    sample_df = next(iter(aggregated.values()))
    metrics = sorted(set(
        col[:-5] for col in sample_df.columns
        if col.endswith("_mean") and col != "timestep"
    ))

    for metric in metrics:
        plt.figure(figsize=(8, 5))

        for gamma in sorted(aggregated.keys()):
            df = aggregated[gamma]
            x = df["timestep"].values
            mean = df[f"{metric}_mean"].values
            std = df[f"{metric}_std"].values

            plt.plot(x, mean, label=f"gamma={gamma}")
            plt.fill_between(x, mean - std, mean + std, alpha=0.2)

        plt.xlabel("Timestep")
        plt.ylabel(metric)
        plt.title(f"{metric} vs Timestep")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        save_path = os.path.join(output_dir, f"{metric}.png")
        plt.savefig(save_path)
        plt.close()

        print(f"Saved {save_path}")


def main():
    log_dir = "storage"  # change this to your log directory
    runs_by_gamma = load_runs(log_dir)

    if not runs_by_gamma:
        print("No valid CSV files found.")
        return

    aggregated = aggregate_runs(runs_by_gamma)
    plot_metrics(aggregated)


if __name__ == "__main__":
    main()

