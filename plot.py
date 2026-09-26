import os
import glob
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from rad_embeddings.paths import default_storage_dir, parse_log_name

# Runs that share these fields are treated as seeds of the same configuration.
GROUP_KEYS = ("max_size", "n_tokens", "binary_reward", "gamma", "sampler", "p")


def load_runs(log_dir):
    """
    Load all CSVs named by rad_embeddings.paths.log_path and group them by configuration.
    Returns:
        runs_by_group = {
            (max_size, n_tokens, binary_reward, gamma, sampler, p): [df_seed1, df_seed2, ...],
            ...
        }
    """
    runs_by_group = {}

    for filepath in sorted(glob.glob(os.path.join(log_dir, "*.csv"))):
        run = parse_log_name(filepath)
        if run is None:
            print(f"Skipping unmatched file: {filepath}")
            continue

        df = pd.read_csv(filepath)
        runs_by_group.setdefault(tuple(run[k] for k in GROUP_KEYS), []).append(df)

    return runs_by_group


def group_labels(groups):
    """
    Label each group by only the fields that differ across groups (gamma if none do).
    """
    varying = [i for i in range(len(GROUP_KEYS)) if len({g[i] for g in groups}) > 1]
    varying = varying or [GROUP_KEYS.index("gamma")]
    return {g: ", ".join(f"{GROUP_KEYS[i]}={g[i]}" for i in varying) for g in groups}


def aggregate_runs(runs_by_group, labels):
    """
    Keep only completed runs (same final timestep as the longest run in each group),
    then compute mean/std across seeds.
    """
    aggregated = {}

    for group, runs in runs_by_group.items():
        # Find the maximum final timestep
        max_timestep = max(run["timestep"].iloc[-1] for run in runs)

        # Keep only runs that reached max timestep
        completed_runs = [
            run for run in runs
            if run["timestep"].iloc[-1] == max_timestep
        ]

        dropped = len(runs) - len(completed_runs)
        if dropped > 0:
            print(f"{labels[group]}: dropped {dropped} incomplete runs")

        if len(completed_runs) == 0:
            print(f"{labels[group]}: no completed runs found, skipping")
            continue

        base = completed_runs[0][["timestep"]].copy()
        metrics = [c for c in completed_runs[0].columns if c != "timestep"]

        for metric in metrics:
            values = np.stack([run[metric].values for run in completed_runs], axis=0)
            base[f"{metric}_mean"] = values.mean(axis=0)
            base[f"{metric}_std"] = values.std(axis=0)

        aggregated[group] = base

    return aggregated


def plot_metrics(aggregated, labels, output_dir):
    """
    Plot all metrics with mean ± std shading for each group.
    """
    os.makedirs(output_dir, exist_ok=True)

    sample_df = next(iter(aggregated.values()))
    metrics = sorted(set(
        col[:-5] for col in sample_df.columns
        if col.endswith("_mean") and col != "timestep"
    ))

    for metric in metrics:
        plt.figure(figsize=(8, 5))

        for group in sorted(aggregated.keys(), key=lambda g: tuple(float("-inf") if v is None else v for v in g)):
            df = aggregated[group]
            x = df["timestep"].values
            mean = df[f"{metric}_mean"].values
            std = df[f"{metric}_std"].values

            plt.plot(x, mean, label=labels[group])
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
    parser = argparse.ArgumentParser(description="Plot training curves (mean ± std over seeds) from CSV logs.")
    parser.add_argument("--log-dir", type=str, default=default_storage_dir(), help="Directory containing the CSV logs (default: the package's bundled storage)")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory for the plots (default: <log-dir>/plots)")
    args = parser.parse_args()

    runs_by_group = load_runs(args.log_dir)

    if not runs_by_group:
        print("No valid CSV files found.")
        return

    labels = group_labels(list(runs_by_group))
    aggregated = aggregate_runs(runs_by_group, labels)
    plot_metrics(aggregated, labels, args.output_dir or os.path.join(args.log_dir, "plots"))


if __name__ == "__main__":
    main()

