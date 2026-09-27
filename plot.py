import os
import glob
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from rad_embeddings.paths import config_name, default_storage_dir, logs_dir, parse_log_name, plots_dir

# Light-mode colors from the dataviz reference palette
SERIES = "#2a78d6"
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"


def load_runs(log_dir):
    """
    Load all CSVs named by rad_embeddings.paths.log_path and group them by configuration (everything but the seed).
    Returns:
        runs_by_config = {
            config_name: (run_without_seed, [df_seed1, df_seed2, ...]),
            ...
        }
    """
    runs_by_config = {}

    for filepath in sorted(glob.glob(os.path.join(log_dir, "*.csv"))):
        run = parse_log_name(filepath)
        if run is None:
            print(f"Skipping unmatched file: {filepath}")
            continue

        df = pd.read_csv(filepath)
        run.pop("seed")
        runs_by_config.setdefault(config_name(**run), (run, []))[1].append(df)

    return runs_by_config


def aggregate_runs(name, runs):
    """
    Keep only completed runs (same final timestep as the longest run of the config),
    then compute mean/std across seeds. Returns (aggregated df, number of seeds kept), or None.
    """
    max_timestep = max(run["timestep"].iloc[-1] for run in runs)
    completed_runs = [run for run in runs if run["timestep"].iloc[-1] == max_timestep]

    dropped = len(runs) - len(completed_runs)
    if dropped > 0:
        print(f"{name}: dropped {dropped} incomplete runs")

    base = completed_runs[0][["timestep"]].copy()
    metrics = [c for c in completed_runs[0].columns if c != "timestep"]

    for metric in metrics:
        values = np.stack([run[metric].values for run in completed_runs], axis=0)
        base[f"{metric}_mean"] = values.mean(axis=0)
        base[f"{metric}_std"] = values.std(axis=0)

    return base, len(completed_runs)


def plot_config(df, n_seeds, run, output_dir):
    """
    One line chart per metric (mean ± std over seeds), all saved in output_dir.
    """
    os.makedirs(output_dir, exist_ok=True)

    metrics = [col[:-5] for col in df.columns if col.endswith("_mean")]
    seeds = "1 seed" if n_seeds == 1 else f"mean ± std over {n_seeds} seeds"
    subtitle = (
        f"sampler {run['sampler']} · p {run['p']} · max_size {run['max_size']} · n_tokens {run['n_tokens']} · "
        f"binary_reward {run['binary_reward']} · gamma {run['gamma']} · {seeds}"
    )
    x = df["timestep"].values

    for metric in metrics:
        mean = df[f"{metric}_mean"].values
        std = df[f"{metric}_std"].values

        fig, ax = plt.subplots(figsize=(8, 5), facecolor=SURFACE)
        ax.set_facecolor(SURFACE)

        ax.fill_between(x, mean - std, mean + std, color=SERIES, alpha=0.1, linewidth=0)
        ax.plot(x, mean, color=SERIES, linewidth=2, solid_joinstyle="round", solid_capstyle="round")

        # Label only the final value, at the end of the line
        ax.plot(x[-1], mean[-1], "o", color=SERIES, markersize=8, markeredgecolor=SURFACE, markeredgewidth=2)
        final = f"{mean[-1]:,.0f}" if abs(mean[-1]) >= 1000 else f"{mean[-1]:.3g}"
        ax.annotate(final, (x[-1], mean[-1]), xytext=(8, 0), textcoords="offset points",
                    va="center", color=INK_SECONDARY, fontsize=9)

        ax.grid(True, color=GRID, linewidth=0.75, linestyle="-")
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
        ax.tick_params(colors=GRID, labelcolor=INK_MUTED, labelsize=9)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
        # No offset/exponent label: it would sit on top of the subtitle
        ax.ticklabel_format(axis="y", style="plain", useOffset=False)
        ax.set_xlabel("Timestep", color=INK_SECONDARY, fontsize=10)

        ax.set_title(metric, loc="left", color=INK, fontsize=13, fontweight="bold", pad=22)
        ax.text(0, 1.02, subtitle, transform=ax.transAxes, color=INK_SECONDARY, fontsize=8.5, va="bottom")

        fig.savefig(os.path.join(output_dir, f"{metric}.png"), facecolor=SURFACE, bbox_inches="tight", dpi=100)
        plt.close(fig)

    print(f"Saved {len(metrics)} plots to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Plot training curves (mean ± std over seeds) from CSV logs, one folder per configuration.")
    parser.add_argument("--storage-dir", type=str, default=default_storage_dir(), help="Storage directory: reads <storage-dir>/logs, writes <storage-dir>/plots with one subfolder per configuration (default: the package's bundled storage)")
    args = parser.parse_args()
    output_dir = plots_dir(args.storage_dir)

    runs_by_config = load_runs(logs_dir(args.storage_dir))

    if not runs_by_config:
        print("No valid CSV files found.")
        return

    for name, (run, runs) in runs_by_config.items():
        df, n_seeds = aggregate_runs(name, runs)
        plot_config(df, n_seeds, run, os.path.join(output_dir, name))


if __name__ == "__main__":
    main()
