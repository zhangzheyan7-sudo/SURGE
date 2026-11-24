from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_history(history: list[dict[str, Any]], path: str | Path) -> None:
    """Four panels for compact records and older full histories."""
    if not history:
        return
    epochs = [row["epoch"] for row in history]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    colors = {"train": "#0072B2", "val": "#D55E00"}

    def plot_present(ax, key, label, style="-", color=None, multiplier=1.0):
        values = [multiplier * row.get(key, float("nan")) for row in history]
        if any(math.isfinite(value) for value in values):
            ax.plot(epochs, values, style, label=label, color=color)

    for split in ("train", "val"):
        plot_present(axes[0, 0], f"{split}_loss", split, color=colors[split])
        plot_present(axes[0, 1], f"{split}_epe", f"{split}/final", color=colors[split])
        plot_present(axes[0, 1], f"{split}_geometry_epe", f"{split}/geometry", "--", colors[split])
        legacy = next((name for name in ("geometry_epe_batch", "recovered_affine_epe")
                       if any(f"{split}_{name}" in row for row in history)), None)
        if legacy and not any(f"{split}_geometry_epe" in row for row in history):
            plot_present(axes[0, 1], f"{split}_{legacy}", f"{split}/geometry (legacy batch)", ":", colors[split])
        for threshold, color in zip((1, 2, 3, 5), ("#0072B2", "#E69F00", "#009E73", "#CC79A7")):
            plot_present(axes[1, 0], f"{split}_cmr_{threshold}px", f"{split}/CMR@{threshold}",
                         "-" if split == "train" else "--", color, 100.0)
    lr_keys = sorted({key for row in history for key in row
                      if key.startswith("lr_") and not key.startswith("lr_floor_")})
    for key in lr_keys:
        plot_present(axes[1, 1], key, key.removeprefix("lr_"))
        floor_key = f"lr_floor_{key.removeprefix('lr_')}"
        if any(row.get(floor_key, 0) > 0 for row in history):
            plot_present(axes[1, 1], floor_key, f"{key.removeprefix('lr_')} floor", ":")
    axes[0, 0].set_title("Total training loss")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 1].set_title("Final output and global geometry EPE")
    axes[0, 1].set_ylabel("EPE (pixels; lower is better)")
    axes[1, 0].set_title("CMR within pixel threshold")
    axes[1, 0].set_ylabel("CMR (%)")
    axes[1, 0].set_ylim(0, 100)
    axes[1, 1].set_title("Learning rate by parameter group")
    axes[1, 1].set_ylabel("Learning rate")
    if lr_keys:
        axes[1, 1].set_yscale("log")
    for ax in axes.flat:
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.22)
        if ax.lines:
            ax.legend(fontsize=8, ncol=2)
    previous = None
    for row in history:
        signature = (row.get("output_affine_global", 0), row.get("output_revision", 0))
        if previous is not None and signature != previous:
            for ax in axes.flat:
                ax.axvline(row["epoch"] - 0.5, color="0.4", linestyle=":", alpha=0.6)
        previous = signature
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + ".tmp" + path.suffix)
    fig.savefig(temporary, dpi=160)
    plt.close(fig)
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot compact or legacy training history")
    parser.add_argument("--history", required=True)
    parser.add_argument("--output", default="", help="Defaults to training_curves.png beside the history file")
    args = parser.parse_args()
    history_path = Path(args.history)
    output_path = Path(args.output) if args.output else history_path.parent / "training_curves.png"
    plot_history(json.loads(history_path.read_text(encoding="utf-8")), output_path)


if __name__ == "__main__":
    main()
