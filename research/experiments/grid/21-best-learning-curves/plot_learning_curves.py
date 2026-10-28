"""
Learning-curve plots for the best-config runs.

One figure per run (architecture x seed), two panels sharing the epoch axis:
  top     training loss per epoch (raw, plus a rolling median)
  bottom  relative error |I - I_ref| / |I_ref| of each of the 8 integrals at the
          intermediate checkpoints; the last checkpoint is the final evaluation
          (the training script asserts they are equal) and is marked separately.

Also writes one overview figure with all runs (rows = architectures,
columns = seeds) on shared axes.

Reads every partition subdirectory of the mirror results directory and writes
PNGs to <partition>/plots/.

Usage:
  python research/experiments/grid/21-best-learning-curves/plot_learning_curves.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

from physics import reference_scipy, PARAM_SETS
from utils.paths import get_mirror_path

ARCH_ORDER = ["sechiren", "siren", "wire"]
ARCH_LABEL = {"sechiren": "SECHIREN", "siren": "SIREN", "wire": "WIRE"}

# Categorical palette, fixed order (validated set; one slot per integral).
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
          "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK = "#0b0b0b"
INK_2 = "#52514e"
INK_MUTED = "#8a8984"
GRID = "#e4e3df"
LOSS_RAW = "#b7d3f6"
LOSS_SMOOTH = "#1c5cab"

plt.rcParams.update({
    "font.size": 9,
    "axes.edgecolor": INK_MUTED,
    "axes.labelcolor": INK_2,
    "xtick.color": INK_2,
    "ytick.color": INK_2,
    "axes.titlecolor": INK,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "legend.frameon": False,
    "savefig.dpi": 200,
})


def param_label(k: int) -> str:
    a, b, m, n = PARAM_SETS[k]
    return f"I$_{k + 1}$ (a={a}, b={b}, m={m}, n={n})"


def load_refs() -> np.ndarray:
    print("Computing scipy reference integrals ...", flush=True)
    return np.array([reference_scipy(*p)[0] for p in PARAM_SETS])


def rel_errors(inter: pd.DataFrame, refs: np.ndarray) -> np.ndarray:
    vals = inter[[f"I_{k}_val" for k in range(1, 9)]].to_numpy()
    return np.abs(vals - refs) / np.abs(refs)


def draw_loss(ax, hist: pd.DataFrame, n_epochs: int):
    ax.plot(hist["epoch"], hist["loss"], color=LOSS_RAW, lw=0.6, label="loss (per epoch)")
    win = max(1, n_epochs // 100)
    smooth = hist["loss"].rolling(win, center=True, min_periods=1).median()
    ax.plot(hist["epoch"], smooth, color=LOSS_SMOOTH, lw=1.6,
            label=f"rolling median ({win} ep)")
    ax.set_yscale("log")
    ax.set_ylabel("training loss")


def draw_errors(ax, inter: pd.DataFrame, rel: np.ndarray, show_labels: bool,
                digits_axis: bool = True):
    ep = inter["epoch"].to_numpy()
    for k in range(8):
        ax.plot(ep, rel[:, k], color=SERIES[k], lw=1.2, marker="o", ms=3.5,
                mec="white", mew=0.6, label=param_label(k) if show_labels else None)
        # final evaluation = last checkpoint
        ax.plot(ep[-1], rel[-1, k], marker="D", ms=6, color=SERIES[k],
                mec=INK, mew=0.8, ls="none", zorder=5)
    ax.plot(ep, rel.mean(axis=1), color=INK, lw=1.8, ls="--",
            label="mean over 8 integrals" if show_labels else None, zorder=4)
    ax.set_yscale("log")
    ax.set_ylabel(r"relative error $|I-I_{ref}|/|I_{ref}|$")
    # right-hand reference: correct digits = -log10(rel err)
    if digits_axis:
        sec = ax.secondary_yaxis("right", functions=(lambda x: -np.log10(x),
                                                     lambda d: 10.0 ** (-d)))
        sec.yaxis.set_major_locator(FixedLocator(range(0, 12)))
        sec.yaxis.set_major_formatter(FuncFormatter(lambda d, _: f"{d:g}"))
        sec.yaxis.set_minor_locator(NullLocator())
        sec.set_ylabel("correct digits", color=INK_2)


def plot_run(arch, seed, hist, inter, rel, summary, out: Path):
    n_epochs = int(hist["epoch"].max())
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.5, 7.2), sharex=True,
                                   gridspec_kw={"height_ratios": [1, 1.3]})
    draw_loss(ax1, hist, n_epochs)
    ax1.legend(loc="upper right", fontsize=8)
    draw_errors(ax2, inter, rel, show_labels=True)
    ax2.set_xlabel("epoch")
    ax2.set_xlim(0, n_epochs * 1.03)
    ax2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3, fontsize=7.5)

    fin = rel[-1]
    fin_digits = -np.log10(fin)
    ax2.annotate(f"final (epoch {n_epochs}, ◆)\nmean rel. err {fin.mean():.2e}\n"
                 f"digits min/mean/max {fin_digits.min():.2f} / "
                 f"{fin_digits.mean():.2f} / {fin_digits.max():.2f}",
                 xy=(0.01, 0.03), xycoords="axes fraction", fontsize=7.5,
                 color=INK_2, va="bottom")

    cfg = (f"hidden={summary['hidden_sizes']}" if arch != "wire"
           else f"entry_width={int(summary['entry_width'])}")
    fig.suptitle(f"{ARCH_LABEL[arch]}  seed {seed}   ({cfg}, ω₀={summary['omega_0']:g}, "
                 f"lr={summary['lr']:g}, {n_epochs} ep, npp={int(summary['n_per_param'])})",
                 fontsize=10, color=INK, x=0.02, ha="left")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def plot_overview(runs: dict, seeds: list, out: Path):
    fig, axes = plt.subplots(2 * len(ARCH_ORDER), len(seeds),
                             figsize=(3.2 * len(seeds), 2.0 * 2 * len(ARCH_ORDER)),
                             sharey="row",
                             gridspec_kw={"height_ratios": [1, 1.2] * len(ARCH_ORDER)})
    for i, arch in enumerate(ARCH_ORDER):
        for j, seed in enumerate(seeds):
            hist, inter, rel = runs[(arch, seed)]
            n_epochs = int(hist["epoch"].max())
            a1, a2 = axes[2 * i, j], axes[2 * i + 1, j]
            draw_loss(a1, hist, n_epochs)
            draw_errors(a2, inter, rel, show_labels=(i == 0 and j == 0),
                        digits_axis=(j == len(seeds) - 1))
            a2.sharex(a1)
            a1.tick_params(labelbottom=False)
            a1.set_title(f"{ARCH_LABEL[arch]}  s{seed}", fontsize=9, loc="left")
            if j:
                a1.set_ylabel("")
                a2.set_ylabel("")
    handles, labels = axes[1, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, fontsize=8,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Learning curves and per-integral relative error at checkpoints "
                 "(◆ = final evaluation)", fontsize=11, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.04, 1, 0.98))
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def main():
    results_dir = get_mirror_path(__file__, "results")
    refs = load_refs()
    for part in sorted(p for p in results_dir.iterdir()
                       if p.is_dir() and (p / "history").is_dir()):
        summary = pd.concat(pd.read_csv(p) for p in part.glob("results_gpu*.csv"))
        plots = part / "plots"
        plots.mkdir(exist_ok=True)
        runs, seeds = {}, sorted(summary["seed"].unique())
        for arch in ARCH_ORDER:
            for seed in seeds:
                hist = pd.read_csv(part / "history" / f"history_{arch}_s{seed}.csv")
                inter = pd.read_csv(part / "intermediate"
                                    / f"intermediate_{arch}_s{seed}.csv").sort_values("epoch")
                rel = rel_errors(inter, refs)
                row = summary[(summary["arch"] == arch) & (summary["seed"] == seed)].iloc[0]
                # sanity: our reference matches the one used during the run
                assert abs(rel[-1].mean() - row["mean_rel_err"]) < 1e-6 * max(1, row["mean_rel_err"]), \
                    (arch, seed, rel[-1].mean(), row["mean_rel_err"])
                plot_run(arch, seed, hist, inter, rel, row,
                         plots / f"learning_curve_{arch}_s{seed}.png")
                runs[(arch, seed)] = (hist, inter, rel)
                print(f"  {part.name}: {arch} s{seed} done", flush=True)
        plot_overview(runs, seeds, plots / "learning_curves_overview.png")
        print(f"Plots -> {plots}")


if __name__ == "__main__":
    main()
