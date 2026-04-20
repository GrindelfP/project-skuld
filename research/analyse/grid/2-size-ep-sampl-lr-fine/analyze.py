import datetime
import pandas as pd
import matplotlib.pyplot as plt

from utils.paths import get_mirror_path

RES_DIR = get_mirror_path(__file__, "results")
CSV_PATH = RES_DIR / "results_grid.csv"

if not CSV_PATH.exists():
    raise FileNotFoundError(f"results not found {CSV_PATH}")

df = pd.read_csv(CSV_PATH)

# cast numeric columns
for col in ["hidden_size", "epochs", "n_samples", "lr",
            "mean_err", "max_err", "test_loss", "train_sec"]:
    df[col] = pd.to_numeric(df[col])

err_cols = [c for c in df.columns if c.startswith("err_")]
for c in err_cols:
    df[c] = pd.to_numeric(df[c])

# ── 1. Top-10 ─────────────────────────────────────────────────────────
print("=== Top 10 configs by mean_err ===")
top = df.nsmallest(10, "mean_err")[
    ["hidden_size", "epochs", "n_samples", "lr",
     "mean_err", "max_err", "test_loss", "train_sec"]
]
print(top.to_string(index=False))
print()

# ── 2. Heatmap hidden_size × epochs ───────────────────────────────────
pivot = (df.groupby(["hidden_size", "epochs"])["mean_err"]
           .mean()
           .unstack("epochs"))

fig, ax = plt.subplots(figsize=(8, 5))
im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn_r",
               norm=plt.matplotlib.colors.LogNorm())
ax.set_xticks(range(len(pivot.columns)))
ax.set_xticklabels(pivot.columns)
ax.set_yticks(range(len(pivot.index)))
ax.set_yticklabels(pivot.index)
ax.set_xlabel("epochs")
ax.set_ylabel("hidden_size")
ax.set_title("Mean error (log scale) — averaged over n_samples and lr")
plt.colorbar(im, ax=ax)

for i in range(len(pivot.index)):
    for j in range(len(pivot.columns)):
        val = pivot.values[i, j]
        ax.text(j, i, f"{val:.1e}", ha="center", va="center", fontsize=7)

plt.tight_layout()
plt.savefig(RES_DIR / "heatmap_hidden_epochs.png", dpi=150)
plt.show()

# ── 3. Per-integral error for the best config ─────────────────────────
best = df.loc[df["mean_err"].idxmin()]
print("=== Best config ===")
print(best[["hidden_size", "epochs", "n_samples", "lr",
            "mean_err", "max_err", "test_loss", "train_sec"]])

labels = [f"I({c[4]},{c[5]},{c[6]},{c[7]})" for c in err_cols]
vals   = [best[c] for c in err_cols]

fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(labels, vals, color="steelblue", edgecolor="navy")
ax.set_yscale("log")
ax.set_ylabel("Absolute error")
ax.set_title(f"Per-integral errors — best config "
             f"(h={int(best.hidden_size)}, ep={int(best.epochs)}, "
             f"n={int(best.n_samples)}, lr={best.lr})")
ax.tick_params(axis="x", rotation=45)
ax.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig(RES_DIR / "best_config_errors.png", dpi=150)
plt.show()


fig, ax = plt.subplots()
scatter = ax.scatter(df["train_sec"], df["mean_err"],
           c=df["hidden_size"], cmap="viridis", alpha=0.7)
ax.set_yscale("log"); ax.set_xscale("log")
ax.set_xlabel("Train time (s)"); ax.set_ylabel("Mean error")
plt.colorbar(scatter, ax=ax, label="hidden_size")
plt.savefig(RES_DIR / "pareto_time_vs_error.png", dpi=150)

best = df.loc[df["mean_err"].idxmin()]
summary = f"""# Grid Search Results — {datetime.datetime.now()}

## Setup
- Integration: I(a,b,m,n), 3-dim, parametric NNI
- Grid: hidden_size={sorted(df.hidden_size.unique().tolist())},
  epochs={sorted(df.epochs.unique().tolist())},
  n_samples={sorted(df.n_samples.unique().tolist())},
  lr={sorted(df.lr.unique().tolist())}
- Total configs: {len(df)}

## Best config
| Param | Value |
|-------|-------|
| hidden_size | {int(best.hidden_size)} |
| epochs | {int(best.epochs)} |
| n_samples | {int(best.n_samples)} |
| lr | {best.lr} |
| mean_err | {best.mean_err:.3e} |
| max_err | {best.max_err:.3e} |
| train_sec | {best.train_sec:.1f} |
"""
with open(RES_DIR / "summary.md", "a") as f:
    f.write(summary)
