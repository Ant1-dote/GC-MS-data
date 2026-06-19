"""SHAP explainability: summary + waterfall plots."""
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
OUT = Path("output")
plt.rcParams.update({"font.size":11,"axes.titlesize":13})
shap_vals = np.load(OUT/"shap_values.npy")
expected = float(np.load(OUT/"shap_base.npy")[0])
imp = pd.read_parquet(OUT/"feature_importance.parquet")
feat_names = imp["feature"].tolist()
# === Top-25 bar ===
print("[1/3] Top-25 SHAP bar...")
mean_shap = np.abs(shap_vals).mean(axis=0)
top_idx = np.argsort(mean_shap)[::-1][:25]
fig, ax = plt.subplots(figsize=(10,7))
ax.barh(range(25), mean_shap[top_idx][::-1], color="#2196F3", height=0.7, edgecolor="white")
ax.set_yticks(range(25))
ax.set_yticklabels([feat_names[i] for i in top_idx[::-1]], fontsize=8)
ax.set_xlabel("Mean |SHAP|")
ax.set_title("Top 25 features by mean |SHAP|")
plt.tight_layout()
fig.savefig(OUT/"shap_bar_top25.png", dpi=150, bbox_inches="tight")
plt.close()
print("  saved: shap_bar_top25.png")
# === Beeswarm (top 20) ===
print("[2/3] Beeswarm summary...")
top20 = top_idx[:20]
rng = np.random.RandomState(42)
fig, ax = plt.subplots(figsize=(10,8))
for i in range(20):
    vals = shap_vals[:, top20[i]]
    jitter = rng.uniform(-0.3,0.3,len(vals))
    ax.scatter(vals, 19-i+jitter, s=2, alpha=0.12, c="#2196F3", edgecolors="none")
ax.set_yticks(range(20))
ax.set_yticklabels([feat_names[i] for i in top20], fontsize=8)
ax.axvline(0, color="gray", ls="--", lw=0.6)
ax.set_xlabel("SHAP value")
ax.set_title("Top 20 SHAP features (beeswarm)")
plt.tight_layout()
fig.savefig(OUT/"shap_beeswarm.png", dpi=150, bbox_inches="tight")
plt.close()
print("  saved: shap_beeswarm.png")
# === Waterfall (9 samples) ===
print("[3/3] Waterfall grid (9 samples)...")
idx_samp = rng.choice(len(shap_vals),9,replace=False)
fig, axes = plt.subplots(3,3,figsize=(18,15))
for ax_i, sidx in enumerate(idx_samp):
    r,c = divmod(ax_i,3)
    ax = axes[r,c]
    row = shap_vals[sidx]
    order = np.argsort(np.abs(row))[::-1][:15]
    vals = row[order]
    names = [feat_names[i] for i in order]
    cum = np.concatenate([[expected], expected+np.cumsum(vals)])
    colors = ["#e74c3c" if v>0 else "#3498db" for v in vals]
    for i in range(len(order)):
        ax.barh(i, vals[i], left=cum[i], color=colors[i], height=0.7, edgecolor="white", linewidth=0.3)
    ax.axvline(expected, color="gray", ls="--", lw=0.8, alpha=0.5)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(names, fontsize=7)
    ax.set_xlabel("DBE", fontsize=8)
    ax.set_title(f"Sample {sidx}", fontsize=10)
plt.suptitle(f"SHAP Waterfall (base={expected:.2f})", fontsize=14, y=1.01)
plt.tight_layout()
fig.savefig(OUT/"shap_waterfall_grid.png", dpi=150, bbox_inches="tight")
plt.close()
print("  saved: shap_waterfall_grid.png")
print("All SHAP plots done.")
