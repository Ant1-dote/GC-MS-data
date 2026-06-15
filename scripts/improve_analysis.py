import pandas as pd, numpy as np, time
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import mean_absolute_error, r2_score
from lightgbm import LGBMRegressor
from sklearn.model_selection import KFold
OUT = Path("output")
t0 = time.time()
np.random.seed(42)
print("[1] Loading...")
df = pd.read_parquet(OUT / "combined.parquet")
spec_mat = np.load(OUT / "spectral_matrix.npy")
mask = df["DBE"].notna() & (df["n_fragments"] >= 3)
df = df[mask].copy(); spec_mat = spec_mat[mask.values]
print("  n=" + str(len(df)))
print("[2] PCA + KMeans...")
from sklearn.decomposition import PCA
pca = PCA(n_components=100, random_state=42)
X_pca = pca.fit_transform(spec_mat)
print("  var=" + str(round(pca.explained_variance_ratio_.sum(), 3)))
km = MiniBatchKMeans(n_clusters=50, random_state=42, batch_size=10000, n_init=3)
clusters = km.fit_predict(X_pca)
df["cluster"] = clusters
print("[3] Cluster analysis...")
global_avg = spec_mat.mean(axis=0)
print("  Cluster summary (top 5 + last 2):")
for c in range(50):
    idx = (clusters == c); n_c = idx.sum()
    if n_c == 0: continue
    sub = df[idx]
    diff = spec_mat[idx].mean(axis=0) - global_avg
    top_idx = np.argsort(diff)[-5:][::-1]
    if c < 5 or c >= 48:
        dbe_m = str(round(sub["DBE"].mean(), 1))
        mzs = str([int(top_idx[0]), int(top_idx[1])])
        print("    #" + str(c) + ": n=" + str(n_c) + " DBE=" + dbe_m + " top_mz=" + mzs)
print("  Done in " + str(round(time.time()-t0, 1)) + "s")