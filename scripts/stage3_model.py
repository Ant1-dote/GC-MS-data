import pandas as pd, numpy as np
import time
from pathlib import Path
OUT = Path("output")
t0 = time.time()
print("[Stage 3a] Loading...")
combined = pd.read_parquet(OUT / "combined.parquet")
print("  Shape: " + str(combined.shape))
print("[Stage 3b] Preparing data...")
mask = combined["DBE"].notna()
mask &= combined["n_fragments"] >= 3
model_df = combined[mask].copy()
print("  Valid: " + str(len(model_df)))
spec_mat = np.load(OUT / "spectral_matrix.npy")
spec_mat = spec_mat[mask.values]
print("  Spec: " + str(spec_mat.shape))
X_feat = model_df[["MW","n_fragments","base_peak_mz","base_peak_int","tic_log","spectral_entropy"]].values
X = np.hstack([spec_mat, X_feat])
y_dbe = model_df["DBE"].values
print("  X: " + str(X.shape))
print("[Stage 3c] PCA + Clustering...")
from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans
print("  PCA(100)...")
pca = PCA(n_components=100, random_state=42)
X_pca = pca.fit_transform(spec_mat)
evr = pca.explained_variance_ratio_.sum()
print("  PCA var: " + str(round(evr,3)))
print("  KMeans(k=50)...")
km = MiniBatchKMeans(n_clusters=50, random_state=42, batch_size=10000, n_init=3)
clusters = km.fit_predict(X_pca)
model_df = model_df.copy()
model_df["cluster"] = clusters
print("  Clusters: " + str(len(set(clusters))))
print("[Stage 4a] LightGBM...")
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
X_tr, X_te, y_tr, y_te = train_test_split(X, y_dbe, test_size=0.2, random_state=42)
print("  Training DBE model...")
lgb = LGBMRegressor(n_estimators=300, max_depth=12, num_leaves=63, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, min_child_samples=20, random_state=42, verbose=-1)
lgb.fit(X_tr, y_tr)
y_pred = lgb.predict(X_te)
mae = mean_absolute_error(y_te, y_pred)
r2 = r2_score(y_te, y_pred)
print("  MAE: " + str(round(mae,2)) + ", R2: " + str(round(r2,3)))
lgb.booster_.save_model(str(OUT / "lgb_dbe_model.txt"))
print("[Stage 4b] SHAP analysis...")
import shap
n_samp = min(3000, len(X_te))
np.random.seed(42)
idx = np.random.choice(len(X_te), n_samp, replace=False)
X_sam = X_te[idx]
print("  Samples: " + str(n_samp))
explainer = shap.TreeExplainer(lgb)
shap_vals = explainer.shap_values(X_sam)
n_spec = spec_mat.shape[1]
shap_spec = shap_vals[:, :n_spec]
shap_feat = shap_vals[:, n_spec:]
np.save(OUT / "shap_values.npy", shap_vals)
np.save(OUT / "shap_base.npy", np.array([explainer.expected_value]))
imp = pd.DataFrame()
imp["feature"] = ["mz_"+str(i) for i in range(n_spec)] + ["MW","n_frags","bp_mz","bp_int","tic","entropy"]
imp["importance"] = lgb.feature_importances_
imp.to_parquet(OUT / "feature_importance.parquet")
spec_imp = imp.iloc[:n_spec]
top20 = spec_imp.nlargest(20, "importance")
print("  Top m/z bins:")
for _, row in top20.iterrows():
    print("    m/z " + row["feature"].replace("mz_","").rjust(3) + "  imp=" + str(int(row["importance"])))
feat_names = ["MW","n_frags","bp_mz","bp_int","tic","entropy"]
feat_imp = np.abs(shap_feat).mean(axis=0)
print("  Spectral shape SHAP:")
for i, name in enumerate(feat_names):
    print("    " + name + ": " + str(round(feat_imp[i], 4)))
print("Done! Time: " + str(round(time.time()-t0, 1)) + "s")
