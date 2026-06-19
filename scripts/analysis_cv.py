"""""Cross-validation analysis for DBE prediction. 5-fold CV (m/z model) + 5-fold CV (m/z+NL model) + LOCO validation.""" 
import numpy as np, pandas as pd, time, warnings 
from pathlib import Path 
from lightgbm import LGBMRegressor 
from sklearn.model_selection import KFold, cross_val_score 
from sklearn.metrics import mean_absolute_error, r2_score 
warnings.filterwarnings('ignore') 
 
OUT = Path("output") 
t0 = time.time() 
 
print("[1] Loading data...") 
combined = pd.read_parquet(OUT / "combined.parquet") 
mask = combined["DBE"].notna() & (combined["n_fragments"] >= 3) 
df = combined[mask].copy(); n = len(df) 
print(f"  Valid: {n}") 
 
spec_mz = np.load(OUT / "spectral_matrix.npy")[mask.values] 
feat = df[["MW","n_fragments","base_peak_mz","base_peak_int","tic_log","spectral_entropy"]].values 
y = df["DBE"].values 
 
# === Build NL matrix === 
def build_nl(df, nb=400): 
    spec = np.zeros((n, nb), dtype=np.float32) 
    for i in range(n): 
        r = df.iloc[i] 
        try: 
            mz = np.fromstring(r["mz"], sep=",", dtype=np.float32) 
            it = np.fromstring(r["intensity"], sep=",", dtype=np.float32) 
        except: 
            continue 
        nl = r["MW"] - mz 
        idx = np.floor(nl).astype(np.int32) 
        v = (idx >= 0) & (idx < nb) & (nl > 0) 
        for ji, iv in zip(idx[v], it[v]): 
            if iv > spec[i, ji]: 
                spec[i, ji] = iv 
    mx = spec.max(axis=1, keepdims=True) 
    mx[mx == 0] = 1 
    return spec / mx 
 
print("[2] Building NL matrix...") 
spec_nl = build_nl(df) 
print(f"  NL: {spec_nl.shape}") 
 
X_mz = np.hstack([spec_mz, feat]) 
X_mz_nl = np.hstack([spec_mz, spec_nl, feat])
 
# === 5-fold CV (m/z model) === 
print("\n[3] 5-fold CV (m/z model)...") 
lgb = LGBMRegressor(n_estimators=300, max_depth=12, num_leaves=63, learning_rate=0.05, random_state=42, verbose=-1) 
cv5 = KFold(n_splits=5, shuffle=True, random_state=42) 
cv_mae = -cross_val_score(lgb, X_mz, y, cv=cv5, scoring='neg_mean_absolute_error') 
cv_r2 = cross_val_score(lgb, X_mz, y, cv=cv5, scoring='r2') 
print(f"  m/z: MAE={cv_mae.mean():.3f}+/-{cv_mae.std():.3f}, R2={cv_r2.mean():.4f}+/-{cv_r2.std():.4f}") 
 
# === 5-fold CV (m/z+NL model) === 
print("\n[4] 5-fold CV (m/z+NL model)...") 
lgb_nl = LGBMRegressor(n_estimators=300, max_depth=12, num_leaves=63, learning_rate=0.05, random_state=42, verbose=-1) 
cv_mae_nl = -cross_val_score(lgb_nl, X_mz_nl, y, cv=cv5, scoring='neg_mean_absolute_error') 
cv_r2_nl = cross_val_score(lgb_nl, X_mz_nl, y, cv=cv5, scoring='r2') 
print(f"  m/z+NL: MAE={cv_mae_nl.mean():.3f}+/-{cv_mae_nl.std():.3f}, R2={cv_r2_nl.mean():.4f}+/-{cv_r2_nl.std():.4f}") 
 
# === Save m/z+NL model === 
print("\n[5] Saving m/z+NL model...") 
lgb_full = LGBMRegressor(n_estimators=300, max_depth=12, num_leaves=63, learning_rate=0.05, random_state=42, verbose=-1) 
lgb_full.fit(X_mz_nl, y) 
lgb_full.booster_.save_model(str(OUT / "lgb_mz_nl_model.txt")) 
print("  Saved: lgb_mz_nl_model.txt") 
 
# === LOCO === 
print("\n[6] LOCO validation...") 
from sklearn.decomposition import PCA 
from sklearn.cluster import MiniBatchKMeans 
pca = PCA(n_components=100, random_state=42) 
X_pca = pca.fit_transform(spec_mz) 
km = MiniBatchKMeans(n_clusters=50, random_state=42, batch_size=10000, n_init=3) 
clusters = km.fit_predict(X_pca) 
df_cl = df.copy() 
df_cl["cluster"] = clusters
loco_mae, loco_r2 = [], [] 
for c in sorted(df_cl["cluster"].unique()): 
    train_mask = df_cl["cluster"] != c 
    test_mask = df_cl["cluster"] == c 
    if test_mask.sum() < 10: 
        continue 
    X_tr, X_te = X_mz[train_mask.values], X_mz[test_mask.values] 
    y_tr, y_te = y[train_mask.values], y[test_mask.values] 
    l = LGBMRegressor(n_estimators=300, max_depth=12, num_leaves=63, learning_rate=0.05, random_state=42, verbose=-1) 
    l.fit(X_tr, y_tr) 
    p = l.predict(X_te) 
    loco_mae.append(mean_absolute_error(y_te, p)) 
    loco_r2.append(r2_score(y_te, p)) 
    if c  < 3 or c >= 48: 
        print(f"  #{c}: n_tst={test_mask.sum():5d} MAE={loco_mae[-1]:.3f} R2={loco_r2[-1]:.4f}") 
 
print(f"\nLOCO: {len(loco_mae)} clusters, MAE={np.mean(loco_mae):.3f}+/-{np.std(loco_mae):.3f}, R2={np.mean(loco_r2):.4f}+/-{np.std(loco_r2):.4f}") 
 
# Summary 
print("\n" + "="*50) 
print("CV Analysis Results") 
print("="*50) 
print(f"m/z     CV: MAE={cv_mae.mean():.3f} R2={cv_r2.mean():.4f}") 
print(f"m/z+NL  CV: MAE={cv_mae_nl.mean():.3f} R2={cv_r2_nl.mean():.4f}") 
print(f"LOCO       : MAE={np.mean(loco_mae):.3f} R2={np.mean(loco_r2):.4f}") 
print(f"Time: {time.time()-t0:.1f}s")
