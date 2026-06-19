"""Compare 0.5 Da vs 1 Da binning for DBE prediction.""" 
import numpy as np, pandas as pd, time 
from pathlib import Path 
from lightgbm import LGBMRegressor 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import mean_absolute_error, r2_score 
 
OUT = Path("output") 
print("[Bin Exp] Loading data...") 
combined = pd.read_parquet(OUT / "combined.parquet") 
mask = combined["DBE"].notna() & (combined["n_fragments"] >= 3) 
df = combined[mask].copy(); n = len(df) 
print(f"  Samples: {n}") 
 
def build_05da(df, nb=1600): 
    spec = np.zeros((n, nb), dtype=np.float32) 
    for i in range(n): 
        r = df.iloc[i] 
        try: 
            mz = np.fromstring(r["mz"], sep=",", dtype=np.float32) 
            it = np.fromstring(r["intensity"], sep=",", dtype=np.float32) 
        except: 
            continue 
        idx = np.floor(mz * 2).astype(np.int32) 
        v = (idx >= 0) & (idx < nb) 
        for ji, iv in zip(idx[v], it[v]): 
            if iv > spec[i, ji]: 
                spec[i, ji] = iv 
    mx = spec.max(axis=1, keepdims=True) 
    mx[mx == 0] = 1 
    return spec / mx 
 
t0 = time.time() 
spec_05 = build_05da(df) 
print(f"  0.5 Da: {spec_05.shape} ({time.time()-t0:.1f}s)") 
 
spec_10 = np.load(OUT / "spectral_matrix.npy")[mask.values] 
feat = df[["MW","n_fragments","base_peak_mz","base_peak_int","tic_log","spectral_entropy"]].values 
y = df["DBE"].values 
X10 = np.hstack([spec_10, feat]) 
X05 = np.hstack([spec_05, feat]) 
 
(X10_tr, X10_te, X05_tr, X05_te, y_tr, y_te) = train_test_split(X10, X05, y, test_size=0.2, random_state=42) 
 
base = dict(n_estimators=300, max_depth=12, num_leaves=63, learning_rate=0.05, random_state=42, verbose=-1) 
 
for name, X_tr, X_te in [("1.0 Da", X10_tr, X10_te), ("0.5 Da", X05_tr, X05_te)]: 
    lgb = LGBMRegressor(**base).fit(X_tr, y_tr) 
    p = lgb.predict(X_te) 
    mae = mean_absolute_error(y_te, p) 
    r2 = r2_score(y_te, p) 
    print(f"  {name}: MAE={mae:.3f}, R2={r2:.4f}")
