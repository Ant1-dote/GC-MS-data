"""DBE prediction from GC-MS spectra using trained LightGBM.""" 
import numpy as np 
from pathlib import Path 
from lightgbm import Booster 
 
MODEL_PATH = Path("output/lgb_dbe_model.txt") 
N_BINS = 800 
 
def _bin_spec(mz, it, n=800): 
    """Bin m/z values into 1 Da resolution histogram.""" 
    mat = np.zeros(n, dtype=np.float32) 
    idx = np.floor(mz).astype(np.int32) 
    v = (idx >= 0) & (idx < n) 
    for j, iv in zip(idx[v], it[v]): 
        if iv > mat[j]: 
            mat[j] = iv 
    mx = mat.max() 
    if mx > 0: 
        mat /= mx 
    return mat 
 
def predict(mz_str, it_str, mw): 
    mz = np.array([float(x) for x in mz_str.split(",") if x.strip()]) 
    it = np.array([float(x) for x in it_str.split(",") if x.strip()]) 
    mz_vec = _bin_spec(mz, it) 
    nf = len(mz) 
    bp = mz[it.argmax()] if len(it) > 0 else 0 
    bpi = it.max() if len(it) > 0 else 0 
    tic = np.log10(it.sum() + 1) if len(it) > 0 else 0 
    eps = 1e-10 
    ent = 0.0 
    if nf > 1: 
        p = mz_vec / (mz_vec.sum() + eps) 
        p = p[p > 0] 
        ent = float(-np.sum(p * np.log2(p))) 
    feat = np.array([mw, nf, bp, bpi, tic, ent], dtype=np.float32) 
    X = np.hstack([mz_vec, feat]).reshape(1, -1) 
    model = Booster(model_file=str(MODEL_PATH)) 
    return round(model.predict(X)[0], 2) 
 
if __name__ == "__main__": 
    import sys 
    if len(sys.argv) >= 4: 
        dbe = predict(sys.argv[1], sys.argv[2], float(sys.argv[3])) 
        print("Predicted DBE:", dbe) 
    else: 
        t = "uv run python scripts/predict_dbe.py mz_list int_list mw" 
        print(t) 
        print('Example: uv run python scripts/predict_dbe.py "15,26,27,39,51,52,77" "120,340,260,1110,400,320,999" 78.0')
