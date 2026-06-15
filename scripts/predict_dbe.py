import numpy as np
from pathlib import Path
from lightgbm import Booster

MODEL_PATH = Path("output/lgb_combined_mz_nl.txt")
N_BINS = 200

def _bin_spec(mz, it, n=N_BINS):
    mat = np.zeros(n, dtype=np.float32)
    idx = np.floor(mz).astype(np.int32)
    v = (idx >= 0) & (idx < n)
    for j, iv in zip(idx[v], it[v]):
        if iv > mat[j]: mat[j] = iv
    return mat

def _bin_nl(mw, mz, it, n=N_BINS):
    mat = np.zeros(n, dtype=np.float32)
    nl = mw - mz
    idx = np.floor(nl).astype(np.int32)
    v = (idx >= 0) & (idx < n) & (nl > 0)
    for j, iv in zip(idx[v], it[v]):
        if iv > mat[j]: mat[j] = iv
    return mat

def predict(mz_str, it_str, mw):
    mz = np.array([float(x) for x in mz_str.split(",") if x.strip()])
    it = np.array([float(x) for x in it_str.split(",") if x.strip()])
    mz_vec = _bin_spec(mz, it)
    nl_vec = _bin_nl(mw, mz, it)
    for v in [mz_vec, nl_vec]:
        mx = v.max()
        if mx > 0: v[:] /= mx
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
    feat = np.array([mw, nf, bp, bpi, tic, ent],
                    dtype=np.float32)
    X = np.hstack([mz_vec, nl_vec, feat]).reshape(1, -1)
    model = Booster(model_file=str(MODEL_PATH))
    return round(model.predict(X)[0], 2)

if __name__ == "__main__":
    import sys
    if len(sys.argv) >= 4:
        dbe = predict(sys.argv[1], sys.argv[2],
                      float(sys.argv[3]))
        print("Predicted DBE: " + str(dbe))
    else:
        t = "Usage: uv run python scripts/predict_dbe.py mz_list int_list mw"
        print(t)
        print("Example: uv run python scripts/predict_dbe.py "
              '"12,13,14,15,16" "260,860,1710,8560,9999" 16.031')
