"""DBE prediction using ensemble (LGBM m/z+NL + CNN m/z).""" 
import numpy as np 
from pathlib import Path 
from lightgbm import Booster 
 
OUT = Path("output") 
N_BINS = 800 
NL_BINS = 400 
LGB_PATH = OUT / "lgb_mz_nl_model.txt" 
LGB_FALLBACK = OUT / "lgb_dbe_model.txt" 
CNN_PATH = OUT / "cnn_dbe_model.pt" 
 
def _bin_spec(mz, it, n=800): 
    mat = np.zeros(n, dtype=np.float32) 
    idx = np.floor(mz).astype(np.int32) 
    v = (idx >= 0) & (idx < n) 
    for j, iv in zip(idx[v], it[v]): 
        if iv > mat[j]: mat[j] = iv 
    mx = mat.max() 
    if mx > 0: mat /= mx 
    return mat 
 
def _bin_nl(mw, mz, it, n=400): 
    mat = np.zeros(n, dtype=np.float32) 
    nl = mw - mz 
    idx = np.floor(nl).astype(np.int32) 
    v = (idx >= 0) & (idx < n) & (nl > 0) 
    for j, iv in zip(idx[v], it[v]): 
        if iv > mat[j]: mat[j] = iv 
    mx = mat.max() 
    if mx > 0: mat /= mx 
    return mat 
 
def _feat(mz, it, mw): 
    nf = len(mz) 
    bp = mz[it.argmax()] if nf > 0 else 0 
    bpi = it.max() if nf > 0 else 0 
    tic = np.log10(it.sum() + 1) if nf > 0 else 0 
    ent = 0.0 
    if nf > 1: 
        p = it / (it.sum() + 1e-10) 
        p = p[p > 0] 
        ent = float(-np.sum(p * np.log2(p))) 
    return np.array([mw, nf, bp, bpi, tic, ent], dtype=np.float32)
 
# Lazy-loaded CNN 
_cnn = None 
 
def _load_cnn(): 
    global _cnn 
    if _cnn is not None: 
        return True 
    if not CNN_PATH.exists(): 
        return False 
    import torch 
    import torch.nn as nn 
    import torch.nn.functional as F 
    class CNN1D(nn.Module): 
        def __init__(self, nb=800): 
            super().__init__() 
            self.conv1 = nn.Conv1d(1, 32, 7, padding=3) 
            self.conv2 = nn.Conv1d(32, 64, 5, padding=2) 
            self.conv3 = nn.Conv1d(64, 128, 3, padding=1) 
            self.pool = nn.MaxPool1d(2) 
            self.fc1 = nn.Linear(128 * (nb // 8), 64) 
            self.fc2 = nn.Linear(64, 1) 
            self.drop = nn.Dropout(0.3) 
        def forward(self, x): 
            x = self.pool(F.relu(self.conv1(x))) 
            x = self.pool(F.relu(self.conv2(x))) 
            x = self.pool(F.relu(self.conv3(x))) 
            x = x.view(x.size(0), -1) 
            x = self.drop(F.relu(self.fc1(x))) 
            return self.fc2(x).squeeze() 
    _cnn = CNN1D() 
    _cnn.load_state_dict(torch.load(CNN_PATH, map_location="cpu")) 
    _cnn.eval() 
    return True 
 
def predict(mz_str, it_str, mw): 
    mz = np.array([float(x) for x in mz_str.split(",",) if x.strip()], dtype=np.float32) 
    it = np.array([float(x) for x in it_str.split(",",) if x.strip()], dtype=np.float32) 
    mz_vec = _bin_spec(mz, it) 
    nl_vec = _bin_nl(mw, mz, it) 
    feat = _feat(mz, it, mw) 
    X_lgb = np.hstack([mz_vec, nl_vec, feat]).reshape(1, -1) 
    if LGB_PATH.exists(): 
        model = Booster(model_file=str(LGB_PATH)) 
    else: 
        model = Booster(model_file=str(LGB_FALLBACK)) 
        X_lgb = np.hstack([mz_vec, feat]).reshape(1, -1) 
    pred = model.predict(X_lgb)[0] 
    if _load_cnn(): 
        import torch 
        x = torch.FloatTensor(mz_vec).unsqueeze(0).unsqueeze(0) 
        with torch.no_grad(): 
            pred_cnn = _cnn(x).item() 
        pred = 0.7 * pred + 0.3 * pred_cnn 
    return round(pred, 2) 
 
if __name__ == "__main__": 
    import sys 
    if len(sys.argv) >= 4: 
        dbe = predict(sys.argv[1], sys.argv[2], float(sys.argv[3])) 
        print("Predicted DBE:", dbe) 
    else: 
        print("Usage: uv run python scripts/predict_dbe.py mz_list int_list mw") 
        print('Example: uv run python scripts/predict_dbe.py "15,26,27,39,51,52,77" "120,340,260,1110,400,320,999" 78.0')
