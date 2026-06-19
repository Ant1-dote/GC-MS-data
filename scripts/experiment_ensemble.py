"""Ensemble: LGBM (m/z+NL) + CNN (m/z) weighted average.""" 
import numpy as np, pandas as pd, time, warnings 
from pathlib import Path 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import mean_absolute_error, r2_score 
import torch 
import torch.nn as nn 
import torch.nn.functional as F 
warnings.filterwarnings("ignore") 
 
OUT = Path("output") 
print("[Ensemble] Loading data...") 
combined = pd.read_parquet(OUT / "combined.parquet") 
mask = combined["DBE"].notna() & (combined["n_fragments"] >= 3) 
df = combined[mask].copy(); n = len(df) 
print(f"  n={n}") 
 
spec = np.load(OUT / "spectral_matrix.npy")[mask.values] 
feat = df[["MW","n_fragments","base_peak_mz","base_peak_int","tic_log","spectral_entropy"]].values 
y = df["DBE"].values 
 
# Build NL matrix 
def build_nl(df, nb=400): 
    s = np.zeros((n, nb), dtype=np.float32) 
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
            if iv > s[i, ji]: 
                s[i, ji] = iv 
    mx = s.max(axis=1, keepdims=True) 
    mx[mx == 0] = 1 
    return s / mx
 
print("Building NL matrix...") 
spec_nl = build_nl(df) 
print(f"  NL: {spec_nl.shape}") 
 
X_mn = np.hstack([spec, spec_nl, feat]) 
(X_tr, X_te, Xm_tr, Xm_te, y_tr, y_te) = train_test_split(X_mn, spec, y, test_size=0.2, random_state=42) 
 
# === LGBM m/z+NL === 
from lightgbm import LGBMRegressor 
print("\nTraining LGBM...") 
lgb = LGBMRegressor(n_estimators=300, max_depth=12, num_leaves=63, learning_rate=0.05, random_state=42, verbose=-1) 
lgb.fit(X_tr, y_tr) 
p_lgb = lgb.predict(X_te) 
m_lgb = mean_absolute_error(y_te, p_lgb) 
r_lgb = r2_score(y_te, p_lgb) 
print(f"  LGBM m/z+NL: MAE={m_lgb:.3f}, R2={r_lgb:.4f}") 
 
# === CNN m/z === 
class CNN1D(nn.Module): 
    def __init__(self, nb=800): 
        super().__init__() 
        self.conv1 = nn.Conv1d(1, 32, kernel_size=7, padding=3) 
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2) 
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1) 
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
 
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
model = CNN1D().to(dev) 
opt = torch.optim.Adam(model.parameters(), lr=1e-3) 
 
X_tr_t = torch.FloatTensor(Xm_tr).unsqueeze(1).to(dev) 
X_te_t = torch.FloatTensor(Xm_te).unsqueeze(1).to(dev) 
y_tr_t = torch.FloatTensor(y_tr).to(dev) 
 
print("Training CNN...") 
t0 = time.time(); bs = 256 
perm = torch.randperm(len(X_tr_t)) 
n_b = (len(X_tr_t) + bs - 1) // bs 
for ep in range(20): 
    perm = torch.randperm(len(X_tr_t)) 
    for bi in range(n_b): 
        ix = perm[bi*bs:(bi+1)*bs] 
        pred = model(X_tr_t[ix]) 
        loss = F.mse_loss(pred, y_tr_t[ix]) 
        opt.zero_grad(); loss.backward(); opt.step() 
    if ep % 5 == 0: 
        with torch.no_grad(): 
            pp = [] 
            for i in range(0, len(X_te_t), bs): 
                pp.append(model(X_te_t[i:i+bs]).cpu()) 
            p = torch.cat(pp).numpy() 
            print(f"  Ep {ep}: MAE={mean_absolute_error(y_te, p):.3f}") 
 
with torch.no_grad(): 
    pp = [] 
    for i in range(0, len(X_te_t), bs): 
        pp.append(model(X_te_t[i:i+bs]).cpu()) 
p_cnn = torch.cat(pp).numpy() 
m_cnn = mean_absolute_error(y_te, p_cnn) 
r_cnn = r2_score(y_te, p_cnn) 
print(f"CNN: MAE={m_cnn:.3f}, R2={r_cnn:.4f}") 
print(f"Time: {time.time()-t0:.1f}s") 
 
# === Ensemble === 
print("\n" + "="*50) 
print("Ensemble Results") 
print("="*50) 
for w in [0.5, 0.6, 0.7, 0.8, 0.9]: 
    p_ens = w * p_lgb + (1-w) * p_cnn 
    mae = mean_absolute_error(y_te, p_ens) 
    r2 = r2_score(y_te, p_ens) 
    imp = (r2 - r_lgb) / r_lgb * 100 
    print(f"  LGBM*{w:.1f}+CNN*{1-w:.1f}: MAE={mae:.3f}, R2={r2:.4f} (+{imp:.1f}%)")
 
# === Save models === 
lgb.booster_.save_model(str(OUT / "lgb_mz_nl_model.txt")) 
torch.save(model.state_dict(), str(OUT / "cnn_dbe_model.pt")) 
print(f"\nModels saved to:") 
print(f"  {OUT / 'lgb_mz_nl_model.txt'}") 
print(f"  {OUT / 'cnn_dbe_model.pt'}")
