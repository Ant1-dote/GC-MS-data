"""1D CNN for DBE prediction from GC-MS spectra.""" 
import numpy as np, pandas as pd, time 
from pathlib import Path 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import mean_absolute_error, r2_score 
 
OUT = Path("output") 
print("[CNN] Loading...") 
combined = pd.read_parquet(OUT / "combined.parquet") 
mask = combined["DBE"].notna() & (combined["n_fragments"] >= 3) 
df = combined[mask].copy(); n = len(df) 
print(f"  n={n}") 
 
spec = np.load(OUT / "spectral_matrix.npy")[mask.values] 
y = df["DBE"].values 
 
X_tr, X_te, y_tr, y_te = train_test_split(spec, y, test_size=0.2, random_state=42) 
 
import torch 
import torch.nn as nn 
import torch.nn.functional as F 
 
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
 
X_tr_t = torch.FloatTensor(X_tr).unsqueeze(1).to(dev) 
X_te_t = torch.FloatTensor(X_te).unsqueeze(1).to(dev) 
y_tr_t = torch.FloatTensor(y_tr).to(dev) 
y_te_t = torch.FloatTensor(y_te).to(dev) 
 
print("Training CNN...") 
t0=time.time(); bs=256 
n_b = (len(X_tr_t)+bs-1)//bs 
for ep in range(20): 
    perm = torch.randperm(len(X_tr_t)) 
    for bi in range(n_b): 
        ix = perm[bi*bs:(bi+1)*bs] 
        pred = model(X_tr_t[ix]) 
        loss = F.mse_loss(pred, y_tr_t[ix]) 
        opt.zero_grad(); loss.backward(); opt.step() 
    if ep%5==0: 
        with torch.no_grad(): 
            # batch inference to avoid OOM
            p_parts = []
            for i in range(0, len(X_te_t), bs):
                p_parts.append(model(X_te_t[i:i+bs]).cpu())
            p = torch.cat(p_parts).numpy()
            print(f"  Ep {ep}: MAE={mean_absolute_error(y_te, p):.3f}") 
 
with torch.no_grad(): 
    # batch inference to avoid OOM
    p_parts = []
    for i in range(0, len(X_te_t), bs):
        p_parts.append(model(X_te_t[i:i+bs]).cpu())
    p = torch.cat(p_parts).numpy() 
print(f"CNN: MAE={mean_absolute_error(y_te, p):.3f}, R2={r2_score(y_te, p):.4f}") 
print(f"Time: {time.time()-t0:.1f}s")