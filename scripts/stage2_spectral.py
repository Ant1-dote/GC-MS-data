"""Stage 2: 谱图向量化与碎片特征提取。"""
import pandas as pd, numpy as np, time
from pathlib import Path

OUT = Path("output")
MZ_MAX = 800
N_BINS = MZ_MAX

t0 = time.time()
print("[Stage 2a] 加载去重数据...")
df = pd.read_parquet(OUT / "stage1_dedup.parquet")
n = len(df)
print(f"  共 {n} 个化合物")

print("[Stage 2b] 谱图向量化...")
n_frags = np.zeros(n, dtype=np.int32)
bp_mz = np.zeros(n, dtype=np.float32)
bp_int = np.zeros(n, dtype=np.float32)
tic_arr = np.zeros(n, dtype=np.float32)
batch_size = 10000
n_batches = (n + batch_size - 1) // batch_size
spec_list = []

for bi in range(n_batches):
    s = bi * batch_size
    e = min(s + batch_size, n)
    if bi % 5 == 0:
        print(f"  批次 {bi+1}/{n_batches} ({s}~{e})")
    bdf = df.iloc[s:e]
    mat = np.zeros((e - s, N_BINS), dtype=np.float32)
    for j, (_, row) in enumerate(bdf.iterrows()):
        try:
            mz = np.fromstring(row["mz"], sep=",", dtype=np.float32)
            it = np.fromstring(row["intensity"], sep=",", dtype=np.float32)
        except:
            continue
        if len(mz) == 0:
            continue
        n_frags[s + j] = len(mz)
        tic_arr[s + j] = it.sum()
        mi = it.argmax()
        bp_mz[s + j] = mz[mi]
        bp_int[s + j] = it[mi]
        idx = np.floor(mz).astype(np.int32)
        v = (idx >= 0) & (idx < N_BINS)
        idx2, iv = idx[v], it[v]
        for vi, vv in zip(idx2, iv):
            mat[j, vi] = max(vv, mat[j, vi])
    spec_list.append(mat)

print("[Stage 2c] 合并谱图矩阵...")
spec_mat = np.vstack(spec_list)
del spec_list
rm = spec_mat.max(axis=1, keepdims=True)
rm[rm == 0] = 1
spec_norm = spec_mat / rm
del spec_mat
print(f"  谱图矩阵: {spec_norm.shape}")

print("[Stage 2d] 计算谱图熵...")
eps = 1e-10
ent = np.zeros(n, dtype=np.float32)
for i in range(n):
    if n_frags[i] > 1:
        p = spec_norm[i] / (spec_norm[i].sum() + eps)
        p = p[p > 0]
        ent[i] = -np.sum(p * np.log2(p))

print("[Stage 2e] 保存特征...")
spec_feat = pd.DataFrame({
    "n_fragments": n_frags,
    "base_peak_mz": bp_mz,
    "base_peak_int": bp_int,
    "tic_log": np.log10(tic_arr + 1),
    "spectral_entropy": ent,
})
np.save(OUT / "spectral_matrix.npy", spec_norm)
spec_feat.to_parquet(OUT / "stage2_features.parquet")
cols = ["chemical_name", "CAS", "MW", "DBE", "DBE_per_C",
        "mass_defect", "HC", "OC", "NC", "ONC"]
combined = pd.concat([
    df[cols].reset_index(drop=True), spec_feat], axis=1)
combined.to_parquet(OUT / "combined.parquet")
print(f"Stage 2 完成, 耗时: {time.time()-t0:.1f}s")
print(f"  熵范围: {ent.min():.2f}~{ent.max():.2f}")
print(f"  碎片: {n_frags.min()}~{n_frags.max()}")
print(f"  稀疏度: {(spec_norm == 0).mean()*100:.1f}%")
