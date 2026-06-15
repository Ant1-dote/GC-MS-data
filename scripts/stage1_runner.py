"""Stage 1: 数据清洗、谱图修复、公式特征提取。"""
import pandas as pd
import numpy as np
from pathlib import Path
import time

OUT = Path("output")
OUT.mkdir(exist_ok=True)

t0 = time.time()
print("[Stage 1a] 读取数据...")
df = pd.read_excel(r"data/MS_spectrum_data.xlsx")
print(f"  Shape: {df.shape}")

if "Unnamed: 7" in df.columns:
    df.drop(columns=["Unnamed: 7"], inplace=True)

# 类型统一
df["CAS"] = df["CAS"].apply(
    lambda x: str(int(x)) if isinstance(x, (int, float)) and not pd.isna(x)
    else str(x) if pd.notna(x) else ""
)
df["chemical_name"] = df["chemical_name"].astype(str)
df["chemical_formular"] = df["chemical_formular"].astype(str)
df["mz"] = df["mz"].astype(str)
df["intensity"] = df["intensity"].astype(str)

print("[Stage 1b] 修复214行错位...")
bad = 0
for i in range(len(df)):
    ms = df.iloc[i]["mz"]
    its = df.iloc[i]["intensity"]
    mc = len(ms.split(","))
    ic = len(its.split(","))
    if mc != ic:
        n = min(mc, ic)
        df.iloc[i, df.columns.get_loc("mz")] = ",".join(ms.split(",")[:n])
        df.iloc[i, df.columns.get_loc("intensity")] = ",".join(its.split(",")[:n])
        bad += 1
print(f"  Fixed: {bad}")
assert bad == 214, "Expected 214 misaligned rows"

print("[Stage 1c] 提取化学特征...")
from stage1_formula_features import batch_extract
df = batch_extract(df)
nan_dbe = df["DBE"].isna().sum()
print(f"  DBE缺失: {nan_dbe}/{len(df)}")

df.to_parquet(OUT / "stage1_all.parquet")
print(f"  全量保存: {len(df)} rows")

print("[Stage 1d] 按化合物去重...")
dedup = df.drop_duplicates(subset=["chemical_name", "CAS"], keep="first").copy()
dedup.to_parquet(OUT / "stage1_dedup.parquet")
print(f"  去重后: {len(dedup)} rows")

print(f"\nStage 1 完成, 耗时: {time.time()-t0:.1f}s")
print(f"  唯一化合物: {dedup['chemical_name'].nunique()}")
print(f"  唯一CAS:    {dedup['CAS'].nunique()}")
print(f"  MW范围:     {dedup['MW'].min():.1f} ~ {dedup['MW'].max():.1f}")
print(f"  DBE范围:    {dedup['DBE'].min():.1f} ~ {dedup['DBE'].max():.1f}")
print(f"  含N: {dedup['has_N'].sum()}, 含O: {dedup['has_O'].sum()}, 含S: {dedup['has_S'].sum()}, 含卤素: {dedup['has_halogen'].sum()}")
