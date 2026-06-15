"""GC-MS分析主管线."""
import os, sys, pickle, time
import pandas as pd, numpy as np
from pathlib import Path
OUTPUT = Path("output"); OUTPUT.mkdir(exist_ok=True)
DATA_PATH = r"data/MS_spectrum_data.xlsx"
def log(msg):
    t = time.strftime("%H:%M:%S")
    print(f"[{t}] {msg}")
def stage1_load_and_clean():
    log("读取数据...")
    df = pd.read_excel(DATA_PATH)
    log(f"原始行数: {len(df)}")
    if "Unnamed: 7" in df.columns:
        df.drop(columns=["Unnamed: 7"], inplace=True)
    return df
if __name__ == "__main__":
    log("=== Stage 1 ===")
    df = stage1_load_and_clean()
    log(f"加载完成: {df.shape}")
