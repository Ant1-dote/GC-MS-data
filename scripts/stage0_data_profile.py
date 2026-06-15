"""Stage 0: 数据探查 — 确认m/z范围、重复扫描分布、质量亏损特征。"""
import pandas as pd, numpy as np

df = pd.read_excel(r'data/MS_spectrum_data.xlsx')

# 固定随机采样1万行解析mz
sample = df.sample(10000, random_state=42)
all_mz, all_int = [], []
for _, row in sample.iterrows():
    try:
        mz = [float(x) for x in str(row['mz']).split(',') if x.strip()]
        it = [float(x) for x in str(row['intensity']).split(',') if x.strip()]
        if len(mz) == len(it):
            all_mz.extend(mz); all_int.extend(it)
    except:
        pass

all_mz = np.array(all_mz)
print(f"[m/z range] {all_mz.min():.1f} ~ {all_mz.max():.1f}")
print(f"[unit mass] {(all_mz % 1 == 0).mean()*100:.1f}% are integer m/z")

# 重复测量
dup = df[df.duplicated(subset='chemical_name', keep=False)]
print(f"[multi-scan] {dup.chemical_name.nunique()} / {df.chemical_name.nunique()} compounds have >=2 scans")

# 质量亏损分布
mw = df['MW'].dropna()
mass_defect = mw - np.floor(mw)
print(f"[mass defect] {mass_defect.min():.4f} ~ {mass_defect.max():.4f}")

# 公式解析测试
from stage1_formula_features import parse_formula
test_formulas = ['CH4', 'C2H4', 'C6H6', 'C8H10', 'C6H12O6', 'C4H9NO3', 'C20H40O2', 'C22H44O2']
for f in test_formulas:
    res = parse_formula(f)
    if res:
        dbe = res['C'] + 1 - (res['H'] + res['Hal']) / 2 + res['N'] / 2
        print(f"  {f:15s} -> C={res['C']}, H={res['H']}, DBE={dbe:.1f}")
print(f"[formulas parsed OK]")
