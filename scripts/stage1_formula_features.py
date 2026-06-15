"""分子式解析与化学特征提取模块."""
import re, pandas as pd
from typing import Optional

_PAT = re.compile(r"([A-Z][a-z]*)(\d*\.?\d*)")
_HAL = {"F","Cl","Br","I"}
FEATURE_COLS = ["C","H","O","N","S","DBE","DBE_per_C",
                "mass_defect","HC","OC","NC","ONC",
                "has_N","has_O","has_S","has_halogen"]

def parse_formula(formula: str) -> Optional[dict]:
    """解析分子式为 {元素:原子数} 字典。"""
    if pd.isna(formula) or not isinstance(formula, str):
        return None
    formula = formula.strip()
    if not formula:
        return None
    elems = {}
    for m in _PAT.finditer(formula):
        elem, num = m.groups()
        if elem == "D":  # 氘计入H
            elem = "H"
        elems[elem] = elems.get(elem, 0) + (int(float(num)) if num else 1)
    return elems if elems else None


def calc_dbe(elems: dict) -> float:
    """DBE = C + 1 - (H+卤素)/2 + (N+P)/2。"""
    c = elems.get("C", 0)
    h = sum(elems.get(e, 0) for e in ("H","D"))
    hal = sum(elems.get(e, 0) for e in _HAL)
    n = elems.get("N",0) + elems.get("P",0)
    return c + 1 - (h + hal) / 2 + n / 2


def extract_features(formula: str, mw: float) -> Optional[dict]:
    """从分子式+MW提取化学特征向量。"""
    elems = parse_formula(formula)
    if elems is None:
        return None
    c = elems.get("C",0); h = sum(elems.get(e,0) for e in ("H","D"))
    o = elems.get("O",0); n = elems.get("N",0); s = elems.get("S",0)
    dbe = calc_dbe(elems)
    loss = mw - round(mw)
    return {"C":c,"H":h,"O":o,"N":n,"S":s,"DBE":dbe,
            "DBE_per_C":round(dbe/c,4) if c>0 else 0,
            "mass_defect":loss,"HC":round(h/c,4) if c>0 else 0,
            "OC":round(o/c,4) if c>0 else 0,
            "NC":round(n/c,4) if c>0 else 0,
            "ONC":round((o+n)/c,4) if c>0 else 0,
            "has_N":1 if n>0 else 0,"has_O":1 if o>0 else 0,
            "has_S":1 if s>0 else 0,
            "has_halogen":1 if any(elems.get(e,0)>0 for e in _HAL) else 0}


def batch_extract(df: pd.DataFrame) -> pd.DataFrame:
    """批量提取DataFrame中所有化合物的化学特征。"""
    res = df.copy(); rows = []
    for _, r in df.iterrows():
        feat = extract_features(r.get("chemical_formular",""), r.get("MW",0))
        rows.append(feat if feat else {k:None for k in FEATURE_COLS})
    for col in FEATURE_COLS:
        res[col] = [r[col] for r in rows]
    return res
