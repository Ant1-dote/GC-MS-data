"""Spectral similarity search against 284k GC-MS database.""" 
import numpy as np 
import pandas as pd 
from pathlib import Path 
from sklearn.metrics.pairwise import cosine_similarity 
import streamlit as st 
 
OUT = Path(__file__).resolve().parent.parent / "output" 
 
@st.cache_data 
def load_data(): 
    """Load spectral matrix and metadata, cached by Streamlit.""" 
    spec = np.load(OUT / "spectral_matrix.npy").astype(np.float32) 
    meta = pd.read_parquet(OUT / "combined.parquet") 
    return spec, meta 
 
def bin_vec(mz_str, it_str, nb=800): 
    """Convert m/z,intensity string to normalized 800-dim vector.""" 
    mz = np.array([float(x) for x in mz_str.split(",") if x.strip()]) 
    it = np.array([float(x) for x in it_str.split(",") if x.strip()]) 
    vec = np.zeros(nb, dtype=np.float32) 
    idx = np.floor(mz).astype(np.int32) 
    v = (idx >= 0) & (idx < nb) 
    for j, iv in zip(idx[v], it[v]): 
        if iv > vec[j]: 
            vec[j] = iv 
    mx = vec.max() 
    if mx > 0: 
        vec /= mx 
    return vec 
 
def search(query_vec, spec_mat, meta, top_k=10): 
    """Return top-k most similar compounds with similarity scores.""" 
    sims = cosine_similarity(query_vec.reshape(1, -1), spec_mat)[0] 
    top = np.argsort(sims)[-top_k:][::-1] 
    cols = ["chemical_name", "chemical_formular", "MW", "DBE"] 
    res = meta.iloc[top][cols].copy() 
    res["similarity"] = sims[top].round(4) 
    return res
