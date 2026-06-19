"""GC-MS DBE Predictor - Streamlit Web UI"""
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from scripts.predict_dbe import predict

st.set_page_config(page_title="GC-MS DBE Predictor",layout="wide")
st.title("GC-MS DBE Predictor")
st.markdown("Input MS data to predict DBE")

col1,col2=st.columns(2)

with col1:
    mz_str=st.text_area("m/z list","15,26,27,39,51,52,77")
    int_str=st.text_area("intensity list","120,340,260,1110,400,320,999")
    mw=st.number_input("MW",0.0,2000.0,78.0)
    if st.button("Predict DBE"):
        try:
            dbe=predict(mz_str,int_str,mw)
            st.success(f"Predict DBE = {dbe}")
            mz_arr=np.array([float(x) for x in mz_str.split(",") if x.strip()])
            int_arr=np.array([float(x) for x in int_str.split(",") if x.strip()])
            fig,ax=plt.subplots()
            ax.bar(mz_arr,int_arr,width=0.5,color="#2196F3")
            ax.set_xlabel("m/z")
            ax.set_ylabel("Intensity")
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Error: {e}") 
st.markdown("---") 
st.header("Spectral Similarity Search") 
st.markdown("Find similar compounds from 284k database") 
 
with st.expander("Search", expanded=True): 
    q_mz = st.text_input("Query m/z", "15,26,27,39,51,52,77", key="q_mz") 
    q_int = st.text_input("Query intensity", "120,340,260,1110,400,320,999", key="q_int") 
    top_k = st.slider("Top K", 5, 50, 10) 
    if st.button("Search", type="primary"): 
        from app.search_module import load_data, bin_vec, search 
        with st.spinner("Loading database..."): 
            spec, meta = load_data() 
        with st.spinner("Searching..."): 
            qv = bin_vec(q_mz, q_int) 
            results = search(qv, spec, meta, top_k) 
        st.dataframe(results, use_container_width=True) 
        mz_arr = np.array([float(x) for x in q_mz.split(",") if x.strip()]) 
        int_arr = np.array([float(x) for x in q_int.split(",") if x.strip()]) 
        fig, ax = plt.subplots(figsize=(10, 2.5)) 
        ax.bar(mz_arr, int_arr, width=0.5, color="#FF9800") 
        ax.set_xlabel("m/z"); ax.set_ylabel("Intensity") 
        st.pyplot(fig)
