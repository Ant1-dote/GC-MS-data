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