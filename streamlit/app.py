## ex) streamlit run app.py --server.port 30001 -- --train_gt train.json --valid_gt val.json --valid_csv submission.csv

import streamlit as st

st.set_page_config(page_title="CV12 semantic segmentation", layout="wide")

st.title("CV12 semantic segmentation")
st.markdown(
    """
    <--- Pages
    1. Annotation \n
    2. Validation
"""
)
