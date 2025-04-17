import streamlit as st
from utils.config_loader import get_csv_date_columns,get_csv_label_map

# 変数の宣言
template_dict ={}
template_descriptions={}
required_files={}

csv_label_map={}
get_csv_label_map
date_columns ={}
get_csv_date_columns()

# --- UI ---
st.sidebar.markdown("---")
st.sidebar.subheader("🛠 管理業務メニュー")
template_label = st.sidebar.radio(
    "出力したい項目を選択してください", list(template_dict.keys())
)