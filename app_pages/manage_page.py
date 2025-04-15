import streamlit as st

# 変数の宣言
template_dict ={}
template_descriptions={}
required_files={}
csv_label_map={}
date_columns = {"receive": "伝票日付", "yard": "伝票日付", "shipping": "伝票日付"}

# --- UI ---
st.sidebar.markdown("---")
st.sidebar.subheader("🛠 管理業務メニュー")
template_label = st.sidebar.radio(
    "出力したい項目を選択してください", list(template_dict.keys())
)