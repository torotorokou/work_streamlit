import streamlit as st
import pandas as pd

st.set_page_config(page_title="CSVアップロードテスト", layout="centered")
st.title("📂 CSVアップロード確認")

uploaded_file = st.file_uploader("CSVファイルをアップロードしてください", type="csv")

if uploaded_file is not None:
    st.info(f"✅ アップロード成功: {uploaded_file.name}")

    try:
        df = pd.read_csv(uploaded_file)
        st.success("✅ CSVの読み込みに成功しました！")
        st.write("📊 データプレビュー:")
        st.dataframe(df.head())
    except Exception as e:
        st.error("❌ CSVの読み込みに失敗しました")
        st.exception(e)
else:
    st.warning("⏳ CSVファイルを選択してください")
