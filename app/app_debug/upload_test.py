import streamlit as st
import pandas as pd
import chardet
import io
import unicodedata

st.set_page_config(page_title="CSVアップロードテスト", layout="centered")
st.title("📂 CSVアップロード確認")


def sanitize_filename(filename: str) -> str:
    """
    日本語などを含むファイル名を安全なASCII形式に変換する
    """
    import os

    name, ext = os.path.splitext(filename)
    safe_name = (
        unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
    )
    return f"{safe_name}{ext}"


def load_csv_safely(uploaded_file):
    """
    エンコーディングを自動検出してCSVを読み込む
    """
    try:
        raw_bytes = uploaded_file.read()
        result = chardet.detect(raw_bytes)
        encoding = result["encoding"]
        st.write(f"🔍 検出されたエンコーディング: `{encoding}`")
        uploaded_file.seek(0)
        return pd.read_csv(io.BytesIO(raw_bytes), encoding=encoding)
    except Exception as e:
        st.error("❌ CSVの読み込みに失敗しました")
        st.exception(e)
        return None


# --- ファイルアップロード ---
uploaded_file = st.file_uploader("CSVファイルをアップロードしてください", type="csv")

if uploaded_file is not None:
    safe_name = sanitize_filename(uploaded_file.name)
    st.info(f"✅ アップロード成功: `{safe_name}`")

    df = load_csv_safely(uploaded_file)
    if df is not None:
        st.success("✅ CSVの読み込みに成功しました！")
        st.write("📊 データプレビュー:")
        st.dataframe(df.head())
else:
    st.warning("⏳ CSVファイルを選択してください")
