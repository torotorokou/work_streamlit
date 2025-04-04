import streamlit as st
import time

st.set_page_config(page_title="web版 参謀くん", layout="wide")
# スタイル
st.markdown("""
    <link href="https://fonts.googleapis.com/css2?family=Noto+Sans+JP&display=swap" rel="stylesheet">
    <style>
    html, body, [class*="css"] {
        font-family: 'Noto Sans JP', sans-serif;
    }
    </style>
""", unsafe_allow_html=True)


st.title('web版 参謀くん')

# サイドバーでカテゴリ選択
menu = st.sidebar.selectbox("📂 機能を選択", ["管理業務", "機能１", "機能２"])

if menu == "管理業務":
    st.header("📊 管理業務")

    st.subheader("📄 テンプレート選択")
    template = st.selectbox("テンプレートを選んでください", ["工場日報", "工場搬出入収支表", "集計項目平均表", "管理票"])

    # アップロードされたファイルを辞書で管理
    uploaded_files = {}

    if template == "工場日報":
        with st.container():
            st.markdown("### 📂 CSVファイルのアップロード")
            st.info("以下のファイルをアップロードしてください。")
            uploaded_files["yard"] = st.file_uploader("ヤード一覧", type="csv", key="yard日報")
            uploaded_files["shipping"] = st.file_uploader("出荷一覧", type="csv", key="ship日報")

    elif template == "工場搬出入収支表":
        with st.container():
            st.markdown("### 📂 CSVファイルのアップロード")
            st.info("以下のファイルをアップロードしてください。")

            uploaded_files["accept"] = st.file_uploader("受入一覧", type="csv", key="accept収支")
            uploaded_files["yard"] = st.file_uploader("ヤード一覧", type="csv", key="yard収支")
            uploaded_files["shipping"] = st.file_uploader("出荷一覧", type="csv", key="ship収支")



    elif template == "集計項目平均表":
        with st.container():
            st.markdown("### 📂 CSVファイルのアップロード")
            st.info("以下のファイルをアップロードしてください。")

            uploaded_files["accept"] = st.file_uploader("受入一覧", type="csv", key="accept平均")

    elif template == "管理票":
        with st.container():
            st.markdown("### 📂 CSVファイルのアップロード")
            st.info("以下のファイルをアップロードしてください。")

            uploaded_files["accept"] = st.file_uploader("受入一覧", type="csv", key="accept管理")
            uploaded_files["yard"] = st.file_uploader("ヤード一覧", type="csv", key="yard管理")
            uploaded_files["shipping"] = st.file_uploader("出荷一覧", type="csv", key="ship管理")

    # 必要ファイルがすべてアップロードされたかチェック
    required_files = {
        "工場日報": ["yard", "shipping"],
        "工場搬出入収支表": ["accept", "yard", "shipping"],
        "集計項目平均表": ["accept"],
        "管理票": ["accept", "yard", "shipping"]
    }

    all_uploaded = all(uploaded_files.get(k) is not None for k in required_files[template])

    if all_uploaded:
        st.success("✅ 必要なファイルがすべてアップロードされました！")
        if st.button("📥 Excel出力"):
            st.info("※ ここで出力処理を実装します")
    else:
        st.warning("⚠️ すべての必要なCSVをアップロードしてください。")
