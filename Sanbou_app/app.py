import streamlit as st

st.title('参謀くん')


# サイドバーでカテゴリ選択
menu = st.sidebar.selectbox("機能を選択", ["出力処理", "カレンダー表示"])

if menu == "出力処理":
    st.title("📄 出力処理")

    tab1, tab2 = st.tabs(["CSVアップロード", "出力結果"])

    with tab1:
        st.subheader("CSVファイルのアップロード")

        with st.expander("▶ アップロードステップ"):
            uploaded_file = st.file_uploader("CSVファイルを選択", type="csv")

        with st.expander("▶ テンプレート選択"):
            template = st.selectbox("テンプレートを選んでください", ["帳票A", "帳票B", "帳票C"])

        if uploaded_file and template:
            st.success("準備完了！Excel出力ボタンを押せます。")
            st.button("Excel出力")

    with tab2:
        st.subheader("過去の出力履歴（ダミー表示）")
        st.write("ここに履歴テーブルや再出力ボタンを表示")

elif menu == "カレンダー表示":
    st.title("🗓️ カレンダー表示")
    st.write("ここにCSVから抽出したイベントをカレンダー表示")
