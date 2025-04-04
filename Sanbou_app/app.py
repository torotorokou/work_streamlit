import streamlit as st
import time

st.set_page_config(page_title="web版 参謀くん", layout="wide")
# スタイル


st.title('WEB版 参謀くん')

# サイドバーでカテゴリ選択
menu = st.sidebar.selectbox("📂 機能を選択", ["管理業務", "機能１", "機能２"])


# メインコンテンツ
if menu == "管理業務":
    st.header("📊 管理業務")
    st.subheader("📄 テンプレート選択")

    # 辞書で必要ファイルを定義
    template_dict = {
        "工場日報":"factory_report",
        "工場搬出入収支表": "balance_sheet",
        "集計項目平均表": "average_sheet",
        "管理票":"management_sheet"
    }


    template_label = st.selectbox(
        "テンプレートを選んでください",
        ["選択してください"] + list(template_dict.keys())
    )

    # 内部処理用テンプレートキーを取得
    selected_template = template_dict.get(template_label, None)

    # アップロードされたファイルを辞書で管理
    uploaded_files = {}


    # 必要ファイルチェック
    required_files = {
        "factory_report": ["yard", "shipping"],
        "balance_sheet": ["receive", "yard", "shipping"],
        "average_sheet": ["receive"],
        "management_sheet": ["receive", "yard", "shipping"]
    }

    # ファイルのアップロード欄表示（後から）
    if selected_template:
        with st.container():
            st.markdown("### 📂 CSVファイルのアップロード")
            st.info("以下のファイルをアップロードしてください。")

            #　表示用ラベル
            label_map = {
                "yard": "ヤード一覧",
                "shipping": "出荷一覧",
                "receive": "受入一覧"
            }

            #選択テンプレート毎のCSV表示
            for file_key in required_files[selected_template]:
                label = label_map.get(file_key, file_key)
                uploaded_files[file_key] = st.file_uploader(
                    f"{label}", type="csv", key = file_key + selected_template
                )

        #ファイルチェック
        all_uploaded = all(uploaded_files.get(k) is not None for k in required_files[selected_template])

        if all_uploaded:
            st.success("✅ 必要なファイルがすべてアップロードされました！")
            if st.button("📥 Excel出力"):
                st.info("※ ここで出力処理を実装します")
        else:
            st.warning("⚠️ すべての必要なCSVをアップロードしてください。")

