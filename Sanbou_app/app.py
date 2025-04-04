import streamlit as st
import time
import pandas as pd
from io import BytesIO

st.set_page_config(page_title="web版 参謀くん", layout="centered")
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

            if st.button("📊 計算開始"):
                with st.spinner("計算中..."):

                    # 🟩 プログレスバー開始
                    latest_iteration = st.empty()
                    bar = st.progress(0)

                    dfs = {}
                    total_files = len(uploaded_files)

                    for i, (k, file) in enumerate(uploaded_files.items()):
                        latest_iteration.text(f"{label_map.get(k, k)} を処理中... ({i+1}/{total_files})")
                        df = pd.read_csv(file)

                        # 👇 仮処理：ファイル種別列を追加
                        df["ファイル種別"] = label_map.get(k, k)
                        dfs[k] = df

                    for i in range(100):
                        progress = int(i + 1)
                        bar.progress(progress)
                        time.sleep(0.2)  # 実処理に合わせて削除OK

                    # 🧾 Excel出力
                    output = BytesIO()
                    with pd.ExcelWriter(output, engine="openpyxl") as writer:
                        for k, df in dfs.items():
                            df.to_excel(writer, index=False, sheet_name=label_map.get(k, k))

                    bar.empty()
                    latest_iteration.text("✅ 計算完了！")

                    # 💾 ダウンロード
                    st.download_button(
                        label="📥 Excelファイルをダウンロード",
                        data=output.getvalue(),
                        file_name="出力結果.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                    # ↑↑↑ 処理ここまで ↑↑↑
        else:
            st.warning("⚠️ すべての必要なCSVをアップロードしてください。")

