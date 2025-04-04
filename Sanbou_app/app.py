import streamlit as st
import pandas as pd
import time
from io import BytesIO

from app_pages.top_page import show_top_page
from components.update_log import show_update_log
from components.manual_links import show_manual_links
from components.notice import show_notice
from components.version_info import show_version_info
from components.ui_style import apply_global_style

# ✅ 初期設定
st.set_page_config(page_title="web版 参謀くん", layout="centered")
apply_global_style()  # ← フォント適用
st.query_params["dev_mode"] = "true"  # ← 任意のクエリ活用

# サイドバーでカテゴリ選択
menu = st.sidebar.selectbox("📂 機能を選択", ["トップページ","管理業務", "機能１", "機能２"])

if menu == "トップページ":
    st.title("📘 WEB版 参謀くん")
else:
    st.title(f"📂 {menu}")


if menu == "トップページ":
    show_top_page()
    # サイドバーにお知らせ
    with st.sidebar:
        st.markdown("---")
        show_notice()
        st.markdown("---")
        show_manual_links()
        st.markdown("---")
        show_update_log()
        st.markdown("---")
        show_version_info()

# メインコンテンツ
elif menu == "管理業務":
    # サイドバーにテンプレートメニュー（選択形式）
    st.sidebar.markdown("---")
    st.sidebar.subheader("🛠 管理業務メニュー")

    template_option = st.sidebar.radio(
        "出力したい項目を選択して下さい",
        ["工場日報", "工場搬出収支表","集計項目平均表","管理業務"]
    )

    # メイン画面の表示内容を切り替え
    st.header("📊 管理業務")

    if template_option == "工場日報":
        st.subheader("📝 工場日報の入力")
        # 補足説明（シンプルな段落）
        st.markdown("""
        <div style="margin-left: 2em;">
        この項目では、受入データをもとにABC分類ごとの平均値を計算し、  
        所定のExcelフォーマットにて自動で出力します。
        </div>
        """, unsafe_allow_html=True)

        st.write("ここに工場日報の機能を実装します。")

    elif template_option == "工場搬出収支表":
        st.subheader("📈 工場搬出収支表の集計")
        st.write("ここに収支表のアップロードや集計処理を入れます。")

    elif template_option == "集計項目平均表":
        st.subheader("📤 ABC集計項目平均表の出力")
        st.write("ここで管理票を生成・ダウンロードできます。")


    elif template_option == "管理票":
        st.subheader("📤 管理票の出力")
        st.write("ここで管理票を生成・ダウンロードできます。")

    else:
        st.subheader("📄 項目を選択して下さい")



    st.sidebar.markdown("---")
    st.sidebar.subheader("🛠 管理業務メニュー")
    st.sidebar.checkbox("工場日報")
    st.sidebar.checkbox("収支表の集計")
    st.sidebar.checkbox("管理票の出力")
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

