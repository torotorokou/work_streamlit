# app_pages/manage_work.py
import streamlit as st
from logic.detect_csv import detect_csv_type
from utils.config_loader import load_config
from components.ui_message import show_warning_bubble
from logic.eigyo_management import template_processors
from components.custom_button import centered_button
from utils.preprocessor import prepare_csv_data
from utils.debug_tools import save_debug_parquets



def show_manage_work():
    # --- 内部データ定義 ---
    template_dict = {
        "工場日報": "factory_report",
        "工場搬出入収支表": "balance_sheet",
        "集計項目平均表": "average_sheet",
        "管理票": "management_sheet"
    }

    template_descriptions = {
        "工場日報": "ヤードと出荷データをもとに、工場内の稼働日報を出力します。",
        "工場搬出入収支表": "受入・ヤード・出荷一覧から収支表を自動集計します。",
        "集計項目平均表": "受入データをABC分類し、各品目の平均値を算出して出力します。",
        "管理票": "受入・ヤード・出荷の一覧を使って管理用の帳票を出力します。"
    }


    required_files = {
        "factory_report": ["yard", "shipping"],
        "balance_sheet": ["receive", "yard", "shipping"],
        "average_sheet": ["receive"],
        "management_sheet": ["receive", "yard", "shipping"]
    }

    csv_label_map = {
        "yard": "ヤード一覧",
        "shipping": "出荷一覧",
        "receive": "受入一覧"
    }

        # 各ファイルに対応する日付カラム名を設定（変更可能）
    date_columns = {
        "receive": "伝票日付",
        "yard": "伝票日付",
        "shipping": "伝票日付"
    }

    # --- UI ---
    st.sidebar.markdown("---")
    st.sidebar.subheader("🛠 管理業務メニュー")
    template_label = st.sidebar.radio("出力したい項目を選択してください", list(template_dict.keys()))
    selected_template = template_dict.get(template_label)
    uploaded_files = {}

    st.subheader(f"📝 {template_label} の作成")
    description = template_descriptions.get(template_label, "")
    if description:
        st.markdown(f"""<div style=\"margin-left: 2em; color:#444;\">{description}</div>""", unsafe_allow_html=True)

    config = load_config()
    header_csv_path = config["main_paths"]["check_header_csv"]

    # --- ヘッダーCSVのアップロード ---
    with st.container():
        st.markdown("### 📂 CSVファイルのアップロード")
        st.info("以下のファイルをアップロードしてください。")

        for file_key in required_files[selected_template]:
            label = csv_label_map.get(file_key, file_key)
            uploaded_file = st.file_uploader(label, type="csv", key=f"{file_key}_{selected_template}")
            uploaded_files[file_key] = uploaded_file

            # 🔍 自動判別チェック
            if uploaded_file is not None:
                detected_name = detect_csv_type(uploaded_file, header_csv_path)
                expected_name = label

                if detected_name != expected_name:
                    show_warning_bubble(expected_name, detected_name)
                    uploaded_files[file_key] = None  # 無効化（より堅牢に）

    # --- ファイルチェック ---
    required_keys = required_files[selected_template]
    missing_keys = [k for k in required_keys if uploaded_files.get(k) is None]

    # --- ステータス表示 ---
    if not missing_keys:
        st.success("✅ 必要なファイルがすべてアップロードされました！")

        # --- 書類作成ボタン ---
        st.markdown("---")
        if centered_button("📊 書類作成"):
            # --- 書類作成の前処理 ---
            dfs = prepare_csv_data(uploaded_files, date_columns)

            # デバッグ用CSV保存
            save_debug_parquets(dfs, folder="/work/data/input")

            # --- 各処理の実行 ---
            processor_func = template_processors.get(selected_template)
            if processor_func:
                dfs = processor_func(dfs, csv_label_map)


            # with st.spinner("計算中..."):
            #     latest_iteration = st.empty()
            #     bar = st.progress(0)
            #     dfs = {}

            #     for i, (k, file) in enumerate(uploaded_files.items()):
            #         latest_iteration.text(f"{csv_label_map.get(k, k)} を処理中... ({i+1}/{len(uploaded_files)})")
            #         df = pd.read_csv(file)
            #         df["ファイル種別"] = csv_label_map.get(k, k)
            #         dfs[k] = df

            #     # 実行
            #     processor_func = template_processors.get(selected_template)
            #     if processor_func:
            #         dfs = processor_func(dfs, csv_label_map)

            #     # ↓ 進捗バー演出（オプション）
            #     for i in range(100):
            #         bar.progress(i + 1)
            #         time.sleep(0.005)

            #     # 📁 Excel 出力
            #     output = BytesIO()
            #     with pd.ExcelWriter(output, engine="openpyxl") as writer:
            #         for k, df in dfs.items():
            #             df.to_excel(writer, index=False, sheet_name=csv_label_map.get(k, k))

            #     bar.empty()
            #     latest_iteration.text("✅ 書類作成が完了しました！")

            #     st.download_button(
            #         label="📥 Excelファイルをダウンロード",
            #         data=output.getvalue(),
            #         file_name="出力結果.xlsx",
            #         mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            #     )


    else:
        # 📥 アップロードされたファイル数のカウント
        uploaded_count = len(required_keys) - len(missing_keys)
        total_count = len(required_keys)

        # ✅ 進捗バー
        st.progress(uploaded_count / total_count)

        # ✅ 進捗情報メッセージ
        st.info(f"📥 {uploaded_count} / {total_count} ファイルがアップロードされました")

        # ✅ 各ファイルごとのステータス表示
        for k in required_keys:
            label = csv_label_map.get(k, k)
            if uploaded_files.get(k):
                st.markdown(f"- ✅ **{label}**")
            else:
                # 未アップロード：黄色のハイライト付きで表示
                st.markdown(f'''
                    - ⏳ <strong>{label}</strong><span style="color:#e6a800">（未アップロード）</span>
                    ''', unsafe_allow_html=True)

