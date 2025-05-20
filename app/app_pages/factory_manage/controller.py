import streamlit as st


def factory_manage_work_controller():
    st.title("🏭 工場管理トップページ")

    # --- CSSで行間を調整 ---
    st.sidebar.markdown(
        """
        <style>
        div[data-baseweb="radio"] label {
            display: block;
            margin-bottom: 1.4em;  /* ← 行間を広げる */
            font-size: 1rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # --- ✅ CSS追加（行間を広げる） ---
    st.sidebar.markdown(
        """
        <style>
        div[data-baseweb="radio"] label {
            display: block;
            margin-bottom: 1.2em;  /* ← 行間を広げる */
            font-weight: normal;
            font-size: 1rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # --- ブロック①: 工場管理メニュー ---
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🏗 工場管理メニュー")

    selected_factory_menu = st.sidebar.radio(
        "表示する項目を選択してください",
        ["搬入量（仮）", "廃棄物処理管理表"],
        key="factory_menu",
    )

    # --- ブロック②: 分析メニュー ---
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 分析メニュー")

    selected_analysis_menu = st.sidebar.radio(
        "分析対象を選択してください", ["月間推移", "構成比グラフ"], key="analysis_menu"
    )

    # --- メイン画面処理 ---
    if selected_factory_menu == "搬入量（仮）":
        render_import_volume()
    elif selected_factory_menu == "廃棄物処理管理表":
        render_waste_management_table()

    # 分析部分（条件表示も可）
    if selected_analysis_menu == "月間推移":
        st.info("📈 月間推移グラフをここに表示")
    elif selected_analysis_menu == "構成比グラフ":
        st.info("🥧 処理構成比グラフをここに表示")


def render_import_volume():
    st.subheader("🚛 搬入量（仮）")
    st.write("ここに搬入量を表示します。")


def render_waste_management_table():
    st.subheader("🗑 廃棄物処理管理表")
    st.write("処理実績や分類別の集計を表示します。")


def render_factory_page_menu(options_dict: dict[str, str]) -> str:
    st.sidebar.markdown("---")
    st.sidebar.subheader("🏭 工場管理メニュー")

    selected = st.sidebar.radio(
        "表示する項目を選択してください", list(options_dict.keys())
    )

    st.subheader(f"📄 {selected}")
    description = options_dict.get(selected, "")
    if description:
        st.markdown(
            f"<div style='margin-left: 2em; color:#ccc;'>{description}</div>",
            unsafe_allow_html=True,
        )

    return selected
