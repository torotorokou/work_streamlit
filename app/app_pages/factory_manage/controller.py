import streamlit as st


def factory_manage_work_controller():
    controller = FactoryManageWorkController()
    controller.run()


class FactoryManageWorkController:
    def __init__(self):
        self.menu_options = ["搬入量（仮）", "廃棄物処理管理表"]
        self.selected_menu = None

    def run(self):
        inject_sidebar_css()
        self.selected_menu = render_sidebar(self.menu_options)
        self.route()

    def route(self):
        if self.selected_menu == "搬入量（仮）":
            render_import_volume()
        elif self.selected_menu == "廃棄物処理管理表":
            render_waste_management_table()


def render_sidebar(menu_options: list[str]) -> str:
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🏗 工場管理メニュー")

    selected = st.sidebar.radio(
        "表示する項目を選択してください",
        menu_options,
        key="factory_menu",
    )
    return selected


def render_import_volume():
    st.subheader("🚛 搬入量（仮）")
    st.write("ここに搬入量を表示します。")


def render_waste_management_table():
    st.subheader("🗑 廃棄物処理管理表")
    st.write("処理実績や分類別の集計を表示します。")


def inject_sidebar_css():
    css = """
    <style>
    div[data-baseweb="radio"] label {
        display: block;
        margin-bottom: 1.2em;
        font-weight: normal;
        font-size: 1rem;
    }
    </style>
    """
    st.sidebar.markdown(css, unsafe_allow_html=True)
