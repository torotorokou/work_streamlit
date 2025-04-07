import streamlit as st

def centered_button(label: str, key: str = None) -> bool:
    """
    中央寄せでスタイル付きのボタンを表示する関数。
    
    Parameters:
        label (str): ボタンに表示するテキスト
        key (str): ボタンのキー（任意）
    
    Returns:
        bool: ボタンが押されたかどうか
    """

    # スタイルを一度だけ適用
    if "_custom_centered_button" not in st.session_state:
        st.markdown("""
        <style>
        div.stButton > button {
            background-color: #fbbc04;
            color: #111;
            font-weight: 600;
            font-size: 16px;
            padding: 0.6rem 1.5rem;
            border-radius: 6px;
            border: none;
            min-width: 160px;
            box-shadow: 2px 2px 6px rgba(0,0,0,0.1);
        }
        div.stButton > button:hover {
            background-color: #f9a825;
        }
        </style>
        """, unsafe_allow_html=True)
        st.session_state["_custom_centered_button"] = True

    # 中央寄せレイアウト
    col1, col2, col3 = st.columns([3, 2, 3])
    with col2:
        return st.button(label, key=key)
