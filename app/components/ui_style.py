import streamlit as st


def apply_global_style():
    st.markdown(
        """
    <style>
    /* ✅ サイドバー背景 */
    .css-1d391kg {
        background-color: #e0f2ff;
        border-right: 1px solid #b6e0fe;
    }

/* ✅ ボタン（青グラデーション） */
.stButton>button {
    background: linear-gradient(to right, #3b82f6, #60a5fa);  /* 青グラデーション */
    color: white;
    border: none;
    padding: 0.6em 1.5em;
    border-radius: 6px;
    font-weight: 600;
    font-size: 16px;
    transition: all 0.3s ease-in-out;
}
.stButton>button:hover {
    background: linear-gradient(to right, #1e40af, #2563eb);  /* ホバー時の濃い青グラデーション */
    color: white;
    box-shadow: 0 3px 8px rgba(0, 0, 0, 0.2);
    transform: translateY(-1px);
}


    /* ✅ エクスパンダー */
    .streamlit-expanderHeader {
        font-size: 1.05em;
        color: #1e3a8a;
        font-weight: 600;
    }

    /* ✅ 入力欄 */
    input, textarea {
        background-color: #ffffff;
        border: 1px solid #93c5fd;
        border-radius: 6px;
        padding: 0.5em;
        color: #1e293b;
    }

    /* ✅ DataFrame 表 */
    .stDataFrame {
        background-color: #f0f9ff;
    }

    /* ✅ ページ上部の余白削減 */
    section.main > div:first-child {
        padding-top: 0rem;
        margin-top: 0rem;
    }

    /* ✅ 見出し調整 */
    h1, h2, h3 {
        color: #1e3a8a;
        font-family: 'Segoe UI', sans-serif;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )
