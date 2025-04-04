import streamlit as st

def apply_global_style():
    st.markdown("""
        <style>
        /* 全体にフォントを強制適用（!important付き） */
        html, body, [data-testid="stAppViewContainer"], [data-testid="stMarkdownContainer"] * {
            font-family: 'Noto Sans JP', 'メイリオ', 'Hiragino Kaku Gothic ProN', sans-serif !important;
            font-size: 16px;
            line-height: 1.6;
        }
        </style>
    """, unsafe_allow_html=True)
