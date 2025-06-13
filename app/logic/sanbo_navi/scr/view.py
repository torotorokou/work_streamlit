import streamlit as st
from pdf2image import convert_from_path


# --- PDF画像の読み込み ---
@st.cache_resource
def load_pdf_first_page(path, dpi=100):
    # PDFの1ページ目を画像として読み込む
    return convert_from_path(path, dpi=dpi, first_page=1, last_page=1)


@st.cache_resource
def load_pdf_page(path, page_number, dpi=100):
    # 指定ページのPDFを画像として読み込む
    return convert_from_path(
        path, dpi=dpi, first_page=page_number, last_page=page_number
    )[0]
