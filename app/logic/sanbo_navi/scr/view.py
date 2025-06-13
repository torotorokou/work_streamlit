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


# --- PDFページ表示 ---
def render_pdf_pages(PDF_PATH, pages):
    with st.expander("📘 出典ページのプレビュー"):
        for p in sorted(pages):
            if isinstance(p, int) and p >= 1:
                if p in st.session_state.cache_pdf_pages:
                    st.image(
                        st.session_state.cache_pdf_pages[p],
                        caption=f"Page {p} (cached)",
                        use_container_width=True,  # ← 修正
                    )
                else:
                    with st.spinner(f"📄 Page {p} 読み込み中..."):
                        page_image = load_pdf_page(PDF_PATH, p)
                        st.session_state.cache_pdf_pages[p] = page_image
                        st.image(
                            page_image,
                            caption=f"Page {p}",
                            use_container_width=True,  # ← 修正
                        )
