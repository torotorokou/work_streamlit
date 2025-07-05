import streamlit as st
from pdf2image import convert_from_path

# --- PDF画像の読み込み ---
@st.cache_resource
def load_pdf_first_page(path, dpi=100):
    return convert_from_path(path, dpi=dpi, first_page=1, last_page=1)

@st.cache_resource
def load_pdf_page(path, page_number, dpi=100):
    return convert_from_path(
        path, dpi=dpi, first_page=page_number, last_page=page_number
    )[0]

# --- 1ページ目の表示（新規追加） ---
def render_pdf_first_page(pdf_image):
    st.image(pdf_image, caption="Page 1", use_container_width=True)

# --- PDFページ表示 ---
def render_pdf_pages(PDF_PATH, pages):
    if "cache_pdf_pages" not in st.session_state:
        st.session_state.cache_pdf_pages = {}

    page_numbers = []
    for p in pages:
        if isinstance(p, str) and "-" in p:
            try:
                start_page, end_page = map(int, p.split("-"))
                page_numbers.extend(range(start_page, end_page + 1))
            except:
                st.warning(f"ページ範囲の解析エラー: {p}")
                continue
        else:
            try:
                page_numbers.append(int(p))
            except:
                st.warning(f"ページ番号の解析エラー: {p}")
                continue

    with st.expander("📘 出典ページのプレビュー"):
        for p in sorted(set(page_numbers)):
            if p >= 1:
                if p in st.session_state.cache_pdf_pages:
                    st.image(
                        st.session_state.cache_pdf_pages[p],
                        caption=f"Page {p} (cached)",
                        use_container_width=True,
                    )
                else:
                    with st.spinner(f"📄 Page {p} 読み込み中..."):
                        page_image = load_pdf_page(PDF_PATH, p)
                        st.session_state.cache_pdf_pages[p] = page_image
                        st.image(
                            page_image,
                            caption=f"Page {p}",
                            use_container_width=True,
                        )
