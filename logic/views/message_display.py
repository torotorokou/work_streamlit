import streamlit as st

def show_success(message: str):
    st.success(message)

def show_warning(message: str):
    st.warning(message)

def show_error(message: str):
    st.error(message)

def show_date_mismatch(details: dict):
    for key, values in details.items():
        st.write(f"- `{key}`: {values}")
