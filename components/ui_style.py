import streamlit as st


def apply_global_style():
    st.markdown(
        """
    <style>
    .css-1d391kg {
        background-color: #fff8d6;
        border-right: 1px solid #e5e0c3;
    }

    .stButton>button {
        background-color: #f7c948;
        color: #3d3d3d;
        border: none;
        padding: 0.5em 1.5em;
        border-radius: 8px;
        font-weight: bold;
        transition: background-color 0.3s ease;
    }

    .stButton>button:hover {
        background-color: #f0b429;
        color: white;
    }

    .streamlit-expanderHeader {
        font-size: 1.1em;
        color: #5a4d1c;
        font-weight: 500;
    }

    input, textarea {
        border-radius: 6px;
        border: 1px solid #ddd2a8;
        padding: 0.4em;
        background-color: #fffef5;
    }

    .stDataFrame {
        background-color: #fffce6;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )
