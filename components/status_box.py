import streamlit as st


def render_status_box(message, bg_color, border_color):
    st.markdown(
        f"""
        <div style="margin-top: -0.5em; margin-bottom: 1.5em; padding: 0.4em 1em;
                    background-color: {bg_color}; border-left: 4px solid {border_color};
                    border-radius: 4px; font-weight: 500; color: #111;">
            {message}
        </div>
        """,
        unsafe_allow_html=True,
    )
