import streamlit as st
from logic.detect_csv import detect_csv_type
from components.ui_message import show_warning_bubble


def handle_uploaded_files(required_keys, csv_label_map, header_csv_path):

    uploaded_files = {}
    for key in required_keys:
        uploaded = st.session_state.get(f"uploaded_{key}")
        if uploaded:
            expected_name = csv_label_map.get(key, key)
            detected_name = detect_csv_type(uploaded, header_csv_path)
            if detected_name != expected_name:
                show_warning_bubble(expected_name, detected_name)
                st.session_state[f"uploaded_{key}"] = None
                uploaded_files[key] = None
            else:
                uploaded_files[key] = uploaded
        else:
            uploaded_files[key] = None
    return uploaded_files
