import os
from dotenv import load_dotenv
from openai import OpenAI
import streamlit as st


def get_resource_paths() -> dict:
    return {
        "PDF_PATH": "/work/app/logic/sanbo_navi/local_data/master/SOLVEST.pdf",
        "JSON_PATH": "/work/app/logic/sanbo_navi/local_data/master/structured_SOLVEST_output_final.json",
        "FAISS_PATH": "/work/app/logic/sanbo_navi/local_data/master/vectorstore/solvest_faiss_corrected",
        "ENV_PATH": "/work/app/logic/sanbo_navi/config/.env",
    }
