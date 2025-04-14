# processors/__init__.py
import sys
import os
import importlib
from utils.config_loader import load_config
from utils.logger import app_logger
import streamlit as st


# /work/logic を import path に追加
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

logger = app_logger()
config = load_config()
template_configs = config["templates"]

template_processors = {}

# 各テンプレートの処理関数を動的にインポート
for key, info in template_configs.items():
    try:
        module = importlib.import_module(f"eigyo_management.{key}")
        template_processors[key] = getattr(module, "process")
    except Exception as e:
        logger.warning(f"❌ 処理関数の読み込みエラー: {key} → {e}")
