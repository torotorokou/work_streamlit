import os

def load_settings():
    """環境に応じた設定をロードして返す"""
    app_env = os.getenv("APP_ENV", "dev")

    if app_env == "prod":
        from config.settings.prod import *
    elif app_env == "staging":
        from config.settings.staging import *
    else:
        from config.settings.dev import *

    # ここでモジュールスコープから設定を取り出して返す
    settings = {
        "ENV_NAME": ENV_NAME,
        "DEBUG": DEBUG,
        "STREAMLIT_SERVER_PORT": STREAMLIT_SERVER_PORT,
    }
    return settings
