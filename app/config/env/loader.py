import os

def load_settings():
    """環境に応じた設定をロードして返す"""
    app_env = os.getenv("APP_ENV", "dev")

    if app_env == "prod":
        from config.env import prod as settings_module
    elif app_env == "staging":
        from config.env import staging as settings_module
    else:
        from config.env import dev as settings_module

    settings = {
        "ENV_NAME": settings_module.ENV_NAME,
        "DEBUG": settings_module.DEBUG,
        "STREAMLIT_SERVER_PORT": settings_module.STREAMLIT_SERVER_PORT,
    }
    return settings
