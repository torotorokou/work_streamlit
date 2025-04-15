import logging
import os
import socket
import getpass
import time
from utils.config_loader import load_config_json


# 日本時間に変換する関数（UTC + 9時間）
def jst_time(*args):
    return time.localtime(time.time() + 9 * 60 * 60)


def app_logger(to_console=True) -> logging.Logger:
    config = load_config_json()
    log_path = config["main_paths"]["app"]

    # ログフォルダがなければ作成
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    # ロガーの取得と初期設定
    logger = logging.getLogger("app_logger")
    logger.setLevel(logging.DEBUG)

    # 重複して出力されないように既存のハンドラを削除
    if logger.hasHandlers():
        logger.handlers.clear()

    # ホスト名・ユーザー名の取得
    hostname = socket.gethostname()
    username = getpass.getuser()

    # ログの出力フォーマット（日本時間対応）
    formatter = logging.Formatter(
        f"%(asctime)s [%(levelname)s] (%(filename)s:%(lineno)d) [{hostname}/{username}] %(message)s"
    )
    formatter.converter = jst_time  # ← ここがポイント！

    # ファイル出力設定
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # コンソール出力（Streamlitでも見える）
    if to_console:
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    return logger
