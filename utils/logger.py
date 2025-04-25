import logging
import os
import socket
import getpass
import time
from utils.config_loader import get_path_config


# 日本時間に変換する関数（UTC + 9時間）
def jst_time(*args):
    return time.localtime(time.time() + 9 * 60 * 60)


def app_logger(to_console=True) -> logging.Logger:
    config = get_path_config()
    log_path = config["logs"]["app"]

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


def debug_logger(to_console=True) -> logging.Logger:
    config = get_path_config()
    log_path = config["logs"]["debug"]  # config.yaml内で debug.log を別定義

    # ログフォルダがなければ作成
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    logger = logging.getLogger("debug_logger")
    logger.setLevel(logging.DEBUG)

    if logger.hasHandlers():
        logger.handlers.clear()

    hostname = socket.gethostname()
    username = getpass.getuser()

    formatter = logging.Formatter(
        f"%(asctime)s [%(levelname)s] (%(filename)s:%(lineno)d) [{hostname}/{username}] %(message)s"
    )
    formatter.converter = jst_time

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    if to_console:
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    return logger
