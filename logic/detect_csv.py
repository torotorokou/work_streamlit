from utils.config_loader import get_csv_type_signatures
import pandas as pd
from utils.logger import app_logger


def load_template_signatures(header_csv_path: str) -> dict:
    """
    CSVファイルからテンプレートのカラム情報を辞書として読み込む。
    """
    df = pd.read_csv(header_csv_path)
    templates = {}

    for _, row in df.iterrows():
        name = row["template_name"]
        cols = [row[f"column{i}"] for i in range(1, 6) if pd.notna(row[f"column{i}"])]
        templates[name] = cols

    return templates


def detect_csv_type(file) -> str:
    # logger = app_logger()
    try:
        signatures = get_csv_type_signatures()  # dict[str, list[str]]
        df = pd.read_csv(file, nrows=1)
        cols = list(df.columns)[:5]

        for key, expected in signatures.items():
            if cols == expected:  # ✅ 順序・長さ含めて完全一致
                return key

        return "不明な形式"

    except Exception as e:
        return f"読み込みエラー: {e}"
