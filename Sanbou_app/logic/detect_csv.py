# logic/detect_csv.py
from utils.file_loader import read_csv
import pandas as pd

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


def detect_csv_type(file, header_csv_path: str) -> str:
    try:
        # 判別ルール読み込み
        signatures = load_template_signatures(header_csv_path)

        # ✅ キャッシュ付き読み込み（1行だけ）
        df = read_csv(file, nrows=1)
        cols = list(df.columns)[:5]

        for name, expected in signatures.items():
            if cols[:len(expected)] == expected:
                return name

        return "不明な形式"
    except Exception as e:
        return f"読み込みエラー: {e}"