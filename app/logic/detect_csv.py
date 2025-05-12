from utils.logger import app_logger  # ← Streamlit環境用のロガー取得
from utils.file_loader import read_csv
from utils.config_loader import receive_header_definition
import pandas as pd


def load_template_signatures(df) -> dict:
    """
    CSVファイルからテンプレートのカラム情報を辞書として読み込む。
    """
    templates = {}

    for _, row in df.iterrows():
        name = row["template_name"]
        cols = [row[f"column{i}"] for i in range(1, 6) if pd.notna(row[f"column{i}"])]
        templates[name] = cols

    return templates


def detect_csv_type(file) -> str:
    logger = app_logger()
    try:
        logger.info("📥 detect_csv_type(): 開始")

        # 判別ルール読み込み
        df_csv = receive_header_definition()
        # logger.info(
        #     f"🧾 ヘッダー定義DataFrame（先頭5行）:\n{df_csv.head().to_string(index=False)}"
        # )

        signatures = load_template_signatures(df_csv)
        # logger.info(f"📌 判別ルール（signatures）: {signatures}")

        # ✅ ファイルのカーソルを先頭に戻す（重要）
        file.seek(0)
        df = read_csv(file, nrows=1)
        cols = list(df.columns)[:5]
        # logger.info(f"📊 アップロードCSVの先頭列: {cols}")

        for name, expected in signatures.items():
            # logger.info(f"🔍 比較中: 種別 = {name}, 期待ヘッダー = {expected}")
            if cols[: len(expected)] == expected:
                logger.info(f"✅ 種別判定成功: {name}")
                return name

        logger.warning("⚠️ 種別が一致しませんでした → 不明な形式")
        return "不明な形式"

    except Exception as e:
        logger.error(f"❌ 読み込みエラー: {e}", exc_info=True)
        return f"読み込みエラー: {e}"
