from datetime import datetime
import pandas as pd


def get_weekday_japanese(date_input):
    """日付（strまたはdatetime）から日本語の曜日を返す"""
    weekdays_ja = ["日", "月", "火", "水", "木", "金", "土"]

    if isinstance(date_input, str):
        # 区切り文字に応じてパース
        if "/" in date_input:
            date_input = datetime.strptime(date_input, "%Y/%m/%d")
        elif "-" in date_input:
            date_input = datetime.strptime(date_input, "%Y-%m-%d")
        else:
            raise ValueError(
                "日付形式は 'YYYY-MM-DD' または 'YYYY/MM/DD' にしてください"
            )

    weekday_index = (date_input.weekday() + 1) % 7
    return weekdays_ja[weekday_index]


def extract_first_date(df: pd.DataFrame, col: str = "伝票日付") -> pd.Timestamp:
    """データから最初の日付を返す（責務①）"""
    return pd.to_datetime(df[col].dropna().iloc[0])
