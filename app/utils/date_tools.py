import pandas as pd
from datetime import datetime, date
from typing import Union


def get_weekday_japanese(date_input: Union[str, datetime]) -> str:
    """日付（strまたはdatetime）から日本語の曜日を返す"""
    weekdays_ja = ["日", "月", "火", "水", "木", "金", "土"]

    if isinstance(date_input, str):
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


def to_japanese_era(dt: Union[datetime, date]) -> str:
    """西暦を和暦表記に変換"""
    if hasattr(dt, "date"):
        dt = dt.date()  # datetime → date に変換

    if dt >= date(2019, 5, 1):
        year = dt.year - 2018
        era = "令和"
    elif dt >= date(1989, 1, 8):
        year = dt.year - 1988
        era = "平成"
    elif dt >= date(1926, 12, 25):
        year = dt.year - 1925
        era = "昭和"
    else:
        return f"{dt.year}年"

    return f"{era}元年" if year == 1 else f"{era}{year}年"


def to_japanese_month_day(dt) -> str:
    """
    任意の日付オブジェクトから「3月1日」の形式に変換する。
    pandas.Timestamp / datetime.date 両対応。

    Parameters
    ----------
    dt : datetime.date, datetime.datetime, pandas.Timestamp
        日付オブジェクト

    Returns
    -------
    str : 例 "3月1日"
    """
    if hasattr(dt, "date"):
        dt = dt.date()
    return f"{dt.month}月{dt.day}日"
