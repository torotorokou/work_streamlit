from datetime import date, timedelta
from typing import List, Union
import jpholiday


def get_japanese_holidays(
    start: date, end: date, as_str: bool = True
) -> Union[List[str], List[date]]:
    """
    指定した期間の日本の祝日を取得する関数。

    Args:
        start (date): 開始日
        end (date): 終了日
        as_str (bool): Trueなら"YYYY-MM-DD"形式、Falseならdate型

    Returns:
        Union[List[str], List[date]]: 祝日のリスト（文字列またはdate型）
    """
    holidays = [
        d
        for d in (start + timedelta(days=i) for i in range((end - start).days + 1))
        if jpholiday.is_holiday(d)
    ]
    return [d.strftime("%Y-%m-%d") for d in holidays] if as_str else holidays
