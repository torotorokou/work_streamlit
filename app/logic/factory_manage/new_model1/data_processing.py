import pandas as pd
import numpy as np


def get_target_items(df_raw, top_n=5):
    return df_raw["品名"].value_counts().head(top_n).index.tolist()


def generate_reserve_features(df_reserve, top_k_clients=10):
    df_reserve = df_reserve.copy()
    df_reserve["予約日"] = pd.to_datetime(df_reserve["予約日"])

    top_clients = df_reserve["予約得意先名"].value_counts().head(top_k_clients).index
    df_reserve["上位得意先フラグ"] = (
        df_reserve["予約得意先名"].isin(top_clients).astype(int)
    )

    df_reserve["台数"] = pd.to_numeric(df_reserve["台数"], errors="coerce").fillna(0)

    df_feat = df_reserve.groupby("予約日").agg(
        予約件数=("予約得意先名", "count"),
        固定客予約数=("固定客", lambda x: x.sum()),
        非固定客予約数=("固定客", lambda x: (~x).sum()),
        上位得意先予約数=("上位得意先フラグ", "sum"),
        合計台数=("台数", "sum"),
        平均台数=("台数", "mean"),
    )
    df_feat["固定客比率"] = df_feat["固定客予約数"] / df_feat["予約件数"]
    return df_feat.fillna(0)


def pivot_weight_data(past_raw, target_items):
    # 廃棄物ごとの日次合計
    df_pivot = (
        past_raw.groupby(["伝票日付", "品名"])["正味重量"].sum().unstack(fill_value=0)
    )
    for item in target_items:
        if item not in df_pivot.columns:
            df_pivot[item] = 0
    df_pivot = df_pivot.sort_index()
    df_pivot["合計"] = df_pivot[target_items].sum(axis=1)
    return df_pivot


def create_base_features(df_pivot, target_items):
    df_feat = pd.DataFrame(index=df_pivot.index)
    for item in target_items:
        df_feat[f"{item}_前日値"] = df_pivot[item].shift(1)
        df_feat[f"{item}_前週平均"] = df_pivot[item].shift(1).rolling(7).mean()

    df_feat["合計_前日値"] = df_pivot["合計"].shift(1)
    df_feat["合計_3日平均"] = df_pivot["合計"].shift(1).rolling(3).mean()
    df_feat["合計_3日合計"] = df_pivot["合計"].shift(1).rolling(3).sum()
    df_feat["合計_前週平均"] = df_pivot["合計"].shift(1).rolling(7).mean()

    return df_feat


def add_calendar_features(df_feat, holidays):
    df_feat["曜日"] = df_feat.index.dayofweek
    df_feat["週番号"] = df_feat.index.isocalendar().week
    holiday_dates = pd.to_datetime(holidays)
    df_feat["祝日フラグ"] = df_feat.index.isin(holiday_dates).astype(int)
    df_feat["祝日前フラグ"] = df_feat.index.map(
        lambda d: (d + pd.Timedelta(days=1)) in holiday_dates
    ).astype(int)
    df_feat["祝日後フラグ"] = df_feat.index.map(
        lambda d: (d - pd.Timedelta(days=1)) in holiday_dates
    ).astype(int)
    return df_feat


def add_median_weight_feature(df_feat, past_raw):
    daily_avg = past_raw.groupby("伝票日付")["正味重量"].median()
    df_feat["1台あたり重量_過去中央値"] = (
        daily_avg.shift(1).rolling(60, min_periods=10).median()
    )
    return df_feat


def merge_weather(df_feat, df_weather):
    if df_weather is not None:
        df_feat = df_feat.merge(
            df_weather, left_index=True, right_index=True, how="left"
        ).fillna(0)
    return df_feat


def generate_weight_features(past_raw, target_items, holidays, df_weather=None):
    df_pivot = pivot_weight_data(past_raw, target_items)
    df_feat = create_base_features(df_pivot, target_items)
    df_feat = add_calendar_features(df_feat, holidays)
    df_feat = add_median_weight_feature(df_feat, past_raw)
    df_feat = merge_weather(df_feat, df_weather)
    df_feat = df_feat.dropna()
    df_pivot = df_pivot.loc[df_feat.index]
    return df_feat, df_pivot


import pandas as pd
import requests


def fetch_tokyo_weather(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Open-Meteoの無料APIを使って東京の天気情報を取得（日本語で取得したいなら別API案もあり）
    """
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": 35.6895,
        "longitude": 139.6917,
        "start_date": start_date,
        "end_date": end_date,
        "daily": "precipitation_sum,temperature_2m_max,temperature_2m_min",
        "timezone": "Asia/Tokyo",
    }
    res = requests.get(url, params=params)
    data = res.json()
    df = pd.DataFrame(data["daily"])
    df["time"] = pd.to_datetime(df["time"])
    df = df.set_index("time")
    return df


def preprocess_weather(df_weather: pd.DataFrame) -> pd.DataFrame:
    """
    必要に応じて天気列を加工（one-hotはここでは不要。連続値でもOK）
    """
    df_weather = df_weather.rename(
        columns={
            "temperature_2m_max": "気温_最高",
            "temperature_2m_min": "気温_最低",
            "precipitation_sum": "降水量",
        }
    )
    return df_weather
