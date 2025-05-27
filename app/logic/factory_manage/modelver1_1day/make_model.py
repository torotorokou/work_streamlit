# --- get_japanese_holidays を利用する前提 ---
from datetime import date, timedelta
from typing import List, Union
import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    GradientBoostingClassifier,
)
from sklearn.model_selection import KFold
from sklearn.base import clone
from utils.get_holydays import get_japanese_holidays


def train_model_with_holiday(df_raw: pd.DataFrame, model_dir: str = "./models") -> None:
    import os

    os.makedirs(model_dir, exist_ok=True)

    # --- 日付整形 & 祝日取得 ---
    df_raw = df_raw.copy()
    df_raw["伝票日付"] = df_raw["伝票日付"].str.replace(r"\(.*\)", "", regex=True)
    df_raw["伝票日付"] = pd.to_datetime(df_raw["伝票日付"], errors="coerce")
    start = df_raw["伝票日付"].min().date()
    end = df_raw["伝票日付"].max().date()
    holidays = get_japanese_holidays(start, end, as_str=True)

    df_raw["正味重量"] = pd.to_numeric(df_raw["正味重量"], errors="coerce")
    df_raw = df_raw.dropna(subset=["正味重量"])
    df_pivot = (
        df_raw.groupby(["伝票日付", "品名"])["正味重量"].sum().unstack(fill_value=0)
    )
    df_pivot["合計"] = df_pivot.sum(axis=1)

    df_feat = pd.DataFrame(index=df_pivot.index)
    df_feat["混合廃棄物A_前日"] = df_pivot["混合廃棄物A"].shift(1)
    df_feat["混合廃棄物B_前日"] = df_pivot["混合廃棄物B"].shift(1)
    df_feat["合計_前日"] = df_pivot["合計"].shift(1)
    df_feat["合計_3日平均"] = df_pivot["合計"].shift(1).rolling(3).mean()
    df_feat["合計_3日合計"] = df_pivot["合計"].shift(1).rolling(3).sum()
    df_feat["曜日"] = df_feat.index.dayofweek
    df_feat["週番号"] = df_feat.index.isocalendar().week

    daily_sum = df_raw.groupby("伝票日付")["正味重量"].sum()
    daily_count = df_raw.groupby("伝票日付")["受入番号"].nunique()
    daily_avg = daily_sum / daily_count
    df_feat["1台あたり正味重量_前日中央値"] = daily_avg.shift(1).expanding().median()

    df_feat["祝日フラグ"] = df_feat.index.isin(pd.to_datetime(holidays)).astype(int)
    df_feat = df_feat.dropna()
    df_pivot = df_pivot.loc[df_feat.index]

    ab_features = [
        "混合廃棄物A_前日",
        "混合廃棄物B_前日",
        "合計_前日",
        "合計_3日平均",
        "合計_3日合計",
        "曜日",
        "週番号",
        "1台あたり正味重量_前日中央値",
        "祝日フラグ",
    ]
    target_items = ["混合廃棄物A", "混合廃棄物B", "混合廃棄物(ｿﾌｧｰ･家具類)"]
    base_models = [
        ("elastic", ElasticNet(alpha=0.1, l1_ratio=0.5)),
        ("rf", RandomForestRegressor(n_estimators=100, random_state=42)),
    ]
    meta_model_stage1 = ElasticNet(alpha=0.1, l1_ratio=0.5)
    gbdt_model = GradientBoostingRegressor(
        n_estimators=150, learning_rate=0.05, max_depth=4, random_state=42
    )
    clf_model = GradientBoostingClassifier(
        n_estimators=100, learning_rate=0.05, max_depth=3, random_state=42
    )
    kf = KFold(n_splits=5)

    X_features_all = {}
    stacked_preds = {}

    for item in target_items:
        X = (
            df_feat[ab_features]
            if item == "混合廃棄物A"
            else df_feat[[c for c in ab_features if "1台あたり" not in c]]
        )
        y = df_pivot[item]
        X_features_all[item] = X

        train_meta = np.zeros((X.shape[0], len(base_models)))
        for i, (_, model) in enumerate(base_models):
            for train_idx, val_idx in kf.split(X):
                model_ = clone(model)
                model_.fit(X.iloc[train_idx], y.iloc[train_idx])
                train_meta[val_idx, i] = model_.predict(X.iloc[val_idx])

        meta_model_stage1.fit(train_meta, y)
        stacked_preds[item] = meta_model_stage1.predict(train_meta)

    df_stage1 = pd.DataFrame(
        {f"{k}_予測": v for k, v in stacked_preds.items()}, index=df_feat.index
    )
    for col in [
        "曜日",
        "週番号",
        "合計_前日",
        "1台あたり正味重量_前日中央値",
        "祝日フラグ",
    ]:
        df_stage1[col] = df_feat[col]

    y_total = df_pivot["合計"]
    gbdt_model.fit(df_stage1, y_total)
    y_total_binary = (y_total < 90000).astype(int)
    clf_model.fit(df_stage1.drop(columns=["祝日フラグ"]), y_total_binary)

    joblib.dump(
        (base_models, meta_model_stage1, gbdt_model, clf_model),
        f"{model_dir}/models.pkl",
    )
    joblib.dump((X_features_all, df_feat, df_pivot), f"{model_dir}/features.pkl")


df_raw = pd.read_csv("/work/app/data/input/20240501-20250422.csv")
train_model_with_holiday(df_raw, model_dir="/work/app/data/models")
