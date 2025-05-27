import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge, LogisticRegression, ElasticNet
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    GradientBoostingClassifier,
)
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.base import clone
from utils.get_holydays import get_japanese_holidays


import joblib
import os


def train_and_save_models(
    df_raw: pd.DataFrame, holidays: list[str], save_dir: str = "models"
):
    os.makedirs(save_dir, exist_ok=True)

    # --- ピボット ---
    df_pivot = (
        df_raw.groupby(["伝票日付", "品名"])["正味重量"].sum().unstack(fill_value=0)
    )
    df_pivot["合計"] = df_pivot.sum(axis=1)

    # --- 特徴量作成 ---
    df_feat = pd.DataFrame(index=df_pivot.index)
    df_feat["混合廃棄物A_前日"] = df_pivot["混合廃棄物A"].shift(1)
    df_feat["混合廃棄物B_前日"] = df_pivot["混合廃棄物B"].shift(1)
    df_feat["合計_前日"] = df_pivot["合計"].shift(1)
    df_feat["合計_3日平均"] = df_pivot["合計"].shift(1).rolling(3).mean()
    df_feat["合計_3日合計"] = df_pivot["合計"].shift(1).rolling(3).sum()
    df_feat["曜日"] = df_feat.index.dayofweek
    df_feat["週番号"] = df_feat.index.isocalendar().week
    daily_avg = df_raw.groupby("伝票日付")["正味重量"].median()
    df_feat["1台あたり正味重量_前日中央値"] = daily_avg.shift(1).expanding().median()
    holiday_dates = pd.to_datetime(holidays)
    df_feat["祝日フラグ"] = df_feat.index.isin(holiday_dates).astype(int)
    df_feat = df_feat.dropna()
    df_pivot = df_pivot.loc[df_feat.index]

    # --- 学習対象・特徴量 ---
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

    # --- スタッキング学習 ---
    X_features_all = {}
    stacked_preds = {}
    kf = KFold(n_splits=5)

    for item in target_items:
        X = (
            df_feat[ab_features]
            if item == "混合廃棄物A"
            else df_feat[[c for c in ab_features if "1台あたり" not in c]]
        )
        y = df_pivot[item]
        X_features_all[item] = X
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=False
        )

        train_meta = np.zeros((X_train.shape[0], len(base_models)))
        for i, (_, model) in enumerate(base_models):
            for train_idx, val_idx in kf.split(X_train):
                model_ = clone(model)
                model_.fit(X_train.iloc[train_idx], y_train.iloc[train_idx])
                train_meta[val_idx, i] = model_.predict(X_train.iloc[val_idx])
        meta_model_stage1.fit(train_meta, y_train)

        test_meta = np.column_stack(
            [
                clone(model).fit(X_train, y_train).predict(X_test)
                for _, model in base_models
            ]
        )
        stacked_preds[item] = meta_model_stage1.predict(test_meta)

    # --- ステージ2入力 ---
    index_final = X_test.index
    df_stage1 = pd.DataFrame(
        {f"{k}_予測": v for k, v in stacked_preds.items()}, index=index_final
    )
    for col in [
        "曜日",
        "週番号",
        "合計_前日",
        "1台あたり正味重量_前日中央値",
        "祝日フラグ",
    ]:
        df_stage1[col] = df_feat.loc[index_final, col]

    # --- 合計モデル・分類モデル ---
    y_total_final = df_pivot.loc[df_stage1.index, "合計"]
    gbdt_model.fit(df_stage1, y_total_final)
    y_total_binary = (y_total_final < 90000).astype(int)
    clf_model = GradientBoostingClassifier(
        n_estimators=100, learning_rate=0.05, max_depth=3, random_state=42
    )
    clf_model.fit(df_stage1.drop(columns=["祝日フラグ"]), y_total_binary)

    # --- 保存 ---
    joblib.dump(meta_model_stage1, f"{save_dir}/meta_model_stage1.pkl")
    joblib.dump(gbdt_model, f"{save_dir}/gbdt_model_stage2.pkl")
    joblib.dump(clf_model, f"{save_dir}/clf_model.pkl")
    joblib.dump(ab_features, f"{save_dir}/ab_features.pkl")
    joblib.dump(X_features_all, f"{save_dir}/X_features_all.pkl")
    joblib.dump(df_feat, f"{save_dir}/df_feat.pkl")
    joblib.dump(df_pivot, f"{save_dir}/df_pivot.pkl")

    print(f"✅ モデル学習＆保存完了 → {save_dir}/ に保存されました")


def make_model():
    # --- データ読込 ---
    base_dir = "/work/app/data/input"
    df_raw = pd.read_csv(f"{base_dir}/20240501-20250422.csv", encoding="utf-8")[
        ["伝票日付", "正味重量", "品名"]
    ]
    df_2020 = pd.read_csv(f"{base_dir}/2020顧客.csv")[
        ["伝票日付", "商品", "正味重量_明細"]
    ]
    df_2021 = pd.read_csv(f"{base_dir}/2021顧客.csv")[
        ["伝票日付", "商品", "正味重量_明細"]
    ]
    df_2023 = pd.read_csv(f"{base_dir}/2023_all.csv", low_memory=False)[
        ["伝票日付", "商品", "正味重量_明細"]
    ]

    # --- 過去データの整形 ---
    df_all = pd.concat([df_2020, df_2021, df_2023])
    df_all.rename(columns={"商品": "品名", "正味重量_明細": "正味重量"}, inplace=True)
    df_all["伝票日付"] = pd.to_datetime(df_all["伝票日付"], errors="coerce")

    # --- 新旧データを結合＆整形 ---
    df_raw = pd.concat([df_raw, df_all])
    df_raw["伝票日付"] = (
        df_raw["伝票日付"].astype(str).str.replace(r"\(.*\)", "", regex=True)
    )
    df_raw["伝票日付"] = pd.to_datetime(df_raw["伝票日付"], errors="coerce")
    df_raw["正味重量"] = pd.to_numeric(df_raw["正味重量"], errors="coerce")
    df_raw = df_raw.dropna(subset=["正味重量", "伝票日付"])

    # --- 祝日取得 ---
    start_date = df_raw["伝票日付"].min()
    end_date = df_raw["伝票日付"].max()
    holidays = get_japanese_holidays(start=start_date, end=end_date, as_str=True)

    # --- モデル学習＆保存 ---
    train_and_save_models(df_raw, holidays, save_dir="/work/app/data/models")


# 実行
make_model()
