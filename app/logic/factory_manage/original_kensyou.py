import pandas as pd
import numpy as np
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    GradientBoostingClassifier,
)
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.base import clone
from utils.config_loader import get_path_from_yaml
from sklearn.linear_model import ElasticNet
from utils.get_holydays import get_japanese_holidays
from logic.factory_manage.sql import save_ukeire_data


# 祝日フラグを含むモデルの学習・検証・予測を一括実行する関数


def train_and_predict_with_holiday(
    df_raw: pd.DataFrame, start_date: str, end_date: str, holidays: list[str]
) -> pd.DataFrame:
    print(len(df_raw))
    print(f"model_中身 = {df_raw['伝票日付'].min()} / {df_raw['伝票日付'].max()}")

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

    daily_avg = df_raw.groupby("伝票日付")["正味重量"].median()

    # daily_count = df_raw.groupby("伝票日付")["受入番号"].nunique()
    # daily_avg = daily_sum / daily_count

    df_feat["1台あたり正味重量_前日中央値"] = daily_avg.shift(1).expanding().median()

    holiday_dates = pd.to_datetime(holidays)
    df_feat["祝日フラグ"] = df_feat.index.isin(holiday_dates).astype(int)

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
    kf = KFold(n_splits=5)

    # --- ステージ1学習 ---
    X_features_all = {}
    stacked_preds = {}

    for item in target_items:
        X = (
            df_feat[ab_features]
            if item == "混合廃棄物A"
            else df_feat[[c for c in ab_features if "1台あたり" not in c]]
        )
        y = df_pivot[item]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=False
        )
        X_features_all[item] = X

        train_meta = np.zeros((X_train.shape[0], len(base_models)))
        for i, (_, model) in enumerate(base_models):
            for j, (train_idx, val_idx) in enumerate(kf.split(X_train)):
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

    y_total_final = df_pivot.loc[df_stage1.index, "合計"]
    gbdt_model.fit(df_stage1, y_total_final)

    # --- 分類モデル学習（GBC使用） ---
    y_total_binary = (y_total_final < 90000).astype(int)
    clf_model = GradientBoostingClassifier(
        n_estimators=100, learning_rate=0.05, max_depth=3, random_state=42
    )
    clf_model.fit(df_stage1.drop(columns=["祝日フラグ"]), y_total_binary)

    # --- 評価 ---
    r2 = r2_score(y_total_final, gbdt_model.predict(df_stage1))
    mae = mean_absolute_error(y_total_final, gbdt_model.predict(df_stage1))

    # --- 将来予測 ---
    last_date = df_feat.index[-1]
    predict_dates = pd.date_range(start=start_date, end=end_date)
    residuals = y_total_final - gbdt_model.predict(df_stage1)
    bias = residuals.mean()
    std = residuals.std()

    results = []
    for predict_date in predict_dates:
        new_row = {
            "混合廃棄物A_前日": df_pivot.loc[last_date, "混合廃棄物A"],
            "混合廃棄物B_前日": df_pivot.loc[last_date, "混合廃棄物B"],
            "合計_前日": df_pivot.loc[last_date, "合計"],
            "合計_3日平均": df_pivot["合計"].shift(1).rolling(3).mean().loc[last_date],
            "合計_3日合計": df_pivot["合計"].shift(1).rolling(3).sum().loc[last_date],
            "曜日": predict_date.dayofweek,
            "週番号": predict_date.isocalendar().week,
            "1台あたり正味重量_前日中央値": df_feat[
                "1台あたり正味重量_前日中央値"
            ].iloc[-1],
            "祝日フラグ": int(predict_date in holiday_dates),
        }
        df_input = pd.DataFrame(new_row, index=[predict_date])
        for item in target_items:
            x_item = (
                df_input[ab_features]
                if item == "混合廃棄物A"
                else df_input[[c for c in ab_features if "1台あたり" not in c]]
            )
            meta_input = np.column_stack(
                [
                    clone(model)
                    .fit(X_features_all[item], df_pivot.loc[df_feat.index, item])
                    .predict(x_item)
                    for _, model in base_models
                ]
            )
            df_input[f"{item}_予測"] = meta_model_stage1.predict(meta_input)[0]

        stage2_input = df_input[
            [
                f"{target_items[0]}_予測",
                f"{target_items[1]}_予測",
                f"{target_items[2]}_予測",
                "曜日",
                "週番号",
                "合計_前日",
                "1台あたり正味重量_前日中央値",
                "祝日フラグ",
            ]
        ]
        y_pred = gbdt_model.predict(stage2_input)[0]
        y_adjusted = y_pred + bias
        lower = y_adjusted - 1.96 * std
        upper = y_adjusted + 1.96 * std

        # --- 分類判定 ---
        label = "通常"
        prob = None
        if 85000 <= y_adjusted <= 95000:
            X_clf = stage2_input.drop(columns=["祝日フラグ"])
            prob = clf_model.predict_proba(X_clf)[0][1]
            classification = clf_model.predict(X_clf)[0]
            label = "警告" if classification == 1 else "注意"

        results.append(
            {
                "日付": predict_date.strftime("%Y-%m-%d"),
                "予測値": y_pred,
                "補正後予測": y_adjusted,
                "下限95CI": lower,
                "上限95CI": upper,
                "判定ラベル": label,
                "未満確率": round(prob, 3) if prob is not None else None,
            }
        )

    df_result = pd.DataFrame(results).set_index("日付")
    print(f"✅ R² = {r2:.3f}, MAE = {mae:,.0f} kg")
    return df_result


holidays = get_japanese_holidays(start="2020-01-01", end="2025-12-31")

# holidays = [
#     "2025-01-01",
#     "2025-01-13",
#     "2025-02-11",
#     "2025-02-23",
#     "2025-03-20",
#     "2025-04-29",
#     "2025-05-03",
#     "2025-05-04",
#     "2025-05-05",
#     "2025-05-06",
#     "2025-07-21",
#     "2025-08-11",
#     "2025-09-15",
#     "2025-09-23",
#     "2025-10-13",
#     "2025-11-03",
#     "2025-11-23",
#     "2025-12-23",
# ]


import pandas as pd
from utils.config_loader import get_path_from_yaml


def log_info(title: str, df: pd.DataFrame, col: str):
    print(f"📊 {title}")
    print(f"・shape: {df.shape}")
    print(f"・{col} dtype: {df[col].dtype}")
    print(f"・NaN件数: {df[col].isna().sum()}")
    if pd.api.types.is_datetime64_any_dtype(df[col]):
        print(f"・範囲: {df[col].min()} ～ {df[col].max()}")
    print("-" * 40)


def make_df_mae():
    print("🔄 処理開始: make_df_mae()")
    base_dir = get_path_from_yaml("input", section="directories")

    # --- 最新データ（df_new） ---
    df_new = pd.read_csv(f"{base_dir}/20240501-20250422.csv", encoding="utf-8")
    df_new = df_new[["伝票日付", "正味重量", "品名"]]

    # 📌 括弧削除 & 日付型へ変換
    df_new["伝票日付"] = (
        df_new["伝票日付"]
        .astype(str)
        .str.replace(r"\(.*\)", "", regex=True)
        .str.strip()
    )
    df_new["伝票日付"] = pd.to_datetime(df_new["伝票日付"], errors="coerce")
    # log_info("📥 最新データ df_new", df_new, "伝票日付")

    # --- 過去データ（df_all） ---
    df_2020 = pd.read_csv(f"{base_dir}/2020顧客.csv")[
        ["伝票日付", "商品", "正味重量_明細"]
    ]
    df_2021 = pd.read_csv(f"{base_dir}/2021顧客.csv")[
        ["伝票日付", "商品", "正味重量_明細"]
    ]
    df_2023 = pd.read_csv(f"{base_dir}/2023_all.csv")[
        ["伝票日付", "商品", "正味重量_明細"]
    ]

    df_all = pd.concat([df_2020, df_2021, df_2023], ignore_index=True)

    # 📌 括弧削除 & 日付変換
    df_all["伝票日付"] = (
        df_all["伝票日付"]
        .astype(str)
        .str.replace(r"\(.*\)", "", regex=True)
        .str.strip()
    )
    df_all["伝票日付"] = pd.to_datetime(df_all["伝票日付"], errors="coerce")

    # 📌 カラム名統一
    df_all.rename(columns={"商品": "品名", "正味重量_明細": "正味重量"}, inplace=True)
    # log_info("📘 過去データ df_all", df_all, "伝票日付")

    # --- 結合 ---
    df_raw = pd.concat([df_new, df_all], ignore_index=True)
    # log_info("🔗 結合データ df_raw", df_raw, "伝票日付")

    # --- 正味重量数値化 & 欠損除去 ---
    df_raw["正味重量"] = pd.to_numeric(df_raw["正味重量"], errors="coerce")
    df_raw = df_raw.dropna(subset=["正味重量"])
    df_raw = df_raw.sort_values("伝票日付").reset_index(drop=True)
    print(f"✅ 最終レコード数: {len(df_raw)}")
    print(
        f"🧾 最終伝票日付範囲: {df_raw['伝票日付'].min()} ～ {df_raw['伝票日付'].max()}"
    )
    print("=" * 60)

    # df_raw = data_seikei(df_raw)

    return df_raw


# def data_seikei(df_raw: pd.DataFrame):
#     df_raw["伝票日付"] = pd.to_datetime(df_raw["伝票日付"], errors="coerce")


if __name__ == "__main__":
    df_raw = make_df_mae()
    save_ukeire_data(df_raw)
    df_pred = train_and_predict_with_holiday(
        df_raw, "2025-06-01", "2025-06-30", holidays
    )
    print(df_pred)
