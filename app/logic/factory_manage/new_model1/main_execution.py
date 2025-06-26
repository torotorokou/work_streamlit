import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, r2_score

from .data_processing import (
    get_target_items,
    generate_reserve_features,
    generate_weight_features,
)
from .model_training import train_and_predict_stage1, train_and_predict_stage2
from .evaluation import evaluate_stage1


def full_walkforward(df_raw, holidays, df_reserve, df_weather=None, top_n=5):
    print("▶️ full_walkforward 開始")
    print("📌 入力データ件数:", len(df_raw))

    df_raw["伝票日付"] = pd.to_datetime(df_raw["伝票日付"])
    df_raw = df_raw.sort_values("伝票日付")
    target_items = get_target_items(df_raw, top_n)
    df_feat, df_pivot = generate_weight_features(
        df_raw, target_items, holidays, df_weather=df_weather
    )

    weather_features = [col for col in df_feat.columns if col.startswith("天気_")]
    feature_list = [
        *[f"{item}_前日値" for item in target_items],
        *[f"{item}_前週平均" for item in target_items],
        "合計_前日値",
        "合計_3日平均",
        "合計_前週平均",
        "曜日",
        "週番号",
        "1台あたり重量_過去中央値",
        "祝日フラグ",
        "祝日前フラグ",
        "祝日後フラグ",
        "予約件数",
        "合計台数",
        "固定客予約数",
        "上位得意先予約数",
        *weather_features,
    ]

    all_actual, all_pred, all_stage1_rows = [], [], []
    stage1_eval = {item: {"y_true": [], "y_pred": []} for item in target_items}
    dates = df_feat.index

    for i, target_date in enumerate(dates):
        if i < 30:
            continue

        df_past_feat = df_feat[df_feat.index < target_date].tail(600)
        df_past_pivot = df_pivot.loc[df_past_feat.index]

        df_reserve_filtered = df_reserve[
            pd.to_datetime(df_reserve["予約日"]) <= target_date
        ]
        df_reserve_feat_all = generate_reserve_features(df_reserve_filtered)

        df_past_feat = df_past_feat.merge(
            df_reserve_feat_all, left_index=True, right_index=True, how="left"
        ).fillna(0)

        df_feat_today = df_feat.loc[[target_date]].copy()
        df_feat_today = df_feat_today.merge(
            df_reserve_feat_all, left_index=True, right_index=True, how="left"
        ).fillna(0)

        print(f"\n=== {target_date.strftime('%Y-%m-%d')} を予測中 ===")
        stage1_result = train_and_predict_stage1(
            df_feat_today,
            df_past_feat,
            df_past_pivot,
            base_models,
            meta_model_proto,
            feature_list,
            target_items,
            stage1_eval,
            df_pivot,
        )

        row = {f"{item}_予測": stage1_result[f"{item}_予測"] for item in target_items}
        for col in df_feat_today.columns:
            if col not in row:
                row[col] = df_feat_today.iloc[0][col]
        row["合計"] = df_pivot.loc[target_date, "合計"]
        all_stage1_rows.append(row)

        if len(all_stage1_rows) > 30:
            total_pred = train_and_predict_stage2(
                all_stage1_rows, stage1_result, df_feat_today, target_items
            )
            total_actual = df_pivot.loc[target_date, "合計"]
            all_actual.append(total_actual)
            all_pred.append(total_pred)

    print("\n===== ステージ2評価結果 (合計) =====")
    if all_actual:
        print(
            f"R² = {r2_score(all_actual, all_pred):.3f}, MAE = {mean_absolute_error(all_actual, all_pred):,.0f}kg"
        )
    else:
        print("評価できるデータが不足しています")

    evaluate_stage1(stage1_eval, target_items)
    return all_actual, all_pred
