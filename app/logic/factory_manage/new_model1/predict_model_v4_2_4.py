import pandas as pd
import numpy as np
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.base import clone
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
import matplotlib.pyplot as plt

# === 特徴量作成（モジュールごとに分割） ===
from logic.factory_manage.new_model1.feature_builder import (
    WeightFeatureBuilder,
    ReserveFeatureBuilder,
)


def get_target_items(df_raw, top_n=5):
    return df_raw["品名"].value_counts().head(top_n).index.tolist()


def train_and_predict_stage1(
    df_feat_today,
    df_past_feat,
    df_past_pivot,
    base_models,
    meta_model_proto,
    feature_list,
    target_items,
    stage1_eval,
    df_pivot,
):
    print("\n▶️ train_and_predict_stage1 開始")

    results = {}
    X_train = df_past_feat[feature_list]
    for item in target_items:
        y_train = df_past_pivot[item]
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        selector = VarianceThreshold(1e-4)
        X_train_filtered = selector.fit_transform(X_train_scaled)

        trained_models = []
        for name, model in base_models:
            print(f"🛠 モデル訓練中: {name} for {item}")
            m = clone(model)
            m.fit(X_train_filtered, y_train)
            trained_models.append(m)

        meta_input_train = np.column_stack(
            [m.predict(X_train_filtered) for m in trained_models]
        )
        meta_model = clone(meta_model_proto)
        meta_model.fit(meta_input_train, y_train)

        X_target = df_feat_today[feature_list]
        X_target_scaled = scaler.transform(X_target)
        X_target_filtered = selector.transform(X_target_scaled)
        meta_input_target = np.column_stack(
            [m.predict(X_target_filtered) for m in trained_models]
        )
        pred = meta_model.predict(meta_input_target)[0]
        results[f"{item}_予測"] = pred

        true_val = df_pivot.loc[df_feat_today.index[0], item]
        stage1_eval[item]["y_true"].append(true_val)
        stage1_eval[item]["y_pred"].append(pred)

    return results


def train_and_predict_stage2(
    all_stage1_rows, stage1_results, df_feat_today, target_items
):
    print("\n▶️ train_and_predict_stage2 開始")
    df_hist = pd.DataFrame(all_stage1_rows[:-1])

    X_train = df_hist.drop(columns=["合計"])
    y_train = df_hist["合計"]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    selector = VarianceThreshold(1e-4)
    X_train_filtered = selector.fit_transform(X_train_scaled)

    gbdt = GradientBoostingRegressor(
        n_estimators=150, learning_rate=0.05, max_depth=4, random_state=42
    )
    gbdt.fit(X_train_filtered, y_train)

    X_target = {
        f"{item}_予測": [stage1_results[f"{item}_予測"]] for item in target_items
    }
    for col in df_feat_today.columns:
        if col not in X_target:
            X_target[col] = df_feat_today.iloc[0][col]
    X_target = pd.DataFrame(X_target)

    X_target_scaled = scaler.transform(X_target)
    X_target_filtered = selector.transform(X_target_scaled)
    total_pred = gbdt.predict(X_target_filtered)[0]
    print(f"✅ 合計予測: {total_pred:.1f}kg")
    return total_pred


def evaluate_stage1(stage1_eval, target_items):
    print("\n===== ステージ1評価結果 =====")
    for item in target_items:
        y_true = np.array(stage1_eval[item]["y_true"])
        y_pred = np.array(stage1_eval[item]["y_pred"])
        r2 = r2_score(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        max_error = np.max(np.abs(y_true - y_pred))
        print(
            f"{item}: R² = {r2:.3f}, MAE = {mae:,.0f}kg, RMSE = {rmse:,.0f}kg, 最大誤差 = {max_error:,.0f}kg"
        )


def full_walkforward(
    df_raw, holidays, df_reserve, min_stage1_days, min_stage2_days, top_n=5
):
    print("▶️ full_walkforward 開始")

    df_raw["伝票日付"] = pd.to_datetime(df_raw["伝票日付"])
    df_raw = df_raw.sort_values("伝票日付")
    target_items = get_target_items(df_raw, top_n)

    builder = WeightFeatureBuilder(df_raw, target_items, holidays)
    df_feat, df_pivot = builder.build()

    reserve_builder = ReserveFeatureBuilder(df_reserve)
    df_reserve_feat_all = reserve_builder.build()

    base_models = [
        ("elastic", ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=10000, tol=1e-2)),
        ("rf", RandomForestRegressor(n_estimators=100, random_state=42)),
    ]
    meta_model_proto = ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=10000, tol=1e-2)

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
        "連休前フラグ",
        "連休後フラグ",
        "予約件数",
        "予約合計台数",
        "固定客予約数",
        "上位得意先予約数",
    ]

    all_actual, all_pred, all_stage1_rows = [], [], []
    stage1_eval = {item: {"y_true": [], "y_pred": []} for item in target_items}
    dates = df_feat.index

    for i, target_date in enumerate(dates):
        if i < min_stage1_days:
            continue

        df_past_feat = df_feat[df_feat.index < target_date].tail(600)
        df_past_pivot = df_pivot.loc[df_past_feat.index]

        df_reserve_today = df_reserve_feat_all[df_reserve_feat_all.index <= target_date]
        df_past_feat = df_past_feat.merge(
            df_reserve_today, left_index=True, right_index=True, how="left"
        ).fillna(0)
        df_feat_today = df_feat.loc[[target_date]].copy()
        df_feat_today = df_feat_today.merge(
            df_reserve_today, left_index=True, right_index=True, how="left"
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

        if len(all_stage1_rows) > min_stage2_days:
            total_pred = train_and_predict_stage2(
                all_stage1_rows, stage1_result, df_feat_today, target_items
            )
            total_actual = df_pivot.loc[target_date, "合計"]
            all_actual.append(total_actual)
            all_pred.append(total_pred)

    print("\n===== ステージ2評価結果 (合計) =====")
    if all_actual:
        r2 = r2_score(all_actual, all_pred)
        mae = mean_absolute_error(all_actual, all_pred)
        rmse = np.sqrt(mean_squared_error(all_actual, all_pred))
        max_error = np.max(np.abs(np.array(all_actual) - np.array(all_pred)))
        print(
            f"R² = {r2:.3f}, MAE = {mae:,.0f}kg, RMSE = {rmse:,.0f}kg, 最大誤差 = {max_error:,.0f}kg"
        )
    else:
        print("評価できるデータが不足しています")

    evaluate_stage1(stage1_eval, target_items)
    return all_actual, all_pred


def plot_feature_importances(names, importances, title="Feature Importance", top_k=15):
    sorted_idx = np.argsort(importances)[-top_k:]
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(sorted_idx)), np.array(importances)[sorted_idx], align="center")
    plt.yticks(range(len(sorted_idx)), np.array(names)[sorted_idx])
    plt.title(title)
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.show()
