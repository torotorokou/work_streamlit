import pandas as pd
import numpy as np
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler


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
    print("▶️ train_and_predict_stage1 開始")
    print("📌 df_feat_today index:", df_feat_today.index)
    print("📌 学習用特徴量サイズ:", df_past_feat.shape)
    print("📌 学習用pivotサイズ:", df_past_pivot.shape)

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

        train_meta = np.column_stack(
            [m.predict(X_train_filtered) for m in trained_models]
        )
        meta_model = clone(meta_model_proto)
        meta_model.fit(train_meta, y_train)

        X_target = df_feat_today[feature_list]
        X_target_scaled = scaler.transform(X_target)
        X_target_filtered = selector.transform(X_target_scaled)

        meta_input = np.column_stack(
            [m.predict(X_target_filtered) for m in trained_models]
        )
        pred = meta_model.predict(meta_input)[0]
        results[f"{item}_予測"] = pred

        true_val = df_pivot.loc[df_feat_today.index[0], item]
        stage1_eval[item]["y_true"].append(true_val)
        stage1_eval[item]["y_pred"].append(pred)

        print(f"✅ {item} 予測: {pred:.1f}kg / 正解: {true_val:.1f}kg")

    return results


def train_and_predict_stage2(
    all_stage1_rows, stage1_results, df_feat_today, target_items
):
    print("▶️ train_and_predict_stage2 開始")
    df_hist = pd.DataFrame(all_stage1_rows[:-1])
    print("📌 df_hist サイズ:", df_hist.shape)

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
