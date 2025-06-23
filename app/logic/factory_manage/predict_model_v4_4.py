import pandas as pd
import numpy as np
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.base import clone


# ---------- 自動特徴量リスト作成 ----------
def get_features_all(target_items):
    features = [f"{item}_前日値" for item in target_items]
    features += [
        "合計_前日値",
        "合計_3日平均",
        "合計_3日合計",
        "1台あたり重量_過去中央値",
        "曜日",
        "週番号",
        "祝日フラグ",
    ]
    return features


# ---------- 重量履歴から特徴量作成 ----------
def generate_weight_features(past_raw, target_items):
    df_pivot = (
        past_raw.groupby(["伝票日付", "品名"])["正味重量"].sum().unstack(fill_value=0)
    )
    df_pivot["全品目合計"] = df_pivot.sum(axis=1)

    df_feat = pd.DataFrame(index=df_pivot.index)
    for item in target_items:
        if item in df_pivot.columns:
            df_feat[f"{item}_前日値"] = df_pivot[item].shift(1)
        else:
            df_feat[f"{item}_前日値"] = 0

    target_sum = df_pivot[target_items].sum(axis=1)
    df_feat["合計_前日値"] = target_sum.shift(1)
    df_feat["合計_3日平均"] = target_sum.shift(1).rolling(3).mean()
    df_feat["合計_3日合計"] = target_sum.shift(1).rolling(3).sum()

    daily_avg = past_raw.groupby("伝票日付")["正味重量"].median()
    df_feat["1台あたり重量_過去中央値"] = daily_avg.shift(1).expanding().median()

    df_feat = df_feat.dropna()
    return df_feat, df_pivot.loc[df_feat.index]


# ---------- 1日分の全特徴量統合 ----------
def generate_features_for_date(df_raw, holidays, target_date, target_items):
    past_raw = df_raw[df_raw["伝票日付"] < target_date]
    if len(past_raw) < 3:
        return None, None

    df_feat, df_pivot = generate_weight_features(past_raw, target_items)
    df_feat["曜日"] = target_date.weekday()
    df_feat["週番号"] = target_date.isocalendar().week
    df_feat["祝日フラグ"] = int(target_date in holidays)
    return df_feat, df_pivot


# ---------- ステージ1 (個別モデル学習) ----------
def train_stage1_models(X, y, base_models, meta_model_proto):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    selector = VarianceThreshold(1e-4)
    X_filtered = selector.fit_transform(X_scaled)

    trained_models = [clone(model).fit(X_filtered, y) for _, model in base_models]
    meta_model = clone(meta_model_proto)
    meta_features = np.column_stack([m.predict(X_filtered) for m in trained_models])
    meta_model.fit(meta_features, y)

    return scaler, selector, trained_models, meta_model


# ---------- ステージ1 予測 ----------
def predict_stage1(X, scaler, selector, trained_models, meta_model):
    X_scaled = scaler.transform(X)
    X_filtered = selector.transform(X_scaled)
    meta_features = np.column_stack([m.predict(X_filtered) for m in trained_models])
    return meta_model.predict(meta_features)[0]


# ---------- ステージ2 (合計補正メタモデル) ----------
def train_stage2_model(stage1_history):
    df = pd.DataFrame(stage1_history).dropna()
    feature_cols = [c for c in df.columns if c not in ["日付", "合計_実績"]]
    X = df[feature_cols]
    y = df["合計_実績"]

    scaler = StandardScaler()
    selector = VarianceThreshold(1e-4)
    X_filtered = selector.fit_transform(scaler.fit_transform(X))

    model = GradientBoostingRegressor(
        n_estimators=150, learning_rate=0.05, max_depth=4, random_state=42
    )
    model.fit(X_filtered, y)
    return model, scaler, selector, feature_cols


# ---------- ステージ2 予測 ----------
def predict_stage2(model, scaler, selector, input_row):
    X_scaled = scaler.transform(input_row)
    X_filtered = selector.transform(X_scaled)
    return model.predict(X_filtered)[0]


# ---------- ステージ2特徴量重要度出力 ----------
def print_stage2_importance(model, feature_cols):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    print("\n===== ステージ2特徴量重要度 (GBDT) =====")
    for idx in indices:
        print(f"{feature_cols[idx]}: {importances[idx]:.4f}")


# ---------- ステージ1 ElasticNet係数出力 ----------
def print_stage1_elastic_importance(elastic_model, selected_columns):
    print("\n===== ステージ1特徴量重要度 (ElasticNet弱学習器) =====")
    for name, coef in zip(selected_columns, elastic_model.coef_):
        print(f"{name}: {coef:.4f}")


# ---------- 完全ウォークフォワード検証パイプライン ----------
def full_walkforward_pipeline(df_raw, holidays, target_items):
    holidays = pd.to_datetime(holidays)
    df_raw["伝票日付"] = pd.to_datetime(df_raw["伝票日付"])
    df_raw = df_raw.sort_values("伝票日付")
    all_dates = pd.to_datetime(np.sort(df_raw["伝票日付"].unique()))

    features_all = get_features_all(target_items)

    base_models = [
        ("elastic", ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=10000)),
        ("rf", RandomForestRegressor(n_estimators=100, random_state=42)),
    ]
    meta_model_proto = ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=10000)

    stage1_history, all_actual, all_pred = [], [], []
    last_elastic_model, last_selected_columns = None, None

    for target_date in all_dates[5:]:
        print(f"\n===== {target_date.date()} 処理中 =====")
        df_feat, df_pivot = generate_features_for_date(
            df_raw, holidays, target_date, target_items
        )
        if df_feat is None:
            print("  履歴不足でスキップ")
            continue

        stage1_row = {}
        for item in target_items:
            use_features = [
                f
                for f in features_all
                if f != "1台あたり重量_過去中央値" or item == "混合廃棄物A"
            ]
            X = df_feat[use_features]
            y = (
                df_pivot[item]
                if item in df_pivot.columns
                else pd.Series(0, index=X.index)
            )

            scaler, selector, models, meta_model = train_stage1_models(
                X, y, base_models, meta_model_proto
            )
            # ステージ1予測直後
            pred = predict_stage1(X.iloc[[-1]], scaler, selector, models, meta_model)
            pred = max(pred, 0)  # ← ここで負値カット
            stage1_row[f"{item}_予測"] = pred
            print(f"    {item} 予測値: {pred:.1f}kg")

            elastic_model = models[0]
            selected_columns = X.columns[selector.get_support()]
            last_elastic_model, last_selected_columns = elastic_model, selected_columns

        if len(stage1_history) >= 30:
            df_input = pd.DataFrame(
                [{**stage1_row, **df_feat.iloc[-1][features_all].to_dict()}]
            )
            model2, scaler2, selector2, feature_cols = train_stage2_model(
                stage1_history
            )
            total_pred = predict_stage2(model2, scaler2, selector2, df_input)

            # ★ target_items合計のみ評価対象
            total_actual = df_pivot[target_items].sum(axis=1).iloc[-1]

            all_pred.append(total_pred)
            all_actual.append(total_actual)
            print(
                f"  【ステージ2】 合計予測: {total_pred:.0f}kg, 実績: {total_actual:.0f}kg"
            )

        total_actual_full = df_pivot[target_items].sum(axis=1).iloc[-1]
        stage1_history.append(
            {
                "日付": target_date,
                **stage1_row,
                **df_feat.iloc[-1][features_all].to_dict(),
                "合計_実績": total_actual_full,
            }
        )

    if all_actual:
        print("\n===== 最終評価結果 =====")
        print(
            f"R² = {r2_score(all_actual, all_pred):.3f}, MAE = {mean_absolute_error(all_actual, all_pred):,.0f}kg"
        )
        print_stage2_importance(model2, feature_cols)
        print_stage1_elastic_importance(last_elastic_model, last_selected_columns)
    else:
        print("履歴不足で評価不能")
