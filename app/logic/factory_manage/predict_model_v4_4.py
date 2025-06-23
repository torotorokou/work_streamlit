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
    features += [f"{item}_前日台数" for item in target_items]
    features += [
        "合計_前日値",
        "合計_3日平均",
        "合計_3日合計",
        "1台あたり重量_過去中央値",
        "曜日",
        "週番号",
        "祝日フラグ",
        "月初フラグ",
        "週初フラグ",
        "祝日前日フラグ",
        "連休前フラグ",
    ]
    return features


# ---------- 重量履歴・台数履歴から特徴量作成 ----------
def generate_weight_and_count_features(past_raw, target_items):
    df_pivot_weight = (
        past_raw.groupby(["伝票日付", "品名"])["正味重量"].sum().unstack(fill_value=0)
    )
    df_pivot_weight["全品目合計"] = df_pivot_weight.sum(axis=1)

    df_pivot_count = (
        past_raw.groupby(["伝票日付", "品名"])["受入番号"]
        .nunique()
        .unstack(fill_value=0)
    )

    df_feat = pd.DataFrame(index=df_pivot_weight.index)

    for item in target_items:
        df_feat[f"{item}_前日値"] = (
            df_pivot_weight[item].shift(1) if item in df_pivot_weight.columns else 0
        )
        df_feat[f"{item}_前日台数"] = (
            df_pivot_count[item].shift(1) if item in df_pivot_count.columns else 0
        )

    target_sum = df_pivot_weight[target_items].sum(axis=1)
    df_feat["合計_前日値"] = target_sum.shift(1)
    df_feat["合計_3日平均"] = target_sum.shift(1).rolling(3).mean()
    df_feat["合計_3日合計"] = target_sum.shift(1).rolling(3).sum()

    daily_avg = past_raw.groupby("伝票日付")["正味重量"].median()
    df_feat["1台あたり重量_過去中央値"] = daily_avg.shift(1).expanding().median()

    df_feat = df_feat.dropna()
    return df_feat, df_pivot_weight.loc[df_feat.index]


# ---------- カレンダー特徴量 ----------
def generate_calendar_features(target_date, holidays):
    return {
        "曜日": target_date.weekday(),
        "週番号": target_date.isocalendar().week,
        "祝日フラグ": int(target_date in holidays),
        "月初フラグ": int(target_date.day <= 5),
        "週初フラグ": int(target_date.weekday() == 0),
        "祝日前日フラグ": int(target_date + pd.Timedelta(days=1) in holidays),
        "連休前フラグ": int(
            (target_date + pd.Timedelta(days=1) in holidays)
            and (target_date + pd.Timedelta(days=2) in holidays)
        ),
    }


# ---------- ステージ1 ----------
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


def predict_stage1(X, scaler, selector, trained_models, meta_model):
    X_scaled = scaler.transform(X)
    X_filtered = selector.transform(X_scaled)
    meta_features = np.column_stack([m.predict(X_filtered) for m in trained_models])
    return meta_model.predict(meta_features)[0]


# ---------- ステージ2 ----------
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


def predict_stage2(model, scaler, selector, input_row):
    X_scaled = scaler.transform(input_row)
    X_filtered = selector.transform(X_scaled)
    return model.predict(X_filtered)[0]


# ---------- ウォークフォワード全体 ----------
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

    for target_date in all_dates[5:]:
        print(f"\n===== {target_date.date()} 処理中 =====")

        past_raw = df_raw[df_raw["伝票日付"] < target_date]
        df_feat, df_pivot = generate_weight_and_count_features(past_raw, target_items)
        calendar_features = generate_calendar_features(target_date, holidays)
        for key, val in calendar_features.items():
            df_feat[key] = val

        stage1_row = {}
        residual_row = {}
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
            pred = predict_stage1(X.iloc[[-1]], scaler, selector, models, meta_model)
            pred = max(pred, 0)
            stage1_row[f"{item}_予測"] = pred

            actual_today = y.iloc[-1]
            residual_row[f"{item}_残差"] = pred - actual_today

        # 残差移動平均
        df_resid = pd.DataFrame(stage1_history + [{**stage1_row, **residual_row}])
        ma_features = {}
        for item in target_items:
            ma5 = df_resid[f"{item}_残差"].iloc[-5:].mean() if len(df_resid) >= 5 else 0
            ma_features[f"{item}_残差_MA5"] = ma5
        stage1_row.update(residual_row)
        stage1_row.update(ma_features)

        if len(stage1_history) >= 30:
            df_input = pd.DataFrame(
                [{**stage1_row, **df_feat.iloc[-1][features_all].to_dict()}]
            ).assign(**ma_features)
            model2, scaler2, selector2, feature_cols = train_stage2_model(
                stage1_history
            )
            total_pred = predict_stage2(model2, scaler2, selector2, df_input)
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
    else:
        print("履歴不足で評価不能")


# ---------- 重要度出力 ----------
def print_stage2_importance(model, feature_cols):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    print("\n===== ステージ2特徴量重要度 (GBDT) =====")
    for idx in indices:
        print(f"{feature_cols[idx]}: {importances[idx]:.4f}")
