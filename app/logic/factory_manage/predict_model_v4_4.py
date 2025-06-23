import pandas as pd
import numpy as np
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
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
        "曜日",
        "週番号",
    ]
    return features


# ---------- 重量履歴特徴量作成 ----------
def generate_weight_features(past_raw, target_items):
    df_pivot = (
        past_raw.groupby(["伝票日付", "品名"])["正味重量"].sum().unstack(fill_value=0)
    )
    df_pivot["全品目合計"] = df_pivot.sum(axis=1)
    df_feat = pd.DataFrame(index=df_pivot.index)
    for item in target_items:
        df_feat[f"{item}_前日値"] = (
            df_pivot[item].shift(1) if item in df_pivot.columns else 0
        )
    target_sum = df_pivot[target_items].sum(axis=1)
    df_feat["合計_前日値"] = target_sum.shift(1)
    df_feat["合計_3日平均"] = target_sum.shift(1).rolling(3).mean()
    df_feat["合計_3日合計"] = target_sum.shift(1).rolling(3).sum()
    df_feat = df_feat.dropna()
    return df_feat, df_pivot.loc[df_feat.index]


# ---------- カレンダー特徴量 ----------
def generate_calendar_features(target_date, holidays):
    return {
        "曜日": target_date.weekday(),
        "週番号": target_date.isocalendar().week,
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


# ---------- 重要度出力 ----------
def print_stage2_importance(model, feature_cols):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    print("\n===== ステージ2特徴量重要度 (GBDT) =====")
    for idx in indices:
        print(f"{feature_cols[idx]}: {importances[idx]:.4f}")


# ---------- 評価指標出力 ----------
def print_metrics(all_actual, all_pred):
    # ここで list → numpy配列へ変換
    y_true = np.array(all_actual)
    y_pred = np.array(all_pred)

    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    max_err = np.max(np.abs(y_true - y_pred))

    print("\n===== 最終評価結果 =====")
    print(f"R² = {r2:.3f}")
    print(f"MAE = {mae:,.0f} kg")
    print(f"RMSE = {rmse:,.0f} kg")
    print(f"MAPE = {mape:.2f} %")
    print(f"最大誤差 = {max_err:,.0f} kg")


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
        df_feat, df_pivot = generate_weight_features(past_raw, target_items)
        calendar_features = generate_calendar_features(target_date, holidays)
        for key, val in calendar_features.items():
            df_feat[key] = val

        stage1_row = {}
        residual_row = {}
        for item in target_items:
            use_features = features_all
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
        print_metrics(all_actual, all_pred)
        print_stage2_importance(model2, feature_cols)
    else:
        print("履歴不足で評価不能")


# ---------- 評価指標 ----------
def print_metrics_per_day(all_actual, all_pred):
    print("\n===== 7日間評価結果 =====")
    for day in range(7):
        actual = all_actual[day]
        pred = all_pred[day]
        r2 = r2_score(actual, pred)
        mae = mean_absolute_error(actual, pred)
        rmse = np.sqrt(mean_squared_error(actual, pred))
        print(f"Day {day + 1}: R²={r2:.3f}, MAE={mae:,.0f}kg, RMSE={rmse:,.0f}kg")


# ---------- 7日先までのシミュレーション ----------
def full_weekly_simulation(df_raw, target_items):
    df_raw["伝票日付"] = pd.to_datetime(df_raw["伝票日付"])
    df_raw = df_raw.sort_values("伝票日付")
    all_dates = pd.to_datetime(np.sort(df_raw["伝票日付"].unique()))
    features_all = get_features_all(target_items)

    base_models = [
        ("elastic", ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=10000)),
        ("rf", RandomForestRegressor(n_estimators=100, random_state=42)),
    ]
    meta_model_proto = ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=10000)

    all_actual = [[] for _ in range(7)]
    all_pred = [[] for _ in range(7)]

    for target_date in all_dates[
        30:-7
    ]:  # 十分履歴があって、かつ7日先まで実績がある日だけ
        print(f"===== {target_date.date()} 処理中 =====")

        # 学習データ準備
        past_raw = df_raw[df_raw["伝票日付"] < target_date]
        df_feat, df_pivot = generate_weight_features(past_raw, target_items)
        stage1_row = {}
        for item in target_items:
            X = df_feat[features_all]
            y = df_pivot[item]
            scaler, selector, models, meta_model = train_stage1_models(
                X, y, base_models, meta_model_proto
            )
            pred = predict_stage1(X.iloc[[-1]], scaler, selector, models, meta_model)
            stage1_row[f"{item}_予測"] = max(pred, 0)
        stage1_history = [
            {
                **stage1_row,
                **generate_calendar_features(target_date),
                "合計_実績": df_pivot[target_items].sum(axis=1).iloc[-1],
            }
        ]
        model2, scaler2, selector2, feature_cols = train_stage2_model(stage1_history)

        # 7日シミュレーション
        sim_date = target_date
        sim_df_feat = df_feat.copy()
        sim_stage1_row = stage1_row.copy()
        for day_ahead in range(7):
            sim_date += pd.Timedelta(days=1)
            calendar = generate_calendar_features(sim_date)

            # 特徴量更新: 前日値を前回予測で更新
            for item in target_items:
                sim_df_feat.loc[sim_date, f"{item}_前日値"] = sim_stage1_row[
                    f"{item}_予測"
                ]
            sim_df_feat.loc[sim_date, "合計_前日値"] = sum(
                sim_stage1_row[f"{item}_予測"] for item in target_items
            )
            sim_df_feat.loc[sim_date, "合計_3日平均"] = (
                sim_df_feat["合計_前日値"].rolling(3).mean().iloc[-1]
            )
            sim_df_feat.loc[sim_date, "合計_3日合計"] = (
                sim_df_feat["合計_前日値"].rolling(3).sum().iloc[-1]
            )
            sim_df_feat.loc[sim_date, "曜日"] = calendar["曜日"]
            sim_df_feat.loc[sim_date, "週番号"] = calendar["週番号"]

            # ステージ1 (各品目)
            sim_stage1_row = {}
            for item in target_items:
                X_new = sim_df_feat[features_all].iloc[[-1]]
                pred = predict_stage1(X_new, scaler, selector, models, meta_model)
                sim_stage1_row[f"{item}_予測"] = max(pred, 0)

            # ステージ2 (合計予測)
            df_input = pd.DataFrame([{**sim_stage1_row, **calendar}])
            total_pred = predict_stage2(model2, scaler2, selector2, df_input)

            # 実績と評価保存
            total_actual = df_raw[
                (df_raw["伝票日付"] == sim_date) & (df_raw["品名"].isin(target_items))
            ]["正味重量"].sum()
            if total_actual > 0:
                all_pred[day_ahead].append(total_pred)
                all_actual[day_ahead].append(total_actual)

    print_metrics_per_day(all_actual, all_pred)
