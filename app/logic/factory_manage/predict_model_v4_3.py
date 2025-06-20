import pandas as pd
import numpy as np
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.base import clone


# ================= 特徴量作成 =====================
def generate_features_for_date(df_raw, holidays, target_date):
    # 履歴抽出（未来ゼロ）
    past_raw = df_raw[df_raw["伝票日付"] < target_date]
    if len(past_raw) < 3:
        return None, None  # 履歴不足

    df_pivot = (
        past_raw.groupby(["伝票日付", "品名"])["正味重量"].sum().unstack(fill_value=0)
    )
    df_pivot["合計"] = df_pivot.sum(axis=1)

    df_feat = pd.DataFrame(index=df_pivot.index)
    df_feat["混合A_前日値"] = df_pivot["混合廃棄物A"].shift(1)
    df_feat["混合B_前日値"] = df_pivot["混合廃棄物B"].shift(1)
    df_feat["合計_前日値"] = df_pivot["合計"].shift(1)
    df_feat["合計_3日平均"] = df_pivot["合計"].shift(1).rolling(3).mean()
    df_feat["合計_3日合計"] = df_pivot["合計"].shift(1).rolling(3).sum()

    daily_avg = past_raw.groupby("伝票日付")["正味重量"].median()
    df_feat["1台あたり重量_過去中央値"] = daily_avg.shift(1).expanding().median()

    df_feat = df_feat.dropna()
    df_pivot = df_pivot.loc[df_feat.index]

    # ✅ 当日情報（未来リークなし）
    target_date_pd = pd.to_datetime(target_date)
    df_feat["曜日"] = target_date_pd.weekday()
    df_feat["週番号"] = target_date_pd.isocalendar().week
    df_feat["祝日フラグ"] = int(target_date_pd in holidays)

    return df_feat, df_pivot


# ================= ステージ1学習 =====================
def train_stage1_models(X_train, y_train, base_models, meta_model):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    selector = VarianceThreshold(threshold=1e-4)
    X_filtered = selector.fit_transform(X_scaled)

    trained_models = []
    for name, model in base_models:
        model_ = clone(model)
        model_.fit(X_filtered, y_train)
        trained_models.append(model_)

    train_meta = np.column_stack([m.predict(X_filtered) for m in trained_models])
    meta_model.fit(train_meta, y_train)

    return scaler, selector, trained_models, meta_model


# ================= ステージ1予測 =====================
def predict_stage1(X_target, scaler, selector, trained_models, meta_model):
    X_scaled = scaler.transform(X_target)
    X_filtered = selector.transform(X_scaled)
    base_preds = np.column_stack([m.predict(X_filtered) for m in trained_models])
    pred = meta_model.predict(base_preds)[0]
    return pred


# ================= ステージ2学習・予測 =====================
def train_and_predict_stage2(stage1_history):
    hist_df = pd.DataFrame(stage1_history).dropna()
    feature_cols = [c for c in hist_df.columns if c not in ["日付", "合計_実績"]]

    scaler2 = StandardScaler()
    X_train_scaled = scaler2.fit_transform(hist_df[feature_cols])
    selector2 = VarianceThreshold(threshold=1e-4)
    X_train_filtered = selector2.fit_transform(X_train_scaled)

    gbdt = GradientBoostingRegressor(
        n_estimators=150, learning_rate=0.05, max_depth=4, random_state=42
    )
    gbdt.fit(X_train_filtered, hist_df["合計_実績"])

    X_target_raw = hist_df.iloc[[-1]][feature_cols]
    X_target_scaled = scaler2.transform(X_target_raw)
    X_target_filtered = selector2.transform(X_target_scaled)
    total_pred = gbdt.predict(X_target_filtered)[0]

    return total_pred, hist_df.iloc[-1]["合計_実績"]


# ================= 全体パイプライン =====================
def full_walkforward_with_debug(df_raw, holidays):
    holidays = pd.to_datetime(holidays)
    df_raw["伝票日付"] = pd.to_datetime(df_raw["伝票日付"])
    df_raw = df_raw.sort_values("伝票日付")
    all_dates = np.sort(df_raw["伝票日付"].unique())

    # ✅ 最初から全特徴量定義
    all_features = [
        "混合A_前日値",
        "混合B_前日値",
        "合計_前日値",
        "合計_3日平均",
        "合計_3日合計",
        "1台あたり重量_過去中央値",
        "曜日",
        "週番号",
        "祝日フラグ",
    ]

    target_items = ["混合廃棄物A", "混合廃棄物B"]
    base_models = [
        ("elastic", ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=10000, tol=1e-2)),
        ("rf", RandomForestRegressor(n_estimators=100, random_state=42)),
    ]
    meta_model_stage1 = ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=10000, tol=1e-2)

    stage1_history = []
    all_actual, all_pred = [], []

    for target_date in all_dates[5:]:
        print(f"\n===== {pd.to_datetime(target_date).date()} 処理中 =====")

        df_feat, df_pivot = generate_features_for_date(df_raw, holidays, target_date)
        if df_feat is None:
            print("  履歴不足でスキップ")
            continue

        stage1_row = {}

        for item in target_items:
            # 混合Aは全特徴量、混合Bは台数特徴量除外
            if item == "混合廃棄物A":
                use_features = all_features
            else:
                use_features = [f for f in all_features if "1台あたり" not in f]

            X_train = df_feat[use_features]
            y_train = df_pivot[item]

            scaler, selector, trained_models, meta_model_stage1 = train_stage1_models(
                X_train, y_train, base_models, meta_model_stage1
            )

            # 直近最新行（target_date用特徴量）
            X_target_raw = X_train.iloc[[-1]]
            pred = predict_stage1(
                X_target_raw, scaler, selector, trained_models, meta_model_stage1
            )
            stage1_row[f"{item}_予測"] = pred
            print(f"    {item} 予測値: {pred:.1f}kg")

        # ステージ2用履歴蓄積
        stage1_history.append(
            {
                "日付": target_date,
                **stage1_row,
                **X_target_raw.iloc[0].to_dict(),  # 全特徴量を保存
                "合計_実績": df_pivot["合計"].iloc[-1],
            }
        )

        if len(stage1_history) >= 30:
            total_pred, total_actual = train_and_predict_stage2(stage1_history)
            all_actual.append(total_actual)
            all_pred.append(total_pred)
            print(
                f"  【ステージ2】 合計予測: {total_pred:.0f}kg, 実績: {total_actual:.0f}kg"
            )

    if len(all_actual) > 0:
        r2 = r2_score(all_actual, all_pred)
        mae = mean_absolute_error(all_actual, all_pred)
        print("\n===== 最終評価結果 =====")
        print(f"R² = {r2:.3f}, MAE = {mae:,.0f}kg")
    else:
        print("履歴不足で評価不能")
