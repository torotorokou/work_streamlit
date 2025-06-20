import pandas as pd
import numpy as np
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.base import clone


def full_walkforward_with_debug(df_raw, holidays):
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
    target_items = ["混合廃棄物A", "混合廃棄物B"]

    base_models = [
        ("elastic", ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=10000, tol=1e-2)),
        ("rf", RandomForestRegressor(n_estimators=100, random_state=42)),
    ]
    meta_model_stage1 = ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=10000, tol=1e-2)

    # === 履歴期間自動化 ===
    df_raw["伝票日付"] = pd.to_datetime(df_raw["伝票日付"])
    df_raw = df_raw.sort_values("伝票日付")
    all_dates = np.sort(df_raw["伝票日付"].unique())
    history_window = len(all_dates)

    stage1_history = []
    all_actual, all_pred = [], []

    for target_date in all_dates[5:]:
        print(f"\n===== {target_date} 処理中 =====")

        # 履歴抽出（未来ゼロ）
        past_raw = df_raw[df_raw["伝票日付"] < target_date]

        # 特徴量作成
        df_pivot = (
            past_raw.groupby(["伝票日付", "品名"])["正味重量"]
            .sum()
            .unstack(fill_value=0)
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
        daily_avg = past_raw.groupby("伝票日付")["正味重量"].median()
        df_feat["1台あたり正味重量_前日中央値"] = (
            daily_avg.shift(1).expanding().median()
        )
        holiday_dates = pd.to_datetime(holidays)
        df_feat["祝日フラグ"] = df_feat.index.isin(holiday_dates).astype(int)
        df_feat = df_feat.dropna()
        df_pivot = df_pivot.loc[df_feat.index]

        # 履歴制限（全部使う）
        df_feat_train = df_feat.tail(history_window)
        df_pivot_train = df_pivot.loc[df_feat_train.index]

        print(
            f"  ステージ1履歴: {df_feat_train.index.min().date()} ～ {df_feat_train.index.max().date()}"
        )

        stage1_row = {}

        for item in target_items:
            X_train_raw = (
                df_feat_train[ab_features]
                if item == "混合廃棄物A"
                else df_feat_train[[c for c in ab_features if "1台あたり" not in c]]
            )
            y_train = df_pivot_train[item]

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_raw)
            selector = VarianceThreshold(threshold=1e-4)
            X_train_filtered = selector.fit_transform(X_train_scaled)

            trained_models = []
            for name, model in base_models:
                model_ = clone(model)
                model_.fit(X_train_filtered, y_train)
                trained_models.append(model_)

            train_meta = np.column_stack(
                [m.predict(X_train_filtered) for m in trained_models]
            )
            meta_model_stage1.fit(train_meta, y_train)

            if target_date in df_feat.index:
                X_target_raw = (
                    df_feat.loc[[target_date], ab_features]
                    if item == "混合廃棄物A"
                    else df_feat.loc[
                        [target_date], [c for c in ab_features if "1台あたり" not in c]
                    ]
                )
                X_target_scaled = scaler.transform(X_target_raw)
                X_target_filtered = selector.transform(X_target_scaled)
                base_preds = np.column_stack(
                    [m.predict(X_target_filtered) for m in trained_models]
                )
                pred = meta_model_stage1.predict(base_preds)[0]
            else:
                pred = np.nan

            stage1_row[f"{item}_予測"] = pred
            print(f"    {item} 予測値: {pred:.1f} kg")

        # ステージ2履歴蓄積
        stage1_history.append(
            {
                "日付": target_date,
                **stage1_row,
                "曜日": (
                    df_feat.loc[target_date, "曜日"]
                    if target_date in df_feat.index
                    else np.nan
                ),
                "週番号": (
                    df_feat.loc[target_date, "週番号"]
                    if target_date in df_feat.index
                    else np.nan
                ),
                "合計_前日": (
                    df_feat.loc[target_date, "合計_前日"]
                    if target_date in df_feat.index
                    else np.nan
                ),
                "1台あたり正味重量_前日中央値": (
                    df_feat.loc[target_date, "1台あたり正味重量_前日中央値"]
                    if target_date in df_feat.index
                    else np.nan
                ),
                "祝日フラグ": (
                    df_feat.loc[target_date, "祝日フラグ"]
                    if target_date in df_feat.index
                    else np.nan
                ),
                "合計_実績": (
                    df_pivot.loc[target_date, "合計"]
                    if target_date in df_pivot.index
                    else np.nan
                ),
            }
        )

        if len(stage1_history) >= history_window:
            hist_df = pd.DataFrame(stage1_history[-history_window:]).dropna()
            feature_cols = [
                c for c in hist_df.columns if c not in ["日付", "合計_実績"]
            ]

            print(
                f"  ステージ2学習: {hist_df['日付'].min().date()} ～ {hist_df['日付'].max().date()}"
            )

            scaler2 = StandardScaler()
            X_train_scaled = scaler2.fit_transform(hist_df[feature_cols])
            selector2 = VarianceThreshold(threshold=1e-4)
            X_train_filtered = selector2.fit_transform(X_train_scaled)

            gbdt = GradientBoostingRegressor(
                n_estimators=150, learning_rate=0.05, max_depth=4, random_state=42
            )
            gbdt.fit(X_train_filtered, hist_df["合計_実績"])

            X_target_raw2 = hist_df.iloc[[-1]][feature_cols]
            X_target_scaled2 = scaler2.transform(X_target_raw2)
            X_target_filtered2 = selector2.transform(X_target_scaled2)
            total_pred = gbdt.predict(X_target_filtered2)[0]

            total_actual = hist_df.iloc[-1]["合計_実績"]
            all_actual.append(total_actual)
            all_pred.append(total_pred)

            print(
                f"  【ステージ2】 合計 予測: {total_pred:.0f}kg, 実績: {total_actual:.0f}kg"
            )

    if len(all_actual) > 0:
        r2 = r2_score(all_actual, all_pred)
        mae = mean_absolute_error(all_actual, all_pred)
        print("\n===== 完全ウォークフォワード結果 =====")
        print(f"R² = {r2:.3f}, MAE = {mae:,.0f}kg")
    else:
        print("履歴不足で評価不能")
