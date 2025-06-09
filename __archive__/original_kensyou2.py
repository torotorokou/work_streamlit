import pandas as pd
import numpy as np
import joblib

# === モデル ===
from sklearn.linear_model import Ridge, ElasticNet, LogisticRegression
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    GradientBoostingClassifier,
)

# === 学習系 ===
from sklearn.model_selection import train_test_split, KFold
from sklearn.base import clone

# === 評価指標 ===
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    accuracy_score,
    roc_auc_score,
)

# === ユーティリティ ===
from utils.config_loader import get_path_from_yaml
from utils.get_holydays import get_japanese_holidays
from logic.factory_manage.sql import load_data_from_sqlite


# =======================
# 1. データ整形処理
# =======================
def prepare_training_data(
    df_raw: pd.DataFrame, holidays: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df_raw = df_raw.copy()
    df_raw["伝票日付"] = pd.to_datetime(df_raw["伝票日付"], errors="coerce")
    df_raw["正味重量"] = pd.to_numeric(df_raw["正味重量"], errors="coerce")
    df_raw = df_raw.dropna(subset=["正味重量"])

    # --- 品目の統合（出現数が少ない品目を「その他」へ） ---
    count_threshold = 30
    important_items = ["混合廃棄物A", "混合廃棄物B", "混合廃棄物(ｿﾌｧｰ･家具類)"]
    item_counts = df_raw["品名"].value_counts()
    rare_items = item_counts[item_counts < count_threshold].index.difference(
        important_items
    )
    df_raw["品名"] = df_raw["品名"].apply(lambda x: "その他" if x in rare_items else x)

    # --- ピボット ---
    df_pivot = (
        df_raw.groupby(["伝票日付", "品名"])["正味重量"].sum().unstack(fill_value=0)
    )
    df_pivot["合計"] = df_pivot.sum(axis=1)

    # --- 特徴量作成 ---
    df_feat = pd.DataFrame(index=df_pivot.index)
    df_feat["混合廃棄物A_前日"] = df_pivot.get(
        "混合廃棄物A", pd.Series(0, index=df_pivot.index)
    ).shift(1)
    df_feat["混合廃棄物B_前日"] = df_pivot.get(
        "混合廃棄物B", pd.Series(0, index=df_pivot.index)
    ).shift(1)
    df_feat["合計_前日"] = df_pivot["合計"].shift(1)
    df_feat["合計_3日平均"] = df_pivot["合計"].shift(1).rolling(3).mean()
    df_feat["合計_3日合計"] = df_pivot["合計"].shift(1).rolling(3).sum()
    df_feat["曜日"] = df_feat.index.dayofweek
    df_feat["週番号"] = df_feat.index.isocalendar().week

    # --- 1台あたり正味重量の前日中央値 ---
    daily_avg = df_raw.groupby("伝票日付")["正味重量"].median()
    df_feat["1台あたり正味重量_前日中央値"] = daily_avg.shift(1).expanding().median()

    # --- 祝日フラグ ---
    holiday_dates = pd.to_datetime(holidays)
    df_feat["祝日フラグ"] = df_feat.index.isin(holiday_dates).astype(int)

    # --- 欠損除去・同期 ---
    df_feat = df_feat.dropna()
    df_pivot = df_pivot.loc[df_feat.index]

    return df_feat, df_pivot


# =======================
# 2. モデル作成処理（最終安定版：収束対策強化 + 評価出力）
# =======================
def train_models(df_feat: pd.DataFrame, df_pivot: pd.DataFrame) -> dict:
    from sklearn.linear_model import ElasticNet
    from sklearn.ensemble import (
        RandomForestRegressor,
        GradientBoostingRegressor,
        GradientBoostingClassifier,
    )
    from sklearn.model_selection import KFold
    from sklearn.base import clone
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_absolute_error, r2_score
    import numpy as np

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

    elastic_pipeline = make_pipeline(
        StandardScaler(),
        ElasticNet(
            alpha=1.0, l1_ratio=0.5, max_iter=30000, tol=1e-2, selection="cyclic"
        ),
    )

    base_models = [
        ("elastic", elastic_pipeline),
        ("rf", RandomForestRegressor(n_estimators=100, random_state=42)),
    ]
    meta_model_stage1 = make_pipeline(
        StandardScaler(),
        ElasticNet(
            alpha=1.0, l1_ratio=0.5, max_iter=30000, tol=1e-2, selection="cyclic"
        ),
    )
    gbdt_model = GradientBoostingRegressor(
        n_estimators=150, learning_rate=0.05, max_depth=4, random_state=42
    )
    clf_model = GradientBoostingClassifier(
        n_estimators=100, learning_rate=0.05, max_depth=3, random_state=42
    )
    kf = KFold(n_splits=5)

    X_features_all = {}
    stacked_preds = {}

    for item in target_items:
        X = (
            df_feat[ab_features]
            if item == "混合廃棄物A"
            else df_feat[[c for c in ab_features if "1台あたり" not in c]]
        )
        y = df_pivot[item]

        split_idx = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        X_features_all[item] = X

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
        preds = meta_model_stage1.predict(test_meta)
        stacked_preds[item] = preds

        # 評価指標表示
        r2 = r2_score(y_test, preds)
        mae = mean_absolute_error(y_test, preds)
        print(f"📘 {item} ステージ1 R² = {r2:.3f}, MAE = {mae:,.0f} kg")

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

    y_total_binary = (y_total_final < 90000).astype(int)
    clf_model.fit(df_stage1.drop(columns=["祝日フラグ"]), y_total_binary)

    # ステージ2評価出力
    stage2_preds = gbdt_model.predict(df_stage1)
    stage2_r2 = r2_score(y_total_final, stage2_preds)
    stage2_mae = mean_absolute_error(y_total_final, stage2_preds)
    print(f"📘 ステージ2合計モデル R² = {stage2_r2:.3f}, MAE = {stage2_mae:,.0f} kg")

    return {
        "base_models": base_models,
        "meta_model_stage1": meta_model_stage1,
        "stage2_model": gbdt_model,
        "clf_model": clf_model,
        "X_features_all": X_features_all,
        "df_feat": df_feat,
        "df_pivot": df_pivot,
        "target_items": target_items,
        "ab_features": ab_features,
        "df_stage1": df_stage1,
    }


# =======================
# 3. 将来予測処理（特徴量逐次更新型）
# =======================
def forecast_future(
    df_feat: pd.DataFrame,
    df_pivot: pd.DataFrame,
    models: dict,
    start_date: str,
    end_date: str,
    holidays: list[str],
) -> pd.DataFrame:
    import pandas as pd
    from sklearn.base import clone
    import numpy as np

    base_models = models["base_models"]
    meta_model_stage1 = models["meta_model_stage1"]
    gbdt_model = models["stage2_model"]
    clf_model = models["clf_model"]
    X_features_all = models["X_features_all"]
    target_items = models["target_items"]
    ab_features = models["ab_features"]
    df_stage1 = models["df_stage1"]

    predict_dates = pd.date_range(start=start_date, end=end_date)
    holiday_dates = pd.to_datetime(holidays)

    # --- 初期状態を保持 ---
    last_date = df_feat.index[-1]
    last_feat = df_feat.loc[last_date].copy()
    last_pivot = df_pivot.loc[last_date].copy()
    last_median = last_feat["1台あたり正味重量_前日中央値"]

    y_total_final = df_pivot.loc[df_stage1.index, "合計"]
    residuals = y_total_final - gbdt_model.predict(df_stage1)
    bias = residuals.mean()
    std = residuals.std()

    results = []

    for predict_date in predict_dates:
        # --- 特徴量を逐次更新 ---
        new_feat = {
            "混合廃棄物A_前日": last_pivot["混合廃棄物A"],
            "混合廃棄物B_前日": last_pivot["混合廃棄物B"],
            "合計_前日": last_pivot["合計"],
            "合計_3日平均": df_pivot["合計"].shift(1).rolling(3).mean().iloc[-1],
            "合計_3日合計": df_pivot["合計"].shift(1).rolling(3).sum().iloc[-1],
            "曜日": predict_date.dayofweek,
            "週番号": predict_date.isocalendar().week,
            "1台あたり正味重量_前日中央値": last_median,
            "祝日フラグ": int(predict_date in holiday_dates),
        }
        df_input = pd.DataFrame(new_feat, index=[predict_date])

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

        # --- ステージ2予測 ---
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

        # --- 予測結果を次回の入力に反映 ---
        last_pivot["混合廃棄物A"] = df_input[f"{target_items[0]}_予測"]
        last_pivot["混合廃棄物B"] = df_input[f"{target_items[1]}_予測"]
        last_pivot["混合廃棄物(ｿﾌｧｰ･家具類)"] = df_input[f"{target_items[2]}_予測"]
        last_pivot["合計"] = sum(
            [
                df_input[f"{target_items[0]}_予測"],
                df_input[f"{target_items[1]}_予測"],
                df_input[f"{target_items[2]}_予測"],
            ]
        )
        df_pivot.loc[predict_date] = last_pivot

    return pd.DataFrame(results).set_index("日付")


if __name__ == "__main__":
    # データの読み込みと前処理
    df_raw = load_data_from_sqlite()

    start_date = df_raw["伝票日付"].min().date()
    end_date = df_raw["伝票日付"].max().date()
    holidays = get_japanese_holidays(start=start_date, end=end_date, as_str=True)

    df_feat, df_pivot = prepare_training_data(df_raw, holidays)
    models = train_models(df_feat, df_pivot)

    # --- 🔍 モデルのステージ1予測 vs 実測 の確認 ---
    import matplotlib.pyplot as plt

    df_stage1 = models["df_stage1"]
    y_true = df_pivot.loc[df_stage1.index, "合計"]
    y_pred = models["stage2_model"].predict(df_stage1)

    # R²とMAE表示
    from sklearn.metrics import r2_score, mean_absolute_error

    print(
        f"📘 ステージ2合計モデル R² = {r2_score(y_true, y_pred):.3f}, MAE = {mean_absolute_error(y_true, y_pred):,.0f} kg"
    )

    # プロット
    plt.figure(figsize=(10, 4))
    plt.plot(y_true.index, y_true, label="実測 合計")
    plt.plot(y_true.index, y_pred, label="予測 合計", linestyle="--")
    plt.axvline(pd.to_datetime("2025-06-01"), color="red", linestyle=":")
    plt.legend()
    plt.title("ステージ2予測 vs 実測")
    plt.grid()
    plt.tight_layout()
    plt.savefig("/work/pred_vs_true.png")  # Docker内なので保存
    print("📊 /work/pred_vs_true.png にグラフを保存しました")

    # --- ✅ 予測 ---
    df_pred = forecast_future(
        df_feat, df_pivot, models, "2025-05-27", "2025-05-27", holidays
    )

    print(df_pred)
