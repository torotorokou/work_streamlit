import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge, LogisticRegression, ElasticNet
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    GradientBoostingClassifier,
)
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.base import clone
from utils.config_loader import get_path_from_yaml


# --- データ読込 ---
base_dir = get_path_from_yaml("input", section="directories")
df_raw = pd.read_csv(f"{base_dir}/20240501-20250422.csv", encoding="utf-8")[
    ["伝票日付", "正味重量", "品名"]
]
df_2020 = pd.read_csv(f"{base_dir}/2020顧客.csv")[["伝票日付", "商品", "正味重量_明細"]]
df_2021 = pd.read_csv(f"{base_dir}/2021顧客.csv")[["伝票日付", "商品", "正味重量_明細"]]
df_2023 = pd.read_csv(f"{base_dir}/2023_all.csv")[["伝票日付", "商品", "正味重量_明細"]]

# --- 過去データの整形 ---
df_all = pd.concat([df_2020, df_2021, df_2023])
df_all["伝票日付"] = pd.to_datetime(df_all["伝票日付"])
df_all.rename(columns={"商品": "品名", "正味重量_明細": "正味重量"}, inplace=True)

# --- 過去データと新データを結合 ---
df_raw = pd.concat([df_raw, df_all])


# --- 予測＆評価一括処理関数 ---
def train_and_predict_with_holiday(
    df_raw: pd.DataFrame, start_date: str, end_date: str, holidays: list[str]
) -> pd.DataFrame:
    # ライブラリ再インポート（Streamlit用などでも安全）

    # --- 前処理 ---
    df_raw = df_raw.copy()
    df_raw["伝票日付"] = df_raw["伝票日付"].str.replace(r"\(.*\)", "", regex=True)
    df_raw["伝票日付"] = pd.to_datetime(df_raw["伝票日付"], errors="coerce")
    df_raw["正味重量"] = pd.to_numeric(df_raw["正味重量"], errors="coerce")
    df_raw = df_raw.dropna(subset=["正味重量"])
    print(len(df_raw))

    # --- ピボット（日付×品名）→ 正味重量（合計） ---
    df_pivot = (
        df_raw.groupby(["伝票日付", "品名"])["正味重量"].sum().unstack(fill_value=0)
    )
    df_pivot["合計"] = df_pivot.sum(axis=1)

    # --- 特徴量作成 ---
    df_feat = pd.DataFrame(index=df_pivot.index)
    df_feat["混合廃棄物A_前日"] = df_pivot["混合廃棄物A"].shift(1)
    df_feat["混合廃棄物B_前日"] = df_pivot["混合廃棄物B"].shift(1)
    df_feat["合計_前日"] = df_pivot["合計"].shift(1)
    df_feat["合計_3日平均"] = df_pivot["合計"].shift(1).rolling(3).mean()
    df_feat["合計_3日合計"] = df_pivot["合計"].shift(1).rolling(3).sum()
    df_feat["曜日"] = df_feat.index.dayofweek
    df_feat["週番号"] = df_feat.index.isocalendar().week

    # 1台あたり正味重量の前日中央値（中長期傾向）
    daily_avg = df_raw.groupby("伝票日付")["正味重量"].median()
    df_feat["1台あたり正味重量_前日中央値"] = daily_avg.shift(1).expanding().median()

    # 祝日フラグ（1:祝日, 0:平日）
    holiday_dates = pd.to_datetime(holidays)
    df_feat["祝日フラグ"] = df_feat.index.isin(holiday_dates).astype(int)

    # 欠損除去
    df_feat = df_feat.dropna()
    df_pivot = df_pivot.loc[df_feat.index]

    # --- 特徴量と対象品目 ---
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

    # ベースモデル（ステージ1）とメタモデル（ElasticNet）
    base_models = [
        ("elastic", ElasticNet(alpha=0.1, l1_ratio=0.5)),
        ("rf", RandomForestRegressor(n_estimators=100, random_state=42)),
    ]
    meta_model_stage1 = ElasticNet(alpha=0.1, l1_ratio=0.5)

    # ステージ2モデル（合計推論用）
    gbdt_model = GradientBoostingRegressor(
        n_estimators=150, learning_rate=0.05, max_depth=4, random_state=42
    )

    # KFoldでステージ1学習
    kf = KFold(n_splits=5)
    X_features_all = {}
    stacked_preds = {}

    for item in target_items:
        # 混合Aのみ前日中央値も使う
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

        # スタッキング特徴生成
        train_meta = np.zeros((X_train.shape[0], len(base_models)))
        for i, (_, model) in enumerate(base_models):
            for train_idx, val_idx in kf.split(X_train):
                model_ = clone(model)
                model_.fit(X_train.iloc[train_idx], y_train.iloc[train_idx])
                train_meta[val_idx, i] = model_.predict(X_train.iloc[val_idx])

        meta_model_stage1.fit(train_meta, y_train)

        # テスト用メタ特徴量生成
        test_meta = np.column_stack(
            [
                clone(model).fit(X_train, y_train).predict(X_test)
                for _, model in base_models
            ]
        )
        stacked_preds[item] = meta_model_stage1.predict(test_meta)

    # --- ステージ2入力構築 ---
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

    # --- 合計モデル学習 ---
    y_total_final = df_pivot.loc[df_stage1.index, "合計"]
    gbdt_model.fit(df_stage1, y_total_final)

    # --- 分類モデル（警告/注意） ---
    y_total_binary = (y_total_final < 90000).astype(int)
    clf_model = GradientBoostingClassifier(
        n_estimators=100, learning_rate=0.05, max_depth=3, random_state=42
    )
    clf_model.fit(df_stage1.drop(columns=["祝日フラグ"]), y_total_binary)

    # --- 精度評価 ---
    r2 = r2_score(y_total_final, gbdt_model.predict(df_stage1))
    mae = mean_absolute_error(y_total_final, gbdt_model.predict(df_stage1))

    # --- 将来予測（指定期間） ---
    last_date = df_feat.index[-1]
    predict_dates = pd.date_range(start=start_date, end=end_date)
    residuals = y_total_final - gbdt_model.predict(df_stage1)
    bias = residuals.mean()
    std = residuals.std()

    results = []
    for predict_date in predict_dates:
        # 1行分の入力データ作成
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

        # ステージ1予測
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

        # ステージ2予測（合計重量＋信頼区間）
        stage2_input = df_input[
            [f"{item}_予測" for item in target_items]
            + [
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

        # 警告分類
        label = "通常"
        prob = None
        if 85000 <= y_adjusted <= 95000:
            X_clf = stage2_input.drop(columns=["祝日フラグ"])
            prob = clf_model.predict_proba(X_clf)[0][1]
            classification = clf_model.predict(X_clf)[0]
            label = "警告" if classification == 1 else "注意"

        # 結果格納
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
