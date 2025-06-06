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

# 祝日フラグを含むモデルの学習・検証・予測を一括実行する関数


def maesyori():
    base_dir = get_path_from_yaml("input", section="directories")
    df_raw = pd.read_csv(f"{base_dir}/20240501-20250422.csv", encoding="utf-8")
    df_raw = df_raw[["伝票日付", "正味重量", "品名"]]

    df2 = pd.read_csv(f"{base_dir}/2020顧客.csv")[["伝票日付", "商品", "正味重量_明細"]]
    df3 = pd.read_csv(f"{base_dir}/2021顧客.csv")[["伝票日付", "商品", "正味重量_明細"]]
    df4 = pd.read_csv(f"{base_dir}/2023_all.csv")[["伝票日付", "商品", "正味重量_明細"]]

    df_all = pd.concat([df2, df3, df4])
    df_all.rename(columns={"商品": "品名", "正味重量_明細": "正味重量"}, inplace=True)
    df_all["伝票日付"] = pd.to_datetime(df_all["伝票日付"], errors="coerce")

    df_raw = pd.concat([df_raw, df_all])

    # 🔧 修正：str.replaceの前にstr型へ明示的に変換
    df_raw["伝票日付"] = (
        df_raw["伝票日付"].astype(str).str.replace(r"\(.*\)", "", regex=True)
    )
    df_raw["伝票日付"] = pd.to_datetime(df_raw["伝票日付"], errors="coerce")
    df_raw["正味重量"] = pd.to_numeric(df_raw["正味重量"], errors="coerce")

    # df_raw = df_raw.dropna(subset=["正味重量", "伝票日付"])  # 🔒 日付もNaT除去

    return df_raw


def train_model_with_holiday(df_raw: pd.DataFrame, holidays: list[str]) -> dict:
    # --- 特徴量生成 ---
    df_pivot = (
        df_raw.groupby(["伝票日付", "品名"])["正味重量"].sum().unstack(fill_value=0)
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

    daily_avg = df_raw.groupby("伝票日付")["正味重量"].median()
    df_feat["1台あたり正味重量_前日中央値"] = daily_avg.shift(1).expanding().median()

    holiday_dates = pd.to_datetime(holidays)
    df_feat["祝日フラグ"] = df_feat.index.isin(holiday_dates).astype(int)

    df_feat = df_feat.dropna()
    df_pivot = df_pivot.loc[df_feat.index]

    # --- 学習用設定 ---
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

    base_models = [
        ("elastic", ElasticNet(alpha=0.1, l1_ratio=0.5)),
        ("rf", RandomForestRegressor(n_estimators=100, random_state=42)),
    ]
    meta_model_stage1 = ElasticNet(alpha=0.1, l1_ratio=0.5)
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
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=False
        )
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
        stacked_preds[item] = meta_model_stage1.predict(test_meta)

        # --- 汎化性能の表示 ---
        y_train_pred = meta_model_stage1.predict(train_meta)
        y_test_pred = stacked_preds[item]

        print(f"\n📘 {item}")
        print(f"　　R² (train) = {r2_score(y_train, y_train_pred):.3f}")
        print(f"　　R² (test)  = {r2_score(y_test, y_test_pred):.3f}")
        print(f"　　MAE (train) = {mean_absolute_error(y_train, y_train_pred):,.0f} kg")
        print(f"　　MAE (test)  = {mean_absolute_error(y_test, y_test_pred):,.0f} kg")

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

    # --- 合計モデル学習と評価 ---
    y_total_final = df_pivot.loc[df_stage1.index, "合計"]
    gbdt_model.fit(df_stage1, y_total_final)
    y_pred_total = gbdt_model.predict(df_stage1)

    print("\n📘 合計予測モデル（GBDT）")
    print(f"　　R² = {r2_score(y_total_final, y_pred_total):.3f}")
    print(f"　　MAE = {mean_absolute_error(y_total_final, y_pred_total):,.0f} kg")

    # --- 分類モデル学習 ---
    y_total_binary = (y_total_final < 90000).astype(int)
    clf_model.fit(df_stage1.drop(columns=["祝日フラグ"]), y_total_binary)

    residuals = y_total_final - y_pred_total

    return {
        "meta_model": meta_model_stage1,
        "base_models": base_models,
        "gbdt_model": gbdt_model,
        "clf_model": clf_model,
        "X_features_all": X_features_all,
        "df_feat": df_feat,
        "df_pivot": df_pivot,
        "ab_features": ab_features,
        "target_items": target_items,
        "holiday_dates": holiday_dates,
        "bias": residuals.mean(),
        "std": residuals.std(),
    }


def predict_with_model(
    model_dict: dict, start_date: str, end_date: str
) -> pd.DataFrame:
    results = []
    predict_dates = pd.date_range(start=start_date, end=end_date)
    last_date = model_dict["df_feat"].index[-1]

    for predict_date in predict_dates:
        new_row = {
            "混合廃棄物A_前日": model_dict["df_pivot"].loc[last_date, "混合廃棄物A"],
            "混合廃棄物B_前日": model_dict["df_pivot"].loc[last_date, "混合廃棄物B"],
            "合計_前日": model_dict["df_pivot"].loc[last_date, "合計"],
            "合計_3日平均": model_dict["df_pivot"]["合計"]
            .shift(1)
            .rolling(3)
            .mean()
            .loc[last_date],
            "合計_3日合計": model_dict["df_pivot"]["合計"]
            .shift(1)
            .rolling(3)
            .sum()
            .loc[last_date],
            "曜日": predict_date.dayofweek,
            "週番号": predict_date.isocalendar().week,
            "1台あたり正味重量_前日中央値": model_dict["df_feat"][
                "1台あたり正味重量_前日中央値"
            ].iloc[-1],
            "祝日フラグ": int(predict_date in model_dict["holiday_dates"]),
        }
        df_input = pd.DataFrame(new_row, index=[predict_date])
        for item in model_dict["target_items"]:
            x_item = (
                df_input[model_dict["ab_features"]]
                if item == "混合廃棄物A"
                else df_input[
                    [c for c in model_dict["ab_features"] if "1台あたり" not in c]
                ]
            )
            meta_input = np.column_stack(
                [
                    clone(model)
                    .fit(
                        model_dict["X_features_all"][item],
                        model_dict["df_pivot"].loc[model_dict["df_feat"].index, item],
                    )
                    .predict(x_item)
                    for _, model in model_dict["base_models"]
                ]
            )
            df_input[f"{item}_予測"] = model_dict["meta_model"].predict(meta_input)[0]

        stage2_input = df_input[
            [
                f"{model_dict['target_items'][0]}_予測",
                f"{model_dict['target_items'][1]}_予測",
                f"{model_dict['target_items'][2]}_予測",
                "曜日",
                "週番号",
                "合計_前日",
                "1台あたり正味重量_前日中央値",
                "祝日フラグ",
            ]
        ]
        y_pred = model_dict["gbdt_model"].predict(stage2_input)[0]
        y_adjusted = y_pred + model_dict["bias"]
        lower = y_adjusted - 1.96 * model_dict["std"]
        upper = y_adjusted + 1.96 * model_dict["std"]

        label = "通常"
        prob = None
        if 85000 <= y_adjusted <= 95000:
            X_clf = stage2_input.drop(columns=["祝日フラグ"])
            prob = model_dict["clf_model"].predict_proba(X_clf)[0][1]
            classification = model_dict["clf_model"].predict(X_clf)[0]
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

    return pd.DataFrame(results).set_index("日付")


if __name__ == "__main__":
    # データの読み込みと前処理
    df_raw = load_data_from_sqlite()

    #
    df_raw = load_data_from_sqlite()
    df_raw["伝票日付"] = pd.to_datetime(df_raw["伝票日付"])
    df_raw = df_raw[df_raw["伝票日付"] >= "2024-04-01"]

    # 祝日データの取得
    start_date = df_raw["伝票日付"].min().date()
    end_date = df_raw["伝票日付"].max().date()
    holidays = get_japanese_holidays(start=start_date, end=end_date, as_str=True)

    # モデルの学習
    make_model = train_model_with_holiday(df_raw, holidays)

    # 予測の実行
    df_pred = predict_with_model(make_model, "2025-06-01", "2025-06-07")

    print(df_pred)
