import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge, LogisticRegression, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.base import clone
from sklearn.metrics import accuracy_score, roc_auc_score
import joblib
from utils.config_loader import get_path_from_yaml
import pandas as pd
import numpy as np
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    GradientBoostingClassifier,
)
from sklearn.model_selection import train_test_split, KFold
from sklearn.base import clone
from sklearn.metrics import r2_score, mean_absolute_error
from utils.get_holydays import get_japanese_holidays
from utils.config_loader import get_path_from_yaml
from logic.factory_manage.sql import load_data_from_sqlite

# data_prep.py
import pandas as pd
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    GradientBoostingClassifier,
)
from sklearn.model_selection import TimeSeriesSplit


def prepare_features(
    df_raw: pd.DataFrame, holidays: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    データから特徴量とターゲット（df_pivot）を作成する

    Args:
        df_raw (pd.DataFrame): 元データ
        holidays (list[str]): 祝日リスト

    Returns:
        tuple:
            - df_feat (pd.DataFrame): 特徴量データ
            - df_pivot (pd.DataFrame): 品目別日別合計データ（ターゲット含む）
    """
    # --- ピボット作成（品目別合計） ---
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

    # --- 1台あたり正味重量の前日中央値（累積） ---
    daily_avg = df_raw.groupby("伝票日付")["正味重量"].median()
    df_feat["1台あたり正味重量_前日中央値"] = daily_avg.shift(1).expanding().median()

    # --- 祝日フラグ ---
    holiday_dates = pd.to_datetime(holidays)
    df_feat["祝日フラグ"] = df_feat.index.isin(holiday_dates).astype(int)

    # --- 欠損除去 ---
    df_feat = df_feat.dropna()
    df_pivot = df_pivot.loc[df_feat.index]

    return df_feat, df_pivot


# model_stage1.py
from sklearn.base import clone
import numpy as np


def train_stage1_models(
    df_feat, df_pivot, tscv, base_models, meta_model_stage1, ab_features, target_items
):
    """
    ステージ1の学習（スタッキング用メタ特徴量生成）

    Args:
        df_feat (pd.DataFrame): 特徴量データ
        df_pivot (pd.DataFrame): 品目別合計データ
        tscv (TimeSeriesSplit): 時系列CV
        base_models (list): ベースモデルのリスト
        meta_model_stage1: メタモデル（ElasticNet）
        ab_features (list): 使用特徴量名
        target_items (list): 対象品目

    Returns:
        tuple:
            - X_features_all (dict): 各品目の特徴量データ
            - stacked_preds (dict): 各品目のテスト予測値
    """
    X_features_all = {}
    stacked_preds = {}

    for item in target_items:
        # --- 品目ごとに特徴量を選択 ---
        X = (
            df_feat[ab_features]
            if item == "混合廃棄物A"
            else df_feat[[c for c in ab_features if "1台あたり" not in c]]
        )
        y = df_pivot[item]

        # --- 時系列の最後20%をテストに使用 ---
        test_size = int(len(X) * 0.2)
        X_train = X.iloc[:-test_size]
        X_test = X.iloc[-test_size:]
        y_train = y.iloc[:-test_size]
        y_test = y.iloc[-test_size:]

        X_features_all[item] = X

        # --- スタッキング用メタ特徴量作成 ---
        train_meta = np.zeros((X_train.shape[0], len(base_models)))
        for i, (_, model) in enumerate(base_models):
            for train_idx, val_idx in tscv.split(X_train):
                model_ = clone(model)
                model_.fit(X_train.iloc[train_idx], y_train.iloc[train_idx])
                train_meta[val_idx, i] = model_.predict(X_train.iloc[val_idx])

        # --- メタモデル学習 ---
        meta_model_stage1.fit(train_meta, y_train)

        # --- テスト予測 ---
        test_meta = np.column_stack(
            [
                clone(model).fit(X_train, y_train).predict(X_test)
                for _, model in base_models
            ]
        )
        stacked_preds[item] = meta_model_stage1.predict(test_meta)

    return X_features_all, stacked_preds


# model_stage2.py
from sklearn.metrics import r2_score, mean_absolute_error


def train_stage2_models(df_stage1, df_pivot, gbdt_model, clf_model):
    """
    ステージ2モデル（GBDT回帰 + 分類器）学習

    Args:
        df_stage1 (pd.DataFrame): ステージ1の出力特徴量
        df_pivot (pd.DataFrame): 品目別合計データ
        gbdt_model: 回帰モデル（GBDT）
        clf_model: 分類モデル（GBC）

    Returns:
        tuple:
            - gbdt_model: 学習済み回帰モデル
            - clf_model: 学習済み分類モデル
            - r2 (float): R²スコア
            - mae (float): MAE
    """
    # --- 回帰学習 ---
    y_total_final = df_pivot.loc[df_stage1.index, "合計"]
    gbdt_model.fit(df_stage1, y_total_final)

    # --- 分類学習（90000未満 or 以上） ---
    y_total_binary = (y_total_final < 90000).astype(int)
    clf_model.fit(df_stage1.drop(columns=["祝日フラグ"]), y_total_binary)

    # --- 評価 ---
    r2 = r2_score(y_total_final, gbdt_model.predict(df_stage1))
    mae = mean_absolute_error(y_total_final, gbdt_model.predict(df_stage1))

    return gbdt_model, clf_model, r2, mae


# predict.py
import numpy as np
import pandas as pd
from sklearn.base import clone


def predict_future(
    df_feat,
    df_pivot,
    df_stage1,
    X_features_all,
    meta_model_stage1,
    gbdt_model,
    clf_model,
    base_models,
    target_items,
    ab_features,
    start_date,
    end_date,
    holidays,
):
    """
    将来期間に対する予測（回帰 + 分類ラベル付け）

    Args:
        df_feat (pd.DataFrame): 特徴量データ
        df_pivot (pd.DataFrame): 品目別合計データ
        df_stage1 (pd.DataFrame): ステージ1の出力特徴量
        X_features_all (dict): 各品目の特徴量
        meta_model_stage1: ステージ1メタモデル
        gbdt_model: 学習済み回帰モデル
        clf_model: 学習済み分類モデル
        base_models (list): ベースモデルのリスト
        target_items (list): 対象品目
        ab_features (list): 使用特徴量名
        start_date (str): 予測開始日
        end_date (str): 予測終了日
        holidays (list): 祝日リスト

    Returns:
        pd.DataFrame: 予測結果
    """
    holiday_dates = pd.to_datetime(holidays)
    last_date = df_feat.index[-1]
    predict_dates = pd.date_range(start=start_date, end=end_date)

    # --- バイアス・標準偏差計算 ---
    y_total_final = df_pivot.loc[df_stage1.index, "合計"]
    residuals = y_total_final - gbdt_model.predict(df_stage1)
    bias = residuals.mean()
    std = residuals.std()

    results = []
    for predict_date in predict_dates:
        # --- 1日の特徴量生成 ---
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
        df_input = pd.DataFrame(new_row, index=[predict_date])

        # --- ステージ1予測 ---
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

        # --- 判定ラベル ---
        label = "通常"
        prob = None
        if 85000 <= y_adjusted <= 95000:
            X_clf = stage2_input.drop(columns=["祝日フラグ"])
            prob = clf_model.predict_proba(X_clf)[0][1]
            classification = clf_model.predict(X_clf)[0]
            label = "警告" if classification == 1 else "注意"

        # --- 結果格納 ---
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
    return df_result


def get_df():
    base_dir = get_path_from_yaml("input", section="directories")

    df_raw = pd.read_csv(f"{base_dir}/20240501-20250422.csv", encoding="utf-8")
    df_raw = df_raw[["伝票日付", "正味重量", "品名"]]
    df2 = pd.read_csv(f"{base_dir}/2020顧客.csv")
    df3 = pd.read_csv(f"{base_dir}/2021顧客.csv")
    df4 = pd.read_csv(f"{base_dir}/2023_all.csv")

    df2 = df2[["伝票日付", "商品", "正味重量_明細"]]
    df3 = df3[["伝票日付", "商品", "正味重量_明細"]]
    df4 = df4[["伝票日付", "商品", "正味重量_明細"]]

    df_all = pd.concat([df2, df3, df4])
    df_all["伝票日付"] = pd.to_datetime(df_all["伝票日付"])

    df_all.rename(columns={"商品": "品名", "正味重量_明細": "正味重量"}, inplace=True)

    df_raw = pd.concat([df_raw, df_all])
    df_raw = df_raw.copy()
    df_raw["伝票日付"] = df_raw["伝票日付"].str.replace(r"\(.*\)", "", regex=True)
    df_raw["伝票日付"] = pd.to_datetime(df_raw["伝票日付"], errors="coerce")
    df_raw["正味重量"] = pd.to_numeric(df_raw["正味重量"], errors="coerce")
    df_raw = df_raw.dropna(subset=["正味重量", "伝票日付"])
    return df_raw


def get_date_holidays(df):
    """
    df内の祝日フラグ=1の日付を一意に取得し、start_date～end_dateの範囲内で返す

    Args:
        df (pd.DataFrame): データ（'伝票日付'、'祝日フラグ'カラムが含まれていること）

    Returns:
        list[str]: 祝日の日付（YYYY-MM-DD 形式）のリスト
    """

    start_date = df["伝票日付"].min().date()
    end_date = df["伝票日付"].max().date()

    # print(f"🔍 祝日抽出範囲: {start_date} ～ {end_date}")

    # --- 祝日フラグが1の行のみ抽出 ---
    mask = df["祝日フラグ"] == 1
    holidays_series = df.loc[mask, "伝票日付"]

    # --- 重複除去 & 日付範囲内で絞り込み ---
    holidays = holidays_series.drop_duplicates()
    holidays = holidays[
        (holidays.dt.date >= start_date) & (holidays.dt.date <= end_date)
    ]

    # --- 日付型を文字列（YYYY-MM-DD）に変換してリスト化 ---
    holidays_list = holidays.dt.strftime("%Y-%m-%d").tolist()

    return holidays_list


def debug(df):
    """
    デバッグ用の関数。DataFrameの基本情報を表示する。

    Args:
        df (pd.DataFrame): デバッグ対象のDataFrame
    """
    df = df["伝票日付"].unique()
    for i in range(len(df)):
        print(df[i])
    return None


def predict_controller(start_date, end_date):
    # --- データ読み込み ---
    df_raw = load_data_from_sqlite()

    # df_rawからholidaysを取得&整形
    holidays = get_date_holidays(df_raw)
    df_raw = df_raw[["伝票日付", "正味重量", "品名"]].copy()

    # デバッグ用
    # debug(get_df())

    # --- パラメータ ---
    target_items = ["混合廃棄物A", "混合廃棄物B", "混合廃棄物(ｿﾌｧｰ･家具類)"]
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
    tscv = TimeSeriesSplit(n_splits=5)

    # --- 特徴量作成 ---
    df_feat, df_pivot = prepare_features(df_raw, holidays)

    # --- ステージ1学習 ---
    X_features_all, stacked_preds = train_stage1_models(
        df_feat,
        df_pivot,
        tscv,
        base_models,
        meta_model_stage1,
        ab_features,
        target_items,
    )

    # --- ステージ2学習 ---
    index_final = df_feat.iloc[int(len(df_feat) * 0.8) :].index
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

    gbdt_model, clf_model, r2, mae = train_stage2_models(
        df_stage1, df_pivot, gbdt_model, clf_model
    )
    print(f"✅ R² = {r2:.3f}, MAE = {mae:,.0f} kg")

    # --- 将来予測 ---
    df_result = predict_future(
        df_feat,
        df_pivot,
        df_stage1,  # 追加済みOK
        X_features_all,
        meta_model_stage1,
        gbdt_model,
        clf_model,
        base_models,
        target_items,
        ab_features,  # ← これを忘れずに入れる
        start_date=start_date,
        end_date=end_date,
        holidays=holidays,
    )

    return df_result


if __name__ == "__main__":
    df_result = predict_controller("2025-06-01", "2025-06-07")
    print(df_result)
