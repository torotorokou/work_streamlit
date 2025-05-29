import pandas as pd
import numpy as np
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.base import clone
from utils.get_holydays import get_japanese_holidays
import joblib
from utils.config_loader import get_path_from_yaml
from datetime import datetime


def predict_with_saved_models(
    start_date: str, end_date: str, holidays: list[str]
) -> pd.DataFrame:
    # モデルパスの設定
    model_dir = get_path_from_yaml(
        ["models", "predicted_import_volume"], section="directories"
    )
    print(model_dir)

    # --- モデル・特徴量・前処理済みデータ読み込み ---
    meta_model_stage1 = joblib.load(f"{model_dir}/meta_model_stage1.pkl")
    gbdt_model = joblib.load(f"{model_dir}/gbdt_model_stage2.pkl")
    clf_model = joblib.load(f"{model_dir}/clf_model.pkl")
    ab_features = joblib.load(f"{model_dir}/ab_features.pkl")
    X_features_all = joblib.load(f"{model_dir}/X_features_all.pkl")
    df_feat = joblib.load(f"{model_dir}/df_feat.pkl")
    df_pivot = joblib.load(f"{model_dir}/df_pivot.pkl")

    # --- 基本情報 ---
    target_items = ["混合廃棄物A", "混合廃棄物B", "混合廃棄物(ｿﾌｧｰ･家具類)"]
    holiday_dates = pd.to_datetime(holidays)

    # --- 信頼区間用バイアスと誤差 ---
    index_final = df_feat.index.intersection(X_features_all["混合廃棄物A"].index)
    df_stage1_pred = pd.DataFrame(index=index_final)
    for item in target_items:
        df_stage1_pred[f"{item}_予測"] = np.nan  # 初期化

    for item in target_items:
        X_item = X_features_all[item].loc[index_final]
        meta_input = np.column_stack(
            [
                clone(model)
                .fit(X_item, df_pivot.loc[index_final, item])
                .predict(X_item)
                for _, model in [
                    ("elastic", ElasticNet(alpha=0.1, l1_ratio=0.5)),
                    ("rf", RandomForestRegressor(n_estimators=100, random_state=42)),
                ]
            ]
        )
        df_stage1_pred[f"{item}_予測"] = meta_model_stage1.predict(meta_input)

    for col in [
        "曜日",
        "週番号",
        "合計_前日",
        "1台あたり正味重量_前日中央値",
        "祝日フラグ",
    ]:
        df_stage1_pred[col] = df_feat.loc[index_final, col]

    y_total_actual = df_pivot.loc[index_final, "合計"]
    y_total_pred = gbdt_model.predict(df_stage1_pred)
    bias = (y_total_actual - y_total_pred).mean()
    std = (y_total_actual - y_total_pred).std()

    # --- 予測処理（指定期間） ---
    last_date = df_feat.index[-1]
    predict_dates = pd.date_range(start=start_date, end=end_date)
    results = []

    for predict_date in predict_dates:
        row = {
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

        df_input = pd.DataFrame(row, index=[predict_date])

        # ステージ1予測を追加
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
                    for _, model in [
                        ("elastic", ElasticNet(alpha=0.1, l1_ratio=0.5)),
                        (
                            "rf",
                            RandomForestRegressor(n_estimators=100, random_state=42),
                        ),
                    ]
                ]
            )
            df_input[f"{item}_予測"] = meta_model_stage1.predict(meta_input)[0]

        # ステージ2予測
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

        # 分類予測（警告 or 注意）
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

    df_result = pd.DataFrame(results).set_index("日付")
    return df_result


import pandas as pd
from sqlalchemy import create_engine
from typing import List, Union
from datetime import date
from utils.config_loader import get_path_from_yaml


def get_holidays_from_sql(
    start: date, end: date, as_str: bool = True
) -> List[Union[date, str]]:
    """
    SQLiteから指定期間の祝日を取得する関数

    Parameters:
        start (date): 開始日
        end (date): 終了日
        as_str (bool): Trueなら'YYYY-MM-DD'形式、Falseならdate型のまま返す

    Returns:
        List[str or date]: 指定期間内の祝日一覧
    """
    db_path = get_path_from_yaml("weight_data", section="sql_database")
    engine = create_engine(f"sqlite:///{db_path}")

    query = f"""
        SELECT DISTINCT 伝票日付
        FROM ukeire
        WHERE 祝日フラグ = 1
        AND 伝票日付 BETWEEN '{start}' AND '{end}'
        ORDER BY 伝票日付
    """

    df = pd.read_sql(query, engine)

    if df.empty:
        return []

    if as_str:
        return df["伝票日付"].dt.strftime("%Y-%m-%d").tolist()
    else:
        return df["伝票日付"].dt.date.tolist()


if __name__ == "__main__":
    start_date = datetime.strptime("2025-05-20", "%Y-%m-%d").date()
    end_date = datetime.strptime("2025-05-27", "%Y-%m-%d").date()

    holidays = get_holidays_from_sql(start=start_date, end=end_date, as_str=True)
    print(holidays)

    start_date_str = start_date.strftime("%Y-%m-%d")
    end_date_str = end_date.strftime("%Y-%m-%d")
    df_result = predict_with_saved_models(start_date_str, end_date_str, holidays)

    print(df_result)
