import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from logic.factory_manage.predict_model_v4_5 import full_walkforward_pipeline


def history_window_search(df_raw, window_list, min_eval_data=10):
    """
    df_raw : 元データ
    window_list : 試す履歴データ日数のリスト (start_index相当)
    min_eval_data : 評価用データ数がこの数より少ない場合はスキップ
    """
    df_raw["伝票日付"] = pd.to_datetime(df_raw["伝票日付"])
    df_raw = df_raw.sort_values("伝票日付")
    all_dates = pd.to_datetime(np.sort(df_raw["伝票日付"].unique()))

    results = []

    for window in window_list:
        if window >= len(all_dates) - 5:
            print(f"\n履歴{window}日: 評価可能データがほぼ無いためスキップ")
            continue

        print(f"\n履歴{window}日で評価中...")
        all_actual, all_pred = full_walkforward_pipeline(df_raw, start_index=window)

        if len(all_actual) < min_eval_data:
            print(f"  -> 評価データが少ないためスキップ ({len(all_actual)}件)")
            continue

        y_true = np.array(all_actual)
        y_pred = np.array(all_pred)
        r2 = r2_score(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        max_err = np.max(np.abs(y_true - y_pred))

        print(
            f"  R² = {r2:.3f}, MAE = {mae:,.0f}kg, RMSE = {rmse:,.0f}kg, MAPE = {mape:.2f}%, 最大誤差={max_err:,.0f}kg"
        )

        results.append(
            {
                "履歴日数": window,
                "評価件数": len(y_true),
                "R2": r2,
                "MAE": mae,
                "RMSE": rmse,
                "MAPE": mape,
                "最大誤差": max_err,
            }
        )

    # 結果をDataFrameで返す
    result_df = pd.DataFrame(results)
    return result_df
