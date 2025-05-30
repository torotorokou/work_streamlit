import pandas as pd
import joblib
from typing import List
from logic.factory_manage.make_df import make_stage1_df, make_stage2_df
from logic.factory_manage.predict_model_ver2


def debug_print(title, start_date, end_date, df):
    print(f"\n📌 {title}")
    print(f"予測対象日: {start_date} ～ {end_date}")
    print("特徴量の一覧:", df.columns.tolist())
    print("データ型:\n", df.dtypes)
    print("欠損数:\n", df.isnull().sum())
    print("サンプル入力（直近）:")
    print(df.tail(1))
    print("=" * 50)


def predict_with_saved_models(
    start_date: str, end_date: str, holidays: List[str], model_dir: str
) -> pd.DataFrame:
    # --- ステージ1特徴量の作成 ---
    df_stage1 = make_stage1_df(start_date, end_date, holidays)

    # --- ステージ1モデルで予測 ---
    model_a = joblib.load(f"{model_dir}/model_a.pkl")
    model_b = joblib.load(f"{model_dir}/model_b.pkl")
    model_c = joblib.load(f"{model_dir}/model_c.pkl")

    df_stage1["混合廃棄物A_予測"] = model_a.predict(df_stage1)
    df_stage1["混合廃棄物B_予測"] = model_b.predict(df_stage1)
    df_stage1["混合廃棄物(ｿﾌｧｰ･家具類)_予測"] = model_c.predict(df_stage1)

    df_stage1_pred = df_stage1[
        ["混合廃棄物A_予測", "混合廃棄物B_予測", "混合廃棄物(ｿﾌｧｰ･家具類)_予測"]
    ].copy()

    # --- ステージ2入力の作成 ---
    df_input = make_stage2_df(df_stage1_pred, start_date, end_date, holidays)

    stage2_input = df_input[
        [
            "混合廃棄物A_予測",
            "混合廃棄物B_予測",
            "混合廃棄物(ｿﾌｧｰ･家具類)_予測",
            "曜日",
            "週番号",
            "合計_前日",
            "1台あたり正味重量_前日中央値",
            "祝日フラグ",
        ]
    ].copy()

    # --- ステージ2予測 ---
    model_total = joblib.load(f"{model_dir}/model_total.pkl")
    y_pred = model_total.predict(stage2_input)

    # --- 結果のまとめ ---
    results = []
    for i, pred in enumerate(y_pred):
        predict_date = stage2_input.index[i]
        results.append({"日付": predict_date, "合計_予測": round(pred, 2)})

    df_result = pd.DataFrame(results).set_index("日付")

    # --- デバッグ出力（最後の状態を確認） ---
    debug_print("df_stage1", start_date, end_date, df_stage1_pred)
    debug_print("df_input", start_date, end_date, df_input)
    debug_print("stage2_input", start_date, end_date, stage2_input)

    return df_result
