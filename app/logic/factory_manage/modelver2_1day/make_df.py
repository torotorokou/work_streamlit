import pandas as pd
from utils.get_holydays import get_japanese_holidays
from utils.sql import save_df_to_sqlite
from utils.config_loader import get_path_from_yaml


def make_df_old():
    # --- 入力ディレクトリパスの取得 ---
    base_dir = get_path_from_yaml("input", section="directories")

    # --- データ読み込み ---
    df_new = pd.read_csv(f"{base_dir}/20240501-20250422.csv", encoding="utf-8")[
        ["伝票日付", "正味重量", "品名"]
    ]
    df_2020 = pd.read_csv(f"{base_dir}/2020顧客.csv")[
        ["伝票日付", "商品", "正味重量_明細"]
    ]
    df_2021 = pd.read_csv(f"{base_dir}/2021顧客.csv")[
        ["伝票日付", "商品", "正味重量_明細"]
    ]
    df_2023 = pd.read_csv(f"{base_dir}/2023_all.csv", low_memory=False)[
        ["伝票日付", "商品", "正味重量_明細"]
    ]

    # --- df_old 整形 ---
    df_old = pd.concat([df_2020, df_2021, df_2023])
    df_old.rename(columns={"商品": "品名", "正味重量_明細": "正味重量"}, inplace=True)
    df_old["伝票日付"] = (
        df_old["伝票日付"].astype(str).str.replace(r"\(.*\)", "", regex=True)
    )
    df_old["伝票日付"] = pd.to_datetime(df_old["伝票日付"], errors="coerce")
    df_old["正味重量"] = pd.to_numeric(df_old["正味重量"], errors="coerce")
    df_old = df_old.dropna(subset=["正味重量", "伝票日付"])

    # --- df_new 整形 ---
    df_new["伝票日付"] = (
        df_new["伝票日付"].astype(str).str.replace(r"\(.*\)", "", regex=True)
    )
    df_new["伝票日付"] = pd.to_datetime(df_new["伝票日付"], errors="coerce")
    df_new["正味重量"] = pd.to_numeric(df_new["正味重量"], errors="coerce")
    df_new = df_new.dropna(subset=["正味重量", "伝票日付"])

    # --- 結合 ---
    df_raw = pd.concat([df_new, df_old], ignore_index=True)

    # --- 祝日フラグ追加 ---
    start_date = df_raw["伝票日付"].min().date()
    end_date = df_raw["伝票日付"].max().date()
    holidays = get_japanese_holidays(start=start_date, end=end_date, as_str=False)
    holiday_set = set(holidays)
    df_raw["祝日フラグ"] = df_raw["伝票日付"].dt.date.apply(
        lambda x: 1 if x in holiday_set else 0
    )

    # --- SQLite保存 ---
    try:
        db_path = get_path_from_yaml("weight_data", section="sql_database")

        save_df_to_sqlite(
            df=df_raw,
            db_path=db_path,
            table_name="ukeire",
            if_exists="replace",
        )
    except Exception as e:
        print(f"❌ SQLite保存中にエラーが発生しました: {e}")


# 実行
if __name__ == "__main__":
    make_df_old()
