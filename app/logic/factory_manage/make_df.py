import pandas as pd
import os
from utils.get_holydays import get_japanese_holidays
from logic.factory_manage.sql import save_df_to_sqlite_unique
from utils.config_loader import get_path_from_yaml
from utils.cleaners import enforce_dtypes, strip_whitespace
from utils.config_loader import get_expected_dtypes_by_template


def make_sql_old():
    """
    過去の複数年分のCSVファイルと最新データを読み込み、
    日付や数値整形、不要データの除外、祝日フラグ付与を行い、
    SQLiteに統合保存するメイン関数。
    """
    base_dir = get_path_from_yaml("input", section="directories")

    # --- 共通定義 ---
    dtype_new = {"伝票日付": "str", "品名": "str", "正味重量": "float64"}
    dtype_old = {"伝票日付": "str", "商品": "str", "正味重量_明細": "float64"}
    usecols_new = ["伝票日付", "正味重量", "品名"]
    usecols_old = ["伝票日付", "商品", "正味重量_明細"]
    old_files = ["2020顧客.csv", "2021顧客.csv", "2022顧客.csv", "2023_all.csv"]

    def load_and_clean_csv(
        filename: str, dtype: dict, usecols: list, is_old: bool
    ) -> pd.DataFrame:
        path = os.path.join(base_dir, filename)
        df = pd.read_csv(path, dtype=dtype, usecols=usecols, encoding="utf-8")

        if is_old:
            df.rename(
                columns={"商品": "品名", "正味重量_明細": "正味重量"}, inplace=True
            )

        # 括弧付き日付の除去と型変換
        df["伝票日付"] = (
            df["伝票日付"].astype(str).str.replace(r"\(.*\)", "", regex=True)
        )
        df["伝票日付"] = pd.to_datetime(df["伝票日付"], errors="coerce")
        df["正味重量"] = pd.to_numeric(df["正味重量"], errors="coerce")

        # NaN除去
        before = len(df)
        df = df.dropna(subset=["正味重量", "伝票日付"])
        after_nan = len(df)
        print(f"🧹 {filename}: NaN除去 {before - after_nan}件 → {after_nan}件")

        # 重量0除去
        df = df[df["正味重量"] != 0]
        print(f"🧹 {filename}: 正味重量=0 除去後 {len(df)}件")

        return df

    # --- 最新データ ---
    print("📥 最新データ読み込み")
    df_new = load_and_clean_csv(
        "20240501-20250422.csv", dtype_new, usecols_new, is_old=False
    )

    # --- 過去データ ---
    print("📥 過去データ読み込み")
    df_old_list = [
        load_and_clean_csv(fname, dtype_old, usecols_old, is_old=True)
        for fname in old_files
    ]
    df_old = pd.concat(df_old_list, ignore_index=True)

    # --- 結合 ---
    df_raw = pd.concat([df_new, df_old], ignore_index=True)
    print(f"📦 総行数（df_new + df_old）: {len(df_raw)}")

    # --- 祝日フラグ付与 ---
    start_date = df_raw["伝票日付"].min().date()
    end_date = df_raw["伝票日付"].max().date()
    holidays = get_japanese_holidays(start=start_date, end=end_date, as_str=False)
    holiday_set = set(holidays)

    df_raw["祝日フラグ"] = df_raw["伝票日付"].dt.date.apply(
        lambda x: 1 if x in holiday_set else 0
    )
    print("🎌 祝日フラグ付与完了")

    # --- SQLite保存 ---
    try:
        db_path = get_path_from_yaml("weight_data", section="sql_database")
        save_df_to_sqlite_unique(df=df_raw, db_path=db_path, table_name="ukeire")
        print("✅ SQLite保存完了")
    except Exception as e:
        print(f"❌ SQLite保存中にエラーが発生しました: {e}")

    return df_raw


def make_sql_db(df: pd.DataFrame):
    """
    与えられたデータフレームから無効データを削除し、
    整形・祝日付与をしてSQLiteに保存する。

    Args:
        df (pd.DataFrame): 元データ
    """
    print(f"🔍 元データの行数: {len(df)}")

    df["伝票日付"] = df["伝票日付"].astype(str).str.replace(r"\(.*\)", "", regex=True)
    df["伝票日付"] = pd.to_datetime(df["伝票日付"], errors="coerce")
    df["正味重量"] = pd.to_numeric(df["正味重量"], errors="coerce")

    dropped_rows = df[df[["伝票日付", "正味重量"]].isna().any(axis=1)]
    dropped_rows.to_csv("dropped_rows.csv", index=False)

    df = df.dropna(subset=["正味重量", "伝票日付"])
    print(f"🔍 dropna後の行数: {len(df)}")

    df = df[df["正味重量"] != 0]
    print(f"🔍 正味重量≠0 の行数: {len(df)}")

    start_date = df["伝票日付"].min().date()
    end_date = df["伝票日付"].max().date()
    holidays = get_japanese_holidays(start=start_date, end=end_date, as_str=False)
    holiday_set = set(holidays)
    df["祝日フラグ"] = df["伝票日付"].dt.date.apply(
        lambda x: 1 if x in holiday_set else 0
    )

    df = df.loc[:, ["伝票日付", "正味重量", "品名", "祝日フラグ"]]
    print(f"🔍 整形後の行数: {len(df)}")

    try:
        db_path = get_path_from_yaml("weight_data", section="sql_database")
        save_df_to_sqlite_unique(
            df=df,
            db_path=db_path,
            table_name="ukeire",
        )
    except Exception as e:
        print(f"❌ SQLite保存中にエラーが発生しました: {e}")


def make_csv(df: pd.DataFrame) -> pd.DataFrame:
    """
    テンプレートに基づいて受入データの空白除去・型変換を行う。

    Args:
        df (pd.DataFrame): 元データ

    Returns:
        pd.DataFrame: 整形済データ
    """
    df = strip_whitespace(df)
    expected_dtypes = get_expected_dtypes_by_template("inbound_volume")
    dtypes = expected_dtypes.get("receive", {})

    if dtypes:
        df = enforce_dtypes(df, dtypes)

    return df


# --- 実行 ---
if __name__ == "__main__":
    make_sql_old()
