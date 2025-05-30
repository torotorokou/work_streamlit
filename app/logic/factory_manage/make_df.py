import pandas as pd
from utils.get_holydays import get_japanese_holidays
from logic.factory_manage.sql import save_df_to_sqlite_unique
from utils.config_loader import get_path_from_yaml
from utils.cleaners import enforce_dtypes, strip_whitespace
from utils.config_loader import get_expected_dtypes_by_template


def make_df_old():
    # --- 入力ディレクトリパスの取得 ---
    base_dir = get_path_from_yaml("input", section="directories")

    # --- 共通定義 ---
    dtype_new = {"伝票日付": "str", "品名": "str", "正味重量": "float64"}
    dtype_old = {"伝票日付": "str", "商品": "str", "正味重量_明細": "float64"}
    usecols_new = ["伝票日付", "正味重量", "品名"]
    usecols_old = ["伝票日付", "商品", "正味重量_明細"]

    # --- CSV読み込み用関数 ---
    def load_csv(filename: str, dtype: dict, usecols: list) -> pd.DataFrame:
        return pd.read_csv(
            f"{base_dir}/{filename}", dtype=dtype, usecols=usecols, encoding="utf-8"
        )

    # --- 新データ ---
    df_new = load_csv("20240501-20250422.csv", dtype=dtype_new, usecols=usecols_new)

    # --- 旧データ（複数年） ---
    old_files = ["2020顧客.csv", "2021顧客.csv", "2022顧客.csv", "2023_all.csv"]
    df_old_list = [
        load_csv(fname, dtype=dtype_old, usecols=usecols_old) for fname in old_files
    ]
    df_old = pd.concat(df_old_list, ignore_index=True)
    df_old.rename(columns={"商品": "品名", "正味重量_明細": "正味重量"}, inplace=True)

    # --- df_old 整形 ---
    print(f"📄 df_old 原始行数: {len(df_old)}")
    df_old["伝票日付"] = (
        df_old["伝票日付"].astype(str).str.replace(r"\(.*\)", "", regex=True)
    )
    df_old["伝票日付"] = pd.to_datetime(df_old["伝票日付"], errors="coerce")
    df_old["正味重量"] = pd.to_numeric(df_old["正味重量"], errors="coerce")
    old_nan_dropped = df_old[df_old[["正味重量", "伝票日付"]].isna().any(axis=1)]
    print(f"🗑 df_old NaNで削除: {len(old_nan_dropped)} 行")
    df_old = df_old.dropna(subset=["正味重量", "伝票日付"])
    old_zero_dropped = df_old[df_old["正味重量"] == 0]
    print(f"🗑 df_old 正味重量=0で削除: {len(old_zero_dropped)} 行")
    df_old = df_old[df_old["正味重量"] != 0]

    # --- df_new 整形 ---
    print(f"📄 df_new 原始行数: {len(df_new)}")
    df_new["伝票日付"] = (
        df_new["伝票日付"].astype(str).str.replace(r"\(.*\)", "", regex=True)
    )
    df_new["伝票日付"] = pd.to_datetime(df_new["伝票日付"], errors="coerce")
    df_new["正味重量"] = pd.to_numeric(df_new["正味重量"], errors="coerce")
    new_nan_dropped = df_new[df_new[["正味重量", "伝票日付"]].isna().any(axis=1)]
    print(f"🗑 df_new NaNで削除: {len(new_nan_dropped)} 行")
    df_new = df_new.dropna(subset=["正味重量", "伝票日付"])
    new_zero_dropped = df_new[df_new["正味重量"] == 0]
    print(f"🗑 df_new 正味重量=0で削除: {len(new_zero_dropped)} 行")
    df_new = df_new[df_new["正味重量"] != 0]

    # --- 結合 ---
    df_raw = pd.concat([df_new, df_old], ignore_index=True)
    print(f"📦 結合後の総行数: {len(df_raw)}")

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
        save_df_to_sqlite_unique(df=df_raw, db_path=db_path, table_name="ukeire")
        print("✅ SQLite保存完了")
    except Exception as e:
        print(f"❌ SQLite保存中にエラーが発生しました: {e}")


def make_sql_db(df: pd.DataFrame):
    print(f"🔍 元データの行数: {len(df)}")

    # --- 日付列の整形（曜日などを除去） ---
    df["伝票日付"] = df["伝票日付"].astype(str).str.replace(r"\(.*\)", "", regex=True)
    df["伝票日付"] = pd.to_datetime(df["伝票日付"], errors="coerce")

    # --- 数値変換（正味重量） ---
    df["正味重量"] = pd.to_numeric(df["正味重量"], errors="coerce")

    # --- 欠損行の保存 ---
    dropped_rows = df[df[["伝票日付", "正味重量"]].isna().any(axis=1)]
    dropped_rows.to_csv("dropped_rows.csv", index=False)

    # --- 欠損除去 ---
    df = df.dropna(subset=["正味重量", "伝票日付"])
    print(f"🔍 dropna後の行数: {len(df)}")

    # --- 正味重量が0の行を削除 ---
    df = df[df["正味重量"] != 0]
    print(f"🔍 正味重量≠0 の行数: {len(df)}")

    # --- 祝日フラグ追加 ---
    start_date = df["伝票日付"].min().date()
    end_date = df["伝票日付"].max().date()
    holidays = get_japanese_holidays(start=start_date, end=end_date, as_str=False)
    holiday_set = set(holidays)
    df["祝日フラグ"] = df["伝票日付"].dt.date.apply(
        lambda x: 1 if x in holiday_set else 0
    )
    print(f"🔍 祝日フラグ追加後の行数: {len(df)}")

    # --- 必要列に限定 ---
    df = df.loc[:, ["伝票日付", "正味重量", "品名", "祝日フラグ"]]
    print(f"🔍 整形後の行数: {len(df)}")

    # --- SQLite保存 ---
    try:
        db_path = get_path_from_yaml("weight_data", section="sql_database")
        save_df_to_sqlite_unique(
            df=df,
            db_path=db_path,
            table_name="ukeire",
        )
    except Exception as e:
        print(f"❌ SQLite保存中にエラーが発生しました: {e}")


def make_csv(df):
    # 空白除去
    df = strip_whitespace(df)

    # テンプレートに基づく型定義を取得
    expected_dtypes = get_expected_dtypes_by_template("inbound_volume")
    dtypes = expected_dtypes.get("receive", {})

    # データ型を適用
    if dtypes:
        df = enforce_dtypes(df, dtypes)

    return df


# 実行
if __name__ == "__main__":
    make_df_old()
