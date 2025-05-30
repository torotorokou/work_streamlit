import pandas as pd
from pandas.testing import assert_frame_equal
from app_pages.factory_manage.pages.inbound_volume_forecast.controller import (
    read_csv_controller,
)
from logic.factory_manage.make_df import make_sql_old
from logic.factory_manage.sql import load_data_from_sqlite

from pandas.testing import assert_frame_equal
import pandas as pd


def debug_compare_df(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    key: str = None,
    name1="df1",
    name2="df2",
    export_csv: bool = False,
    csv_path: str = "差分デバッグ出力.csv",
):
    print("🔍 [Debug] DataFrame 比較開始")
    print(f"📐 {name1}.shape = {df1.shape}, {name2}.shape = {df2.shape}\n")

    # --- カラム違いチェック ---
    cols1 = set(df1.columns)
    cols2 = set(df2.columns)
    if cols1 != cols2:
        print("⚠️ カラム構成が異なります")
        print("  df1.columns - df2.columns:", cols1 - cols2)
        print("  df2.columns - df1.columns:", cols2 - cols1)
        common_cols = sorted(list(cols1 & cols2))
    else:
        common_cols = sorted(df1.columns)

    # --- ソート用カラム選定 ---
    sort_cols = [key] if key else common_cols
    df1_sorted = df1[common_cols].sort_values(by=sort_cols).reset_index(drop=True)
    df2_sorted = df2[common_cols].sort_values(by=sort_cols).reset_index(drop=True)

    # --- 完全一致チェック ---
    if df1_sorted.equals(df2_sorted):
        print("✅ 完全一致（内容・順序・型）")
    else:
        print("❌ データフレームに差異があります。\n")

        # --- 差分のあるセルを表示 ---
        try:
            diff = df1_sorted.compare(df2_sorted, keep_shape=True, keep_equal=False)
            print(f"🧾 差異セル数: {diff.count().sum()}")
            print(diff.head(10))
        except Exception as e:
            print("⚠️ 差分比較に失敗しました:", e)

        # --- assert_frame_equal による厳密チェック ---
        try:
            assert_frame_equal(df1_sorted, df2_sorted, check_dtype=False)
        except AssertionError as e:
            print("🔬 assert_frame_equal 差分:")
            print(e)

        # --- 行単位での差分表示 ---
        df_diff = pd.concat([df1_sorted, df2_sorted]).drop_duplicates(keep=False)
        print(f"\n📛 行単位の差分: {len(df_diff)} 行")
        print(df_diff.head(10))

        if export_csv:
            df_diff.to_csv(csv_path, index=False)
            print(f"📁 差分データをCSV出力しました → {csv_path}")

    print("\n✅ デバッグ比較終了")


# --- データ取得 ---
df1 = read_csv_controller()  # SQLiteから読み込まれたデータ
make_sql_old()  # 保存前の最新データ
df2 = load_data_from_sqlite()
df2 = df2.drop(columns=["祝日フラグ"], errors="ignore")  # 念のため errors="ignore"


# --- 比較実行（差分CSVも出力） ---
debug_compare_df(df1, df2, key="伝票日付", export_csv=True)
