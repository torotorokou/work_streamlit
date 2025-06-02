import pandas as pd
from pandas.testing import assert_frame_equal
from logic.factory_manage.make_df import (
    make_sql_old,
    read_csv_hannnyuu,
    read_csv_hannnyuu_old,
)
from logic.factory_manage.sql import load_data_from_sqlite


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

    # --- 不要カラム除去 ---
    for col in ["祝日フラグ"]:
        df1 = df1.drop(columns=col, errors="ignore")
        df2 = df2.drop(columns=col, errors="ignore")

    # --- データ型比較 ---
    print("📊 カラム型の比較:")
    mismatch_dtypes = {}
    for col in set(df1.columns) & set(df2.columns):
        if col in df1.columns and col in df2.columns:
            t1, t2 = df1[col].dtype, df2[col].dtype
            if t1 != t2:
                mismatch_dtypes[col] = (t1, t2)
    if mismatch_dtypes:
        print("⚠️ 型が異なるカラム:")
        for k, (t1, t2) in mismatch_dtypes.items():
            print(f" - {k}: {t1} ≠ {t2}")
    else:
        print("✅ 全カラムのdtype一致")

    # --- 日付整形 ---
    if "伝票日付" in df1.columns:
        df1["伝票日付"] = pd.to_datetime(df1["伝票日付"], errors="coerce")
    if "伝票日付" in df2.columns:
        df2["伝票日付"] = pd.to_datetime(df2["伝票日付"], errors="coerce")

    # --- カラム構成確認 ---
    cols1 = set(df1.columns)
    cols2 = set(df2.columns)
    if cols1 != cols2:
        print("⚠️ カラム構成が異なります")
        print("  df1 - df2:", cols1 - cols2)
        print("  df2 - df1:", cols2 - cols1)
        common_cols = sorted(list(cols1 & cols2))
    else:
        common_cols = sorted(df1.columns)

    # --- ソートキー決定 ---
    if key and key not in common_cols:
        print(
            f"⚠️ 指定キー '{key}' は存在しません。代わりに {common_cols} を使用します。"
        )
        sort_cols = common_cols
    else:
        sort_cols = [key] if key else common_cols

    # --- ソート・整形 ---
    df1_sorted = (
        df1[common_cols]
        .dropna(subset=sort_cols)
        .sort_values(by=sort_cols)
        .reset_index(drop=True)
    )
    df2_sorted = (
        df2[common_cols]
        .dropna(subset=sort_cols)
        .sort_values(by=sort_cols)
        .reset_index(drop=True)
    )

    # --- 完全一致確認 ---
    if df1_sorted.equals(df2_sorted):
        print("✅ 完全一致（内容・順序・型）")
    else:
        print("❌ 差異があります。\n")

        try:
            diff = df1_sorted.fillna("NaN比較用").compare(
                df2_sorted.fillna("NaN比較用"), keep_shape=True, keep_equal=False
            )
            print(f"🧾 差異セル数: {diff.count().sum()}")
            print(diff.head(10))
        except Exception as e:
            print("⚠️ セル比較に失敗:", e)

        try:
            assert_frame_equal(df1_sorted, df2_sorted, check_dtype=False)
        except AssertionError as e:
            print("🔬 assert_frame_equal 差分:")
            print(e)

        # --- 正確な差分抽出 ---
        df_merge_diff = pd.merge(df1_sorted, df2_sorted, how="outer", indicator=True)
        df_only_df1 = df_merge_diff.query('_merge == "left_only"').drop(
            columns=["_merge"]
        )
        df_only_df2 = df_merge_diff.query('_merge == "right_only"').drop(
            columns=["_merge"]
        )
        df_diff = pd.concat([df_only_df1, df_only_df2], ignore_index=True)

        print(f"📛 df1にしかない行: {len(df_only_df1)}")
        print(f"📛 df2にしかない行: {len(df_only_df2)}")
        print(f"📊 合計差分行数: {len(df_diff)}")
        print(df_diff.head(10))

        if export_csv:
            df_diff.to_csv(csv_path, index=False)
            print(f"📁 差分CSV出力済 → {csv_path}")

    print("\n✅ デバッグ比較終了")


# --- 実行部分（本番用） ---
data = {
    "new": read_csv_hannnyuu(),
    "old": read_csv_hannnyuu_old(),
}
df1 = data["old"]

make_sql_old()  # 最新生成（保存含む）
df2 = load_data_from_sqlite()
df2 = df2.drop(columns=["祝日フラグ"])

debug_compare_df(df1, df2, key="伝票日付", export_csv=True)
