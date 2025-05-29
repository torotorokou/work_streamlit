from sqlalchemy import create_engine, text
from datetime import datetime, date
import pandas as pd
import os
import sqlite3


# ===============================
# 📥 SQLiteから元データを読み込む
# ===============================
def load_data_from_sqlite(
    db_path: str = "/work/app/data/factory_manage/weight_data.db",
) -> pd.DataFrame:
    engine = create_engine(f"sqlite:///{db_path}")
    query = """
        SELECT 伝票日付, 品名, 正味重量
        FROM ukeire
        WHERE 伝票日付 IS NOT NULL AND 品名 IS NOT NULL AND 正味重量 IS NOT NULL
    """
    df = pd.read_sql(query, engine)
    df["伝票日付"] = pd.to_datetime(df["伝票日付"], errors="coerce")
    df["正味重量"] = pd.to_numeric(df["正味重量"], errors="coerce")
    return df.dropna()


# ===============================
# 📅 伝票日付の最小・最大を取得する
# ===============================
def get_date_range_from_sqlite(db_path: str) -> tuple[str, str]:
    engine = create_engine(f"sqlite:///{db_path}")
    with engine.connect() as conn:
        result = conn.execute(
            text("SELECT MIN(伝票日付), MAX(伝票日付) FROM ukeire")
        ).fetchone()
        return result[0], result[1]  # SQLiteはdatetime文字列で返す


def get_training_date_range(
    db_path: str, table_name: str = "ukeire"
) -> tuple[date, date]:
    """
    任意のテーブルから伝票日付の最小・最大を取得し、datetime.date型で返す
    """
    engine = create_engine(f"sqlite:///{db_path}")
    with engine.connect() as conn:
        result = conn.execute(
            text(f"SELECT MIN(伝票日付), MAX(伝票日付) FROM {table_name}")
        ).fetchone()
        if result is None:
            raise ValueError("データが存在しません")
        start_str, end_str = result

    start = datetime.strptime(start_str, "%Y-%m-%d %H:%M:%S").date()
    end = datetime.strptime(end_str, "%Y-%m-%d %H:%M:%S").date()

    return start, end


import os
import sqlite3
import pandas as pd
from sqlite3 import OperationalError


def save_df_to_sqlite_unique(df: pd.DataFrame, db_path: str, table_name: str) -> None:
    """
    SQLiteにDataFrameを保存（全カラムで重複チェックを行い、新規行のみを追記）。

    ステップ：
    1. DBにテーブルが存在するか確認し、既存データを読み込む
    2. dfと既存データを全カラムで比較し、重複行を除外
    3. 新規レコードのみをSQLiteにappend保存
    """

    def convert_datetime_to_str(df: pd.DataFrame) -> pd.DataFrame:
        for col in df.select_dtypes(include=["datetime64[ns]"]).columns:
            df[col] = df[col].dt.strftime("%Y-%m-%d")
        return df

    def convert_nan_to_none(df: pd.DataFrame) -> pd.DataFrame:
        return df.where(pd.notnull(df), None)

    def load_existing_data(conn, table_name: str, expected_columns) -> pd.DataFrame:
        try:
            # テーブルが存在するか確認
            tables = pd.read_sql(
                "SELECT name FROM sqlite_master WHERE type='table';", conn
            )["name"].tolist()

            if table_name not in tables:
                print(
                    f"📂 テーブル '{table_name}' は存在しないため、新規作成対象として扱います。"
                )
                return pd.DataFrame(columns=expected_columns)

            # テーブルから全件読み込み
            return pd.read_sql(f"SELECT * FROM {table_name}", conn)

        except Exception as e:
            print(f"❌ 既存データの読み込みに失敗しました: {e}")
            raise

    conn = None
    try:
        # --- ディレクトリ作成（なければ作る） ---
        os.makedirs(os.path.dirname(db_path), exist_ok=True)

        # --- SQLite接続 ---
        conn = sqlite3.connect(db_path)

        # --- 既存データの読み込み ---
        existing_df = load_existing_data(conn, table_name, df.columns)

        # --- datetime → 文字列, NaN → None に変換 ---
        df = convert_datetime_to_str(df)
        existing_df = convert_datetime_to_str(existing_df)
        df = convert_nan_to_none(df)
        existing_df = convert_nan_to_none(existing_df)

        # --- 重複排除：dfにしか存在しない行を抽出 ---
        if not existing_df.empty:
            merged = df.merge(existing_df.drop_duplicates(), how="left", indicator=True)
            new_records = merged[merged["_merge"] == "left_only"].drop(
                columns=["_merge"]
            )
        else:
            new_records = df.copy()

        # --- 保存対象の確認 ---
        if new_records.empty:
            print("⚠️ 保存対象の新規データがありません（すべて既存と重複）")
            return

        # --- SQLiteへ追記保存 ---
        new_records.to_sql(name=table_name, con=conn, if_exists="append", index=False)
        print(f"✅ 新規 {len(new_records)} 件のデータを保存しました。")

    except Exception as e:
        print(f"❌ SQLite保存中にエラーが発生しました: {e}")

    finally:
        if conn:
            conn.close()
