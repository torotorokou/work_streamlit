from sqlalchemy import create_engine, text
from datetime import datetime, date
import pandas as pd
import os
import sqlite3
from utils.config_loader import get_path_from_yaml
from datetime import timedelta

# 工場の搬入量予測用のモデル


# --- SQLiteから直近の日付を取得する関数 ---
def load_recent_dates_from_sql(db_path: str, days: int = 90):
    conn = sqlite3.connect(db_path)
    df = pd.read_sql("SELECT 伝票日付 FROM ukeire", conn)
    conn.close()

    df["伝票日付"] = pd.to_datetime(df["伝票日付"], errors="coerce")
    df = df.dropna(subset=["伝票日付"])
    recent_dates = df[df["伝票日付"] >= datetime.today() - timedelta(days=days)]
    return recent_dates["伝票日付"].dt.date.unique()


# ===============================
# 📥 SQLiteから元データを読み込む
# ===============================
def load_data_from_sqlite() -> pd.DataFrame:
    df_path = get_path_from_yaml("weight_data", section="sql_database")
    engine = create_engine(f"sqlite:///{df_path}")
    query = """
        SELECT 伝票日付, 品名, 正味重量, 祝日フラグ
        FROM ukeire
        WHERE 伝票日付 IS NOT NULL AND 品名 IS NOT NULL AND 正味重量 IS NOT NULL
    """
    df = pd.read_sql(query, engine)
    df["伝票日付"] = pd.to_datetime(df["伝票日付"], errors="coerce")
    df["正味重量"] = pd.to_numeric(df["正味重量"], errors="coerce")
    df["祝日フラグ"] = pd.to_numeric(df["祝日フラグ"], errors="coerce")
    return df


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

    start = pd.to_datetime(start_str).date()
    end = pd.to_datetime(end_str).date()

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

    # ✅ 必ずdatetime型に変換してから .dt アクセサを使う
    df["伝票日付"] = pd.to_datetime(df["伝票日付"], errors="coerce")

    if as_str:
        return df["伝票日付"].dt.strftime("%Y-%m-%d").tolist()
    else:
        return df["伝票日付"].dt.date.tolist()
