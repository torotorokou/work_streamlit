from sqlalchemy import create_engine, text
from datetime import datetime, date, timedelta
import pandas as pd
import os
import sqlite3
from utils.config_loader import get_path_from_yaml
from sqlite3 import OperationalError
from typing import List, Union


def save_ukeire_data(df: pd.DataFrame) -> None:
    """
    受け入れデータ（df）を SQLite に保存する関数。
    SQLite の保存パスは YAML から取得。
    テーブル名は "ukeire" に固定。

    Parameters:
        df (pd.DataFrame): 保存対象のデータフレーム
    """
    try:
        db_path = get_path_from_yaml("weight_data", section="sql_database")
        save_df_to_sqlite_unique(
            df=df,
            db_path=db_path,
            table_name="ukeire",
        )
        print("✅ データの保存が完了しました")
    except Exception as e:
        print(f"❌ SQLite保存中にエラーが発生しました: {e}")


def load_recent_dates_from_sql(db_path: str, days: int = 90):
    """
    指定したSQLiteデータベースから、直近days日間の伝票日付を取得する。

    Args:
        db_path (str): SQLiteのファイルパス
        days (int): 取得する日数の範囲（デフォルト90日）

    Returns:
        List[date]: 対象期間の伝票日付の一覧（日付型）
    """
    conn = sqlite3.connect(db_path)
    df = pd.read_sql("SELECT 伝票日付 FROM ukeire", conn)
    conn.close()

    df["伝票日付"] = pd.to_datetime(df["伝票日付"], errors="coerce")
    df = df.dropna(subset=["伝票日付"])
    recent_dates = df[df["伝票日付"] >= datetime.today() - timedelta(days=days)]
    return recent_dates["伝票日付"].dt.date.unique()


def load_data_from_sqlite() -> pd.DataFrame:
    """
    SQLiteから工場の搬入量データを読み込み、前処理を行う。
    祝日フラグ列が存在しない場合は自動的に読み飛ばす。

    Returns:
        pd.DataFrame: 加工済みのデータフレーム
    """
    df_path = get_path_from_yaml("weight_data", section="sql_database")
    engine = create_engine(f"sqlite:///{df_path}")

    # まずはカラム一覧を取得
    with engine.connect() as conn:
        col_query = "PRAGMA table_info(ukeire)"
        columns = pd.read_sql(col_query, conn)["name"].tolist()

    # 祝日フラグがあるかどうかでクエリを分岐
    if "祝日フラグ" in columns:
        query = """
            SELECT 伝票日付, 品名, 正味重量, 祝日フラグ
            FROM ukeire
            WHERE 伝票日付 IS NOT NULL AND 品名 IS NOT NULL AND 正味重量 IS NOT NULL
        """
    else:
        query = """
            SELECT 伝票日付, 品名, 正味重量
            FROM ukeire
            WHERE 伝票日付 IS NOT NULL AND 品名 IS NOT NULL AND 正味重量 IS NOT NULL
        """

    # データ読み込みとクレンジング
    df = pd.read_sql(query, engine)
    df = clean_for_compare(df)
    return df


def clean_for_compare(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["伝票日付"] = (
        df["伝票日付"].astype(str).str.replace(r"\(.*\)", "", regex=True).str.strip()
    )
    df["伝票日付"] = pd.to_datetime(df["伝票日付"], errors="coerce")
    df["正味重量"] = pd.to_numeric(df["正味重量"], errors="coerce")
    df = df.dropna(subset=["伝票日付", "正味重量"])
    return df


def get_date_range_from_sqlite(db_path: str) -> tuple[str, str]:
    """
    SQLiteデータベースから伝票日付の最小・最大値を取得する。

    Args:
        db_path (str): SQLiteのパス

    Returns:
        tuple[str, str]: 最小・最大の日付（文字列）
    """
    engine = create_engine(f"sqlite:///{db_path}")
    with engine.connect() as conn:
        result = conn.execute(
            text("SELECT MIN(伝票日付), MAX(伝票日付) FROM ukeire")
        ).fetchone()
        return result[0], result[1]


def get_training_date_range(
    db_path: str, table_name: str = "ukeire"
) -> tuple[date, date]:
    """
    任意のテーブルから伝票日付の最小・最大を取得し、datetime.date型で返す。

    Args:
        db_path (str): SQLiteのパス
        table_name (str): テーブル名（デフォルト "ukeire"）

    Returns:
        tuple[date, date]: 最小・最大の日付
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


def save_df_to_sqlite_unique(df: pd.DataFrame, db_path: str, table_name: str) -> None:
    """
    SQLiteにDataFrameを保存する（全カラムで重複チェックを行い、新規行のみを追記）。

    Args:
        df (pd.DataFrame): 保存対象データ
        db_path (str): SQLiteファイルパス
        table_name (str): 保存対象テーブル名
    """

    def convert_datetime_to_str(df: pd.DataFrame) -> pd.DataFrame:
        for col in df.select_dtypes(include=["datetime64[ns]"]).columns:
            df[col] = df[col].dt.strftime("%Y-%m-%d")
        return df

    def convert_nan_to_none(df: pd.DataFrame) -> pd.DataFrame:
        return df.where(pd.notnull(df), None)

    def load_existing_data(conn, table_name: str, expected_columns) -> pd.DataFrame:
        try:
            tables = pd.read_sql(
                "SELECT name FROM sqlite_master WHERE type='table';", conn
            )["name"].tolist()
            if table_name not in tables:
                print(
                    f"📂 テーブル '{table_name}' は存在しないため、新規作成対象として扱います。"
                )
                return pd.DataFrame(columns=expected_columns)
            return pd.read_sql(f"SELECT * FROM {table_name}", conn)
        except Exception as e:
            print(f"❌ 既存データの読み込みに失敗しました: {e}")
            raise

    conn = None
    try:
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        conn = sqlite3.connect(db_path)
        existing_df = load_existing_data(conn, table_name, df.columns)

        df = convert_datetime_to_str(df)
        existing_df = convert_datetime_to_str(existing_df)
        df = convert_nan_to_none(df)
        existing_df = convert_nan_to_none(existing_df)

        if not existing_df.empty:
            merged = df.merge(existing_df.drop_duplicates(), how="left", indicator=True)
            new_records = merged[merged["_merge"] == "left_only"].drop(
                columns=["_merge"]
            )
        else:
            new_records = df.copy()

        if new_records.empty:
            print("⚠️ 保存対象の新規データがありません（すべて既存と重複）")
            return

        new_records.to_sql(name=table_name, con=conn, if_exists="append", index=False)
        print(f"✅ 新規 {len(new_records)} 件のデータを保存しました。")

    except Exception as e:
        print(f"❌ SQLite保存中にエラーが発生しました: {e}")

    finally:
        if conn:
            conn.close()


def get_holidays_from_sql(
    start: date, end: date, as_str: bool = True
) -> List[Union[date, str]]:
    """
    SQLiteから指定期間の祝日を取得する関数

    Args:
        start (date): 開始日
        end (date): 終了日
        as_str (bool): Trueなら'YYYY-MM-DD'形式、Falseならdate型で返す

    Returns:
        List[Union[date, str]]: 指定期間内の祝日一覧
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

    df["伝票日付"] = pd.to_datetime(df["伝票日付"], errors="coerce")

    if as_str:
        return df["伝票日付"].dt.strftime("%Y-%m-%d").tolist()
    else:
        return df["伝票日付"].dt.date.tolist()
