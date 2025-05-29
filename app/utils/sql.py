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


def save_df_to_sqlite(
    df: pd.DataFrame, db_path: str, table_name: str, if_exists: str = "replace"
) -> None:
    """
    DataFrameをSQLiteに保存する共通関数
    """
    conn = None  # 先に定義しておく

    try:
        # --- ディレクトリが存在しない場合は作成 ---
        db_dir = os.path.dirname(db_path)
        os.makedirs(db_dir, exist_ok=True)

        # --- DB接続と保存 ---
        conn = sqlite3.connect(db_path)
        df.to_sql(name=table_name, con=conn, if_exists=if_exists, index=False)
        print(db_path)
        print(f"✅ {len(df)} 件のデータをテーブル '{table_name}' に保存しました。")

    except Exception as e:
        print(f"❌ SQLite保存中にエラーが発生しました: {e}")

    finally:
        if conn:  # conn が定義済みならクローズ
            conn.close()
