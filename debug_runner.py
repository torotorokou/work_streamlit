import os

os.chdir("/work")

# %% 準備
import pandas as pd
from utils.logger import app_logger
from utils.config_loader import get_template_config
from logic.manage.utils.csv_loader import load_all_filtered_dataframes
from logic.manage.utils.load_template import load_master_and_template
from IPython.display import display
import re
from logic.manage.factory_report import process

# 表示ラベルマップ（処理対象名として使う）
csv_label_map = {"yard": "ヤード一覧", "shipping": "出荷一覧", "receive": "受入一覧"}

debug_shipping = "/work/data/output/debug_shipping.parquet"
debug_yard = "/work/data/output/debug_yard.parquet"

dfs = {
    "shipping": pd.read_parquet(debug_shipping),
    "yard": pd.read_parquet(debug_yard),
}  # テスト用CSV
# dfs
# df_shipping = dfs["shipping"]
# df_shipping
# df_yard = dfs["yard"]
# df_yard
master_csv_shobun = process(dfs)
display(master_csv_shobun)

import pandas as pd
from utils.logger import app_logger
from utils.config_loader import get_template_config
from logic.manage.utils.load_template import load_master_and_template
from utils.value_setter import set_value_fast
from logic.manage.utils.excel_tools import create_label_rows_generic, sort_by_cell_row
from logic.manage.utils.summary_tools import write_sum_to_target_cell
from logic.manage.utils.excel_tools import add_label_rows_and_restore_sum


def process_shobun(df_shipping: pd.DataFrame) -> pd.DataFrame:
    """
    出荷データ（処分）を処理して、マスターCSVに加算・ラベル挿入・整形を行う。

    Parameters:
        df_shipping : pd.DataFrame
            出荷データ（処分）CSV

    Returns:
        pd.DataFrame
            整形済みの出荷処分帳票
    """
    logger = app_logger()

    # --- ① マスターCSVの読み込み ---
    config = get_template_config()["factory_report"]
    master_path = config["master_csv_path"]["shobun"]
    master_csv = load_master_and_template(master_path)

    # --- ② 処分重量を加算（業者別）---
    updated_master_csv = apply_shobun_weight(master_csv, df_shipping)
    logger.info(f"updated_mater_csv:{updated_master_csv}")

    # --- ④ 合計行などを追加集計（業者CD） ---
    target_keys = ["業者名"]
    target_values = ["合計_処分"]
    updated_master_csv2 = write_sum_to_target_cell(
        updated_master_csv, target_keys, target_values
    )
    logger.info(f"updated_mater_csv2:{updated_master_csv2}")

    # ラベル行追加
    updated_master_csv3 = add_label_rows_and_restore_sum(
        updated_master_csv2, label_col="業者名", offset=-1
    )
    logger.info(f"updated_mater_csv3:{updated_master_csv3}")

    # --- ⑤ 表全体を整形（列順・カテゴリ追加など） ---
    final_df = format_shobun_table(updated_master_csv3)

    logger.info("✅ 出荷処分の帳票生成が完了しました。")

    return final_df


def apply_shobun_weight(
    master_csv: pd.DataFrame, df_shipping: pd.DataFrame
) -> pd.DataFrame:
    logger = app_logger()

    # --- 初期処理 ---
    df_shipping = df_shipping.copy()
    df_shipping["業者CD"] = df_shipping["業者CD"].astype(str)
    marugen_num = "8327"

    # --- 丸源処理 ---
    df_marugen = df_shipping[df_shipping["業者CD"] == marugen_num].copy()
    filtered_marugen = df_marugen[df_marugen["品名"].isin(master_csv["品名"])]
    agg_marugen = filtered_marugen.groupby(["業者CD", "品名"], as_index=False)["正味重量"].sum()

    # --- その他業者処理 ---
    df_others = df_shipping[df_shipping["業者CD"] != marugen_num]
    agg_others = df_others.groupby("業者CD", as_index=False)["正味重量"].sum()

    # --- 統合・整形 ---
    aggregated = pd.concat([agg_others, agg_marugen], ignore_index=True)
    aggregated.rename(columns={"業者CD": "業者CD", "品名": "品名"}, inplace=True)

    master_csv["値"] = pd.to_numeric(master_csv["値"], errors="coerce").fillna(0)

    # master_csv, aggregated の両方で文字列型に揃える
    master_csv["業者CD"] = master_csv["業者CD"].astype(str)
    aggregated["業者CD"] = aggregated["業者CD"].astype(str)

    updated_master = master_csv.merge(aggregated, on=["業者CD", "品名"], how="left")
    updated_master["正味重量"] = updated_master["正味重量"].fillna(0)
    updated_master["値"] += updated_master["正味重量"]
    updated_master.drop(columns=["正味重量"], inplace=True)

    return updated_master


def format_shobun_table(master_csv: pd.DataFrame) -> pd.DataFrame:
    """
    出荷処分のマスターCSVから必要な列を整形し、カテゴリを付与する。

    Parameters:
        master_csv : pd.DataFrame
            出荷処分の帳票CSV（"業者名", "セル", "値" を含む）

    Returns:
        pd.DataFrame : 整形後の出荷処分データ
    """
    # 必要列を抽出
    shobun_df = master_csv[["業者名", "セル", "値"]].copy()

    # 不要な列を除外・置換
    shobun_df.rename(columns={"業者名": "大項目"}, inplace=True)

    # カテゴリ列を追加
    shobun_df["カテゴリ"] = "処分"

    return shobun_df


# %%
