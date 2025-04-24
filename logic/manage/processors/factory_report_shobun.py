import pandas as pd
from utils.logger import app_logger
from utils.config_loader import get_template_config
from logic.manage.utils.load_template import load_master_and_template
from utils.value_setter import set_value_fast
from logic.manage.utils.excel_tools import create_label_rows_generic, sort_by_cell_row


def process_shobun(df_shipping: pd.DataFrame) -> pd.DataFrame:
    """
    出荷データ（処分）を処理して、マスターCSVに加算・ラベル挿入を行う。
    """
    logger = app_logger()
    # マスターCSVの読込
    master_path = get_template_config()["factory_report"]["master_csv_path"]["shobun"]
    master_csv = load_master_and_template(master_path)

    columns_key =["業者CD","業者名","品名"]

    # 各処理を実行
    updated_master_csv = apply_shobun_weight(master_csv, df_shipping)
    updated_master_csv2 = add_label_rows(updated_master_csv,columns_key)
    updated_master_csv3 = sum_syobun(updated_master_csv2, df_shipping,columns_key)

    return updated_master_csv3


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
    agg_marugen = filtered_marugen.groupby(["業者CD", "品名"], as_index=False)[
        "正味重量"
    ].sum()

    # --- その他業者処理 ---
    df_others = df_shipping[df_shipping["業者CD"] != marugen_num]
    agg_others = df_others.groupby("業者CD", as_index=False)["正味重量"].sum()

    # --- 統合・整形 ---
    aggregated = pd.concat([agg_others, agg_marugen], ignore_index=True)
    aggregated.rename(columns={"業者CD": "業者CD", "品名": "品名"}, inplace=True)

    master_csv["値"] = pd.to_numeric(master_csv["値"], errors="coerce").fillna(0)

    updated_master = master_csv.merge(aggregated, on=["業者CD", "品名"], how="left")
    updated_master["正味重量"] = updated_master["正味重量"].fillna(0)
    updated_master["値"] += updated_master["正味重量"]
    updated_master.drop(columns=["正味重量"], inplace=True)

    return updated_master


def add_label_rows(master_csv: pd.DataFrame,columns_key) -> pd.DataFrame:
    """
    小項目1をラベルとして追加し、1行下のセルに配置。
    """

    # master_csvのコピーにラベル列を追加
    df_filtered = master_csv[master_csv["業者CD"] != "合計"]
    df_label = create_label_rows_generic(df_filtered,columns_key,label_source_col="業者名", offset=-1)

    df_extended = pd.concat([master_csv, df_label], ignore_index=True)
    df_extended = sort_by_cell_row(df_extended)  # ソート

    return df_extended


def sum_syobun(master_csv, df_shipping,columns_key):

    total = pd.to_numeric(master_csv["値"], errors="coerce").sum()
    set_value_fast(master_csv, columns_key, ["合計", "処分", None], total)


    return master_csv