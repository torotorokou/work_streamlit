import pandas as pd
from utils.logger import app_logger
import re
from utils.config_loader import get_template_config
from logic.manage.utils.load_template import load_master_and_template
from utils.value_setter import set_value


def process_shobun(df_shipping: pd.DataFrame) -> pd.DataFrame:
    """
    出荷データ（処分）を処理して、マスターCSVに加算・ラベル挿入を行う。
    """
    logger = app_logger()
    # マスターCSVの読込
    master_path = get_template_config()["factory_report"]["master_csv_path"]["shobun"]
    master_csv = load_master_and_template(master_path)

    # 各処理を実行
    updated_master_csv = apply_shobun_weight(master_csv, df_shipping)
    updated_master_csv2 = add_label_rows(updated_master_csv)
    uodated_master_csv3 = sum_syukka_syobun(updated_master_csv2, df_shipping)


    return uodated_master_csv3


def apply_shobun_weight(
    master_csv: pd.DataFrame, df_shipping: pd.DataFrame
) -> pd.DataFrame:
    logger = app_logger()

    # --- 初期処理 ---
    df_shipping["業者CD"] = df_shipping["業者CD"].astype(str)
    marugen_num = "8327"

    # --- 丸源処理 ---
    df_marugen = df_shipping[df_shipping["業者CD"] == marugen_num].copy()
    filtered_marugen = df_marugen[df_marugen["品名"].isin(master_csv["小項目2"])]
    agg_marugen = filtered_marugen.groupby(["業者CD", "品名"], as_index=False)[
        "正味重量"
    ].sum()

    # --- その他業者処理 ---
    df_others = df_shipping[df_shipping["業者CD"] != marugen_num]
    agg_others = df_others.groupby("業者CD", as_index=False)["正味重量"].sum()

    # --- 統合・整形 ---
    aggregated = pd.concat([agg_others, agg_marugen], ignore_index=True)
    aggregated.rename(columns={"業者CD": "大項目", "品名": "小項目2"}, inplace=True)

    master_csv["値"] = pd.to_numeric(master_csv["値"], errors="coerce").fillna(0)

    updated_master = master_csv.merge(aggregated, on=["大項目", "小項目2"], how="left")
    updated_master["正味重量"] = updated_master["正味重量"].fillna(0)
    updated_master["値"] += updated_master["正味重量"]
    updated_master.drop(columns=["正味重量"], inplace=True)

    return updated_master


def add_label_rows(master_csv: pd.DataFrame) -> pd.DataFrame:
    """
    小項目1をラベルとして追加し、1行下のセルに配置。
    """

    def shift_cell_row(cell: str, offset: int = 1) -> str:
        match = re.match(r"([A-Z]+)(\d+)", cell)
        if match:
            col, row = match.groups()
            return f"{col}{int(row) + offset}"
        return cell

    df_label = master_csv.copy()
    df_label["セル"] = df_label["セル"].apply(lambda x: shift_cell_row(x, 1))
    df_label["値"] = master_csv["小項目1"]
    df_label[["小項目1", "小項目2", "小項目3"]] = ""
    df_label["大項目"] = None

    df_extended = pd.concat([master_csv, df_label], ignore_index=True)
    df_extended["セル番号"] = df_extended["セル"].apply(
        lambda x: int(re.findall(r"\d+", x)[0])
    )
    df_extended = (
        df_extended.sort_values("セル番号")
        .drop(columns="セル番号")
        .reset_index(drop=True)
    )

    return df_extended

def sum_syukka_syobun(master_csv, df_shipping):

    total = pd.to_numeric(master_csv["値"], errors="coerce").sum()
    set_value(master_csv, "合計", "処分", "", total)

    return master_csv