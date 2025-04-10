import pandas as pd
from utils.config_loader import load_config
from utils.logger import app_logger


def load_config_and_headers(label_map):
    config = load_config()
    redim_headers_path = config["main_paths"]["redim_header_csv_info"]
    df_header = pd.read_csv(redim_headers_path)

    key = "receive"
    header_name = label_map[key]
    target_columns = df_header[header_name].dropna().tolist()

    return config, key, target_columns


def load_receive_data(dfs, key, target_columns):
    return dfs[key][target_columns]


def load_master_and_template(config):
    master_path = config["templates"]["average_sheet"]["master_csv_path"]
    master_csv = pd.read_csv(master_path, encoding="utf-8-sig")

    template_path = config["templates"]["average_sheet"]["template_excel_path"]
    template = pd.read_excel(template_path, sheet_name="テンプレート", engine="openpyxl")

    return master_csv, template


# 各処理の実行
# 台数・重量・台数単価
def daisuu_juuryou_daisuutanka(df_receive, master_csv, template,csv_label_map):
    logger = app_logger()
    tani = "kg"
    denpyou_kubun = "売上"
    item_name = ["混合廃棄物A", "混合廃棄物B", "混合廃棄物(焼却物)"]

    #フィルタリング
    filtered = df_receive[
        (df_receive["伝票区分名"]==denpyou_kubun) &
        (df_receive["単位名"]==tani) &
        (df_receive["品名"].isin(item_name))
    ]

    print(filtered.value_counts())

    #マスターインデックス
    t_index = master_csv[
        (master_csv["ABC項目"] == "A") &
        (master_csv["kg売上単価"] == "重量")
    ].index

    total_weight = filtered[filtered["集計項目CD"]==1]["正味重量"].sum()
    logger.info(f"✅ 正味重量の合計: {total_weight:.2f} kg")
    master_csv.loc[t_index, "値"]= total_weight
    return total_weight
    # t_index = master_csv[
    #     (master_csv["ABC項目"] == "A") &
    #     (master_csv["kg売上単価"] == "台数")
    # ].index

    # total_car = filtered[filtered["集計項目CD"]==1]["受入番号"].nunique()
    # master_csv.loc[t_index, "値"]= total_car

    # t_index = master_csv[
    #     (master_csv["ABC項目"] == "A") &
    #     (master_csv["kg売上単価"] == "台数単価")
    # ].index


    # master_csv.loc[t_index, "値"]= total_weight/ total_car



    return master_csv
