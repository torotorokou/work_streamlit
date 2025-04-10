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
def daisuu_juuryou_daisuutanka(df_receive, master_csv, template, csv_label_map):
    logger = app_logger()

    # ---------------- フィルター条件を辞書で管理 ----------------
    filter_config = {
        "unit_name": "kg",
        "voucher_type": "売上",
        "item_name": ["混合廃棄物A", "混合廃棄物B", "混合廃棄物(焼却物)"]
    }

    # ---------------- ABC項目と集計項目CDの対応表 ----------------
    abc_to_cd = {
        "A": 1,
        "B": 2,
        "C": 3,
        "D": 4,
        "E": 5,
        "F": 6
    }

    # ---------------- ヘルパー関数：指定条件の行に値をセット ----------------
    def set_value(abc_key: str, condition_name: str, value):
        cond = (
            (master_csv["ABC項目"] == abc_key) &
            (master_csv["kg売上単価"] == condition_name)
        )
        master_csv.loc[cond, "値"] = value

    # ---------------- メイン処理ループ：A〜Fの処理 ----------------
    for abc_key, cd in abc_to_cd.items():
        filtered = df_receive[
            (df_receive["伝票区分名"] == filter_config["voucher_type"]) &
            (df_receive["単位名"] == filter_config["unit_name"]) &
            (df_receive["品名"].isin(filter_config["item_name"])) &
            (df_receive["集計項目CD"] == cd)
        ]

        # 安全な数値変換＋集計
        total_weight = pd.to_numeric(filtered["正味重量"], errors="coerce").sum()
        total_car = filtered["受入番号"].nunique()
        unit_price = total_weight / total_car if total_car else 0

        # 結果を master_csv に書き込み
        set_value(abc_key, "重量", total_weight)
        set_value(abc_key, "台数", total_car)
        set_value(abc_key, "台数単価", unit_price)

        logger.info(
            f"✅ {abc_key}区分 → 台数: {total_car}, 重量: {total_weight:.2f}, 単価: {unit_price:.2f}"
        )

    return master_csv


