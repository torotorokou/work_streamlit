import pandas as pd
from utils.logger import app_logger
from logic.manage.utils.csv_loader import load_all_filtered_dataframes
from logic.manage.utils.load_template import load_master_and_template
from utils.config_loader import get_template_config
from utils.value_setter import set_value_fast_safe
from logic.manage.processors.management_sheet.factory_report import update_from_factory_report
from logic.manage.processors.management_sheet.balance_sheet import update_from_balance_sheet


def process(dfs: dict) -> pd.DataFrame:
    logger = app_logger()

    # --- ① マスターCSVの読み込み ---
    config = get_template_config()["management_sheet"]
    master_path = config["master_csv_path"]["management_sheet"]
    master_csv = load_master_and_template(master_path)

    # --- ② テンプレート設定と対象CSVの読み込み ---
    template_key = "management_sheet"
    template_config = get_template_config()[template_key]
    template_name = template_config["key"]
    csv_keys = template_config["required_files"]
    logger.info(f"[テンプレート設定読込] key={template_key}, files={csv_keys}")

    df_dict = load_all_filtered_dataframes(dfs, csv_keys, template_name)
    df_receive = df_dict.get("receive")
    df_shipping = df_dict.get("shipping")
    df_yard = df_dict.get("yard")

    # --- ③ 各処理の適用 ---
    logger.info("▶️ 工場日報からの読込")
    master_csv = update_from_factory_report(dfs, master_csv)

    logger.info("▶️ 搬出入データ処理開始")
    master_csv = update_from_balance_sheet(dfs, master_csv)

    # logger.info("▶️ 有価物データ処理開始")
    # master_csv.loc[master_csv["大項目"] == "有価物", "値"] = (
    #     calculate_total_valuable_material_cost(df_yard, df_shipping)
    # )

    # logger.info("▶️ 搬入台数データ処理開始")
    # master_csv.loc[master_csv["大項目"] == "搬入台数", "値"] = inbound_truck_count(
    #     df_receive
    # )

    # logger.info("▶️ 搬入量データ処理開始")
    # master_csv.loc[master_csv["大項目"] == "搬入量", "値"] = inbound_weight(df_receive)

    # logger.info("▶️ オネストkg / m3 データ処理開始")
    # honest_kg, honest_m3 = calculate_honest_sales_by_unit(df_receive)
    # master_csv.loc[master_csv["大項目"] == "オネストkg", "値"] = honest_kg
    # master_csv.loc[master_csv["大項目"] == "オネストm3", "値"] = honest_m3

    # logger.info("▶️ 有価買取データ処理開始")
    # master_csv.loc[master_csv["大項目"] == "有価買取", "値"] = (
    #     calculate_purchase_value_of_valuable_items(df_receive)
    # )

    # logger.info("▶️ 売上・仕入・損益まとめ処理開始")
    # master_csv = calculate_misc_summary_rows(master_csv, df_receive)

    return master_csv
