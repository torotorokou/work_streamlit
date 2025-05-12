import pandas as pd
from utils.logger import app_logger
from logic.manage.utils.csv_loader import load_all_filtered_dataframes
from logic.manage.utils.load_template import load_master_and_template
from utils.config_loader import get_template_config
from utils.value_setter import set_value_fast_safe
from logic.manage.utils.summary_tools import summary_apply
from logic.manage.utils.dataframe_tools import (
    apply_summary_all_items,
)


def scrap_senbetsu(df_receive, master_csv):
    # --- 必要CSVの読み込み ---
    config = get_template_config()["management_sheet"]
    master_path = config["master_csv_path"]["scrap_senbetsu_map"]
    csv_ss = load_master_and_template(master_path)

    # スクラップと選別の値を取得
    csv_ss = summary_apply(csv_ss, df_receive, ["品名CD"], "正味重量", "値")
    csv_ss_sum = csv_ss.groupby("大項目").sum().reset_index()
    print(csv_ss.columns)

    # 値を代入
    master_csv = apply_summary_all_items(master_csv, csv_ss_sum)

    return master_csv
