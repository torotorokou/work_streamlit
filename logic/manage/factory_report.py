import pandas as pd
from utils.logger import app_logger
from utils.config_loader import get_template_config
from logic.manage.utils.csv_loader import load_all_filtered_dataframes
from logic.manage.processors.factory_report_shobun import process_shobun
from logic.manage.processors.factory_report_yuka import process_yuka


def process(dfs: dict) -> pd.DataFrame:
    """
    Streamlitの選択に基づき、工場日報（処分パート）を処理するエントリーポイント関数。

    """

    #  設定の取得
    logger = app_logger()
    template_key = "factory_report"
    template_config = get_template_config()[template_key]

    template_name = template_config["key"]
    csv_keys = template_config["required_files"]

    # インポートCSVの取得
    df_dict = load_all_filtered_dataframes(dfs, csv_keys, template_name)
    df_shipping = df_dict.get("shipping")
    df_yard = df_dict.get("yard")

    # 出荷処分データの処理
    master_csv_shobun = process_shobun(df_shipping)

    # master_csv_yuka = process_yuka(df_yard, df_shipping)

    return master_csv_shobun
