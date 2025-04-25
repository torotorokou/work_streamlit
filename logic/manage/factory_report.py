import pandas as pd
from utils.logger import app_logger
from utils.config_loader import get_template_config
from logic.manage.utils.csv_loader import load_all_filtered_dataframes
from logic.manage.processors.factory_report_shobun import process_shobun
from logic.manage.processors.factory_report_yuuka import process_yuuka


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

    # 各処理を実行
    # 出荷処分
    master_csv_shobun = process_shobun(df_shipping)
    # logger.info(f"出荷処分;{master_csv_shobun}")
    # 出荷有価
    master_csv_yuka = process_yuuka(df_yard, df_shipping)
    # logger.info(f"出荷有価;{master_csv_yuka}")
    # 出荷ヤード
    master_csv_yard = process_yard(df_yard, df_shipping)
    # 結合
    combined_df = pd.concat([master_csv_yuka, master_csv_shobun], ignore_index=True)
    return combined_df
