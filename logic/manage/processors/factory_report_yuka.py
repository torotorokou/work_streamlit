import pandas as pd
from utils.logger import app_logger
import re
from utils.config_loader import get_template_config
from logic.manage.utils.load_template import load_master_and_template
from utils.value_setter import set_value


def process_yuka(df_shipping: pd.DataFrame) -> pd.DataFrame:
    logger = app_logger()
    # マスターCSVの読込
    master_path = get_template_config()["factory_report"]["master_csv_path"]["shobun"]
    master_csv = load_master_and_template(master_path)
