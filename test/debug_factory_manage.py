import os

os.chdir("/work/app")

import pandas as pd
from utils.logger import app_logger
from app_pages.factory_manage.pages.balance_management_table.process import processor_func

def run_debug_process() -> pd.DataFrame:
    logger = app_logger()

    debug_receive = "/work/app/data/input/debug_receive.parquet"
    debug_shipping = "/work/app/data/input/debug_shipping.parquet"
    debug_yard = "/work/app/data/input/debug_yard.parquet"

    dfs = {
        "receive": pd.read_parquet(debug_receive),
        "shipping": pd.read_parquet(debug_shipping),
        "yard": pd.read_parquet(debug_yard),
    }

    return dfs


dfs=run_debug_process()
processor_func(dfs)