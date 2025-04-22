import os

os.chdir("/work")

# %% 準備
import pandas as pd
from utils.logger import app_logger
from utils.config_loader import get_template_config
from logic.manage.utils.csv_loader import load_all_filtered_dataframes
from logic.manage.utils.load_template import load_master_and_template
from IPython.display import display
import re
from logic.manage.factory_report import process

# 表示ラベルマップ（処理対象名として使う）
csv_label_map = {"yard": "ヤード一覧", "shipping": "出荷一覧", "receive": "受入一覧"}

debug_shipping = "/work/data/output/debug_shipping.parquet"
debug_yard = "/work/data/output/debug_yard.parquet"

dfs = {
    "shipping": pd.read_parquet(debug_shipping),
    "yard": pd.read_parquet(debug_yard),
}  # テスト用CSV
# dfs
# df_shipping = dfs["shipping"]
# df_shipping
# df_yard = dfs["yard"]
# df_yard
master_csv_shobun = process(dfs)
display(master_csv_shobun)

# %% 準備
master_csv_shobun
# %%
