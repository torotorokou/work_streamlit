# %% 準備
from logic.manage.average_sheet import (
    load_config_and_headers,
    load_receive_data,
    load_master_and_template,
    aggregate_vehicle_data,
    calculate_itemwise_summary,
    calculate_itemwise_summary,
    calculate_final_totals,
    apply_rounding,
)
import pandas as pd
from utils.debug_tools import save_debug_parquets
from utils.write_excel import write_values_to_template

# 表示ラベルマップ（処理対象名として使う）
csv_label_map = {"yard": "ヤード一覧", "shipping": "出荷一覧", "receive": "受入一覧"}

debug_parquet = "/work/data/input/debug_receive.parquet"
dfs = {"receive": pd.read_parquet(debug_parquet)}  # テスト用CSV
# %%
# 絞り込みヘッダー情報の読み込み
config, key, target_columns = load_config_and_headers(csv_label_map)
target_columns

# %%
# 受入データの読み込み
df_receive = load_receive_data(dfs, key, target_columns)
df_receive.shape
# %%
# マスターとテンプレートの読み込み
master_csv = load_master_and_template(config)
master_csv

# %%
master_csv1 = aggregate_vehicle_data(df_receive, master_csv)


# %%
master_csv2 = calculate_itemwise_summary(df_receive, master_csv1)
master_csv2


# %%
master_csv3 = calculate_itemwise_summary(df_receive, master_csv2)
master_csv3
# %%
master_csv4 = calculate_final_totals(df_receive, master_csv3)
master_csv4

# %%
master_csv5 = apply_rounding(master_csv4)
master_csv5
