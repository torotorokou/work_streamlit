# %% 準備
from logic.manage.factory_report import(
    process,
    process_shobun
    )
import pandas as pd
from utils.debug_tools import save_debug_parquets
from utils.write_excel import write_values_to_template

# 表示ラベルマップ（処理対象名として使う）
csv_label_map = {"yard": "ヤード一覧", "shipping": "出荷一覧", "receive": "受入一覧"}

debug_shipping = "/work/data/output/debug_shipping.parquet"
debug_yard = "/work/data/output/debug_yard.parquet"

dfs = {"shipping": pd.read_parquet(debug_shipping),
       "yard": pd.read_parquet(debug_yard)}  # テスト用CSV
# dfs
df_shipping = dfs["shipping"]
df_shipping
df_yard = dfs["yard"]
df_yard
# %%
# 処分から作業
master_csv_shobun1 = process(dfs)
master_csv_shobun1
# %%
# 処理１
master_csv_shobun1 = process_shobun(master_csv_shobun1, df_shipping)
master_csv_shobun1

# %%
master_csv1 = aggregate_vehicle_data(df_receive, master_csv)


# %%
master_csv2 = calculate_item_summary(df_receive, master_csv1)
master_csv2


# %%
master_csv3 = calculate_item_summary(df_receive, master_csv2)
master_csv3
# %%
master_csv4 = calculate_final_totals(df_receive, master_csv3)
master_csv4

# %%
master_csv5 = apply_rounding(master_csv4)
master_csv5
