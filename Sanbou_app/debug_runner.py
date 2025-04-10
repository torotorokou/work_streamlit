# %% 準備
from logic.eigyo_management.average_sheet import (
    load_config_and_headers,
    load_receive_data,
    load_master_and_template,
    daisuu_juuryou_daisuutanka
)
import pandas as pd

# 表示ラベルマップ（処理対象名として使う）
csv_label_map = {
    "yard": "ヤード一覧",
    "shipping": "出荷一覧",
    "receive": "受入一覧"
}

debug_parquet = "/work/data/input/debug_receive.parquet"
dfs = {
    "receive": pd.read_parquet(debug_parquet)  # テスト用CSV
}
dfs.items()
# %%
# 絞り込みヘッダー情報の読み込み
config, key, target_columns = load_config_and_headers(csv_label_map)
target_columns

# %%
# 受入データの読み込み
df_receive = load_receive_data(dfs, key, target_columns)
df_receive.head()

# %%
# マスターとテンプレートの読み込み
master_csv, template = load_master_and_template(config)
master_csv.head()

# %%
# master_csv.dtypes
df_receive.dtypes

# %%
df_output = daisuu_juuryou_daisuutanka(df_receive,master_csv, template,csv_label_map)
df_output


# %%