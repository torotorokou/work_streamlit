from utils.logger import app_logger
import pandas as pd
from utils.logger import app_logger
from utils.config_loader import get_template_config
from logic.manage.utils.csv_loader import load_all_filtered_dataframes
from logic.manage.utils.load_template import load_master_and_template


def process(dfs):
    logger = app_logger()

    # --- テンプレート設定の取得 ---
    template_key = "block_unit_price"
    template_config = get_template_config()[template_key]
    template_name = template_config["key"]
    csv_keys = template_config["required_files"]
    logger.info(f"[テンプレート設定読込] key={template_key}, files={csv_keys}")

    # --- コンフィグとマスター読み込み ---
    config = get_template_config()["block_unit_price"]
    master_path = config["master_csv_path"]["vendor_code"]
    master_csv = load_master_and_template(master_path)

    # --- CSVの読み込み ---
    df_dict = load_all_filtered_dataframes(dfs, csv_keys, template_name)
    df_shipping = df_dict.get("shipping")

    # --- 個別処理 ---
    logger.info("▶️ フィルタリング")
    df_after = filter_shipping_by_vendor_rules(master_csv, df_shipping)

    logger.info("▶️ 単価1円追加")
    df_after = apply_unit_price_addition(master_csv, df_after)

    # 固定運搬費の算出
    logger.info("▶️ 運搬費（固定）")
    df_after = process1(master_csv, df_after)

    return master_csv


def filter_shipping_by_vendor_rules(master_csv, df_shipping):
    # --- 業者CDでフィルタ ---
    df_after = df_shipping[df_shipping["業者CD"].isin(master_csv["業者CD"])].copy()

    # --- 品名指定があるものをマージしてフィルタリング ---
    item_filter_df = master_csv[master_csv["品名"].notna()][
        ["業者CD", "品名"]
    ].drop_duplicates()

    # 丸源処理。品名でソートする
    if not item_filter_df.empty:
        # 「業者CDと品名のペア」が一致する行だけ残す（外積フィルタ）
        df_after = df_after.merge(
            item_filter_df, on=["業者CD", "品名"], how="left", indicator=True
        )
        df_after = df_after[
            (df_after["_merge"] == "both")
            | (~df_after["業者CD"].isin(item_filter_df["業者CD"]))
        ]
        df_after = df_after.drop(columns=["_merge"])

    # 正味重量が0を除外
    df_after = df_after[df_after["正味重量"].fillna(0) != 0]


    # 運搬費をmaster_csvから追加
    # 業者CDごとに1件に絞ってからマージ
    unique_master = master_csv.drop_duplicates(subset=["業者CD"])[["業者CD", "運搬社数"]]
    df_after = df_after.merge(unique_master, on="業者CD", how="left")

    # 業者CDで並び替え
    df_after = df_after.sort_values(by="業者CD").reset_index(drop=True)



    return df_after


def apply_unit_price_addition(master_csv, df_shipping: pd.DataFrame) -> pd.DataFrame:
    """
    出荷データ（df）に対して、1円追加情報を業者CD単位でマスターと照合し、
    対象業者の単価に加算を行う処理。
    """
    from logic.manage.utils.summary_tools import apply_column_addition_by_keys

    # --- 単価への1円追加処理（業者CDで結合） ---
    df_after = apply_column_addition_by_keys(
        base_df=df_shipping,
        addition_df=master_csv,
        join_keys=["業者CD"],
        value_col_to_add="1円追加",
        update_target_col="単価",
    )

    return df_after


def process1(master_csv, df_shipping):
    from config.loader.main_path import MainPath
    from logic.readers.read_transport_discount import ReadTransportDiscount

    mainpath = MainPath()  # YAMLからtransport_costsを含むパス群を取得
    reader = ReadTransportDiscount(mainpath)

    df_transport = reader.load_discounted_df()
    print(df_transport.head())


    return df_after
