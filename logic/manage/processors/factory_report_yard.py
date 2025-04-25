import pandas as pd
from utils.logger import app_logger
from utils.config_loader import get_template_config
from logic.manage.utils.load_template import load_master_and_template
from logic.manage.utils.excel_tools import add_label_rows_and_restore_sum
from logic.manage.utils.summary_tools import (
    write_sum_to_target_cell,
    summarize_value_by_cell_with_label,
)
from logic.manage.utils.summary_merge import summary_apply_by_sheet
from utils.value_setter import set_value
from IPython.display import display


def process_yard(df_yard: pd.DataFrame, df_shipping: pd.DataFrame) -> pd.DataFrame:
    logger = app_logger()

    # --- ① マスターCSVの読み込み ---
    config = get_template_config()["factory_report"]
    master_path = config["master_csv_path"]["yard"]
    master_csv = load_master_and_template(master_path)

    # --- ② 有価の値集計処理（df_yard + df_shippingを使用） ---
    updated_master_csv = apply_yard_summary(master_csv, df_yard, df_shipping)

    # # --- ③ セル単位で有価名をマージし、合計を計算 ---
    updated_with_sum = summarize_value_by_cell_with_label(
        updated_master_csv, label_col="品目名"
    )

    # # --- ④ 合計行などを追加集計 ---
    target_keys = ["品目名"]
    target_values = ["合計_ヤード"]
    updated_with_sum2 = write_sum_to_target_cell(
        updated_with_sum, target_keys, target_values
    )

    # # ラベル行追加
    final_df = add_label_rows_and_restore_sum(
        updated_with_sum2, label_col="品目名", offset=-1
    )

    logger.info("✅ 出荷ヤードの帳票生成が完了しました。")
    return final_df


def apply_yard_summary(master_csv, df_yard, df_shipping):
    df_map = {"ヤード": df_yard, "出荷": df_shipping}

    sheet_key_pairs = [
        ("ヤード", ["種類名"]),
        ("ヤード", ["種類名"], ["品名"]),
        ("出荷", ["業者名", "品名"]),
    ]

    master_csv_updated = master_csv.copy()

    for sheet_name, key_cols in sheet_key_pairs:
        data_df = df_map[sheet_name]

        master_csv_updated = summary_apply_by_sheet(
            master_csv=master_csv_updated,
            data_df=data_df,
            sheet_name=sheet_name,
            key_cols=key_cols,
        )

    return master_csv_updated


def format_table(master_csv: pd.DataFrame) -> pd.DataFrame:
    """
    出荷処分のマスターCSVから必要な列を整形し、カテゴリを付与する。

    Parameters:
        master_csv : pd.DataFrame
            出荷処分の帳票CSV（"業者名", "セル", "値" を含む）

    Returns:
        pd.DataFrame : 整形後の出荷処分データ
    """
    # 必要列を抽出
    format_df = master_csv[["品目名", "セル", "値"]].copy()

    # 不要な列を除外・置換
    format_df.rename(columns={"品目名": "大項目"}, inplace=True)

    # カテゴリ列を追加
    format_df["カテゴリ"] = "ヤード"

    return format_df
