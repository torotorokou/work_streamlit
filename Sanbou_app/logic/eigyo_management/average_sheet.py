import pandas as pd
from utils.config_loader import load_config
from utils.logger import app_logger

def load_config_and_headers(label_map):
    config = load_config()
    redim_headers_path = config["main_paths"]["redim_header_csv_info"]
    df_header = pd.read_csv(redim_headers_path)

    key = "receive"
    header_name = label_map[key]
    target_columns = df_header[header_name].dropna().tolist()

    return config, key, target_columns


def load_receive_data(dfs, key, target_columns):
    return dfs[key][target_columns]


def load_master_and_template(config):
    master_path = config["templates"]["average_sheet"]["master_csv_path"]
    master_csv = pd.read_csv(master_path, encoding="utf-8-sig")

    template_path = config["templates"]["average_sheet"]["template_excel_path"]
    template = pd.read_excel(template_path, sheet_name="テンプレート", engine="openpyxl")

    return master_csv, template

# ---------------- ヘルパー関数：指定条件の行に値をセット ----------------
def set_value(master_csv, abc_key: str, condition_name: str, item_name: str, value):
    # ABC項目は必須（空欄は許さない前提とします）
    if not abc_key:
        print("⚠️ ABC項目が未指定です。スキップします。")
        return

    # --- 条件構築 ---
    cond = (master_csv["ABC項目"] == abc_key)

    if condition_name in [None, ""]:
        cond &= master_csv["kg売上単価"].isnull()
    else:
        cond &= master_csv["kg売上単価"] == condition_name

    if item_name in [None, ""]:
        cond &= master_csv["品名"].isnull()
    else:
        cond &= master_csv["品名"] == item_name

    # --- 該当行の確認 ---
    if cond.sum() == 0:
        print(f"⚠️ 該当行が見つかりません（ABC: {abc_key}, 単価: {condition_name}, 品名: {item_name}）")
        return

    # --- 値の代入 ---
    master_csv.loc[cond, "値"] = value


# 各処理の実行
# 台数・重量・台数単価
def daisuu_juuryou_daisuutanka(df_receive, master_csv, template, csv_label_map):
    logger = app_logger()

    # ---------------- フィルター条件を辞書で管理 ----------------
    filter_config = {
        "unit_name": "kg",
        "voucher_type": "売上",
        "item_cd": [1, 2, 4]
    }

    # ---------------- ABC項目と集計項目CDの対応表 ----------------
    abc_to_cd = {
        "A": 1,
        "B": 2,
        "C": 3,
        "D": 4,
        "E": 5,
        "F": 6
    }

    # ---------------- メイン処理ループ：A〜Fの処理 ----------------
    for abc_key, abc_cd in abc_to_cd.items():
        filtered = df_receive[
            # (df_receive["伝票区分名"] == filter_config["voucher_type"]) &
            # (df_receive["単位名"] == filter_config["unit_name"]) &
            (df_receive["集計項目CD"] == abc_cd)
        ]
        logger.info(filtered.shape)
        # 安全な数値変換＋集計
        total_weight = pd.to_numeric(filtered["正味重量"], errors="coerce").sum()
        total_car = filtered["受入番号"].nunique()
        unit_price = total_weight / total_car if total_car else 0

        # 結果を master_csv に書き込み
        set_value(master_csv,abc_key, "重量","", total_weight)
        set_value(master_csv, abc_key, "台数","", total_car)
        set_value(master_csv, abc_key, "台数単価","", unit_price)

        logger.info(
            f"✅ {abc_key}区分 → 台数: {total_car}, 重量: {total_weight:.2f}, 単価: {unit_price:.2f}"
        )

    return master_csv

def abc_indi(df_receive, master_csv, template, csv_label_map):
    logger=app_logger()
        # ---------------- フィルター条件を辞書で管理 ----------------
    filter_config = {
        "unit_name": "kg",
        "voucher_type": "売上",
    }

    # ---------------- ABC項目と集計項目CDの対応表 ----------------
    abc_to_cd = {
        "A": 1,
        "B": 2,
        "C": 3,
        "D": 4,
        "E": 5,
        "F": 6
    }

    item_to_cd ={
        "混合廃棄物A": 1,
        "混合廃棄物B": 2,
        "混合廃棄物(焼却物)": 4,

    }

    # ---------------- メイン処理ループ：A〜Fの処理 ----------------
    for abc_key, abc_cd in abc_to_cd.items():
        for item_key, item_cd in item_to_cd.items():
            filtered = df_receive[
            (df_receive["伝票区分名"] == filter_config["voucher_type"]) &
            (df_receive["単位名"] == filter_config["unit_name"]) &
            (df_receive["集計項目CD"] == abc_cd) &
            (df_receive["品名CD"] == item_cd)
        ]
            # logger.info(filtered.shape)

            # 安全な数値変換＋集計
            total_weight = pd.to_numeric(filtered["正味重量"], errors="coerce").sum()
            total_sell = pd.to_numeric(filtered["金額"], errors="coerce").sum()
            ave_sell =  total_sell / total_weight if total_sell else 0

            # 結果を master_csv に書き込み
            set_value(master_csv,abc_key, "平均単価",item_key, ave_sell)
            set_value(master_csv, abc_key, "kg",item_key, total_weight)
            set_value(master_csv, abc_key, "売上",item_key, total_sell)

            logger.info(
                f"✅ {abc_key}・{item_key} → 売上: {total_sell}, 重量: {total_weight:.2f}, 単価: {ave_sell:.2f}"
            )

    return master_csv

def abc_sum(df_receive, master_csv, template, csv_label_map):
    logger = app_logger()

    # 変数
    abc_to_cd = {
        "A": 1,
        "B": 2,
        "C": 3,
        "D": 4,
        "E": 5,
        "F": 6
    }
    
    item_to_cd ={
        "混合廃棄物A": 1,
        "混合廃棄物B": 2,
        "混合廃棄物(焼却物)": 4,

    }

    # 品名別の合計
    for item_key, _ in item_to_cd.items():
        filtered = master_csv[
            (master_csv["品名"] == item_key)
       ]

        # 混合廃棄物の各品名で合計
        total_weight = master_csv[master_csv["kg売上単価"]=="kg"].sum()
        total_sell =  master_csv[master_csv["kg売上単価"]=="売上"].sum()
        ave_sell =  total_sell / total_weight if total_sell else 0

        # 結果を master_csv に書き込み
        set_value(master_csv,"合計", "平均単価",item_key, ave_sell)
        set_value(master_csv,"合計", "kg",item_key, total_weight)
        set_value(master_csv,"合計", "売上",item_key, total_sell)
    
    # ABC業者ごとの合計
    for abc_key, abc_cd in abc_to_cd.items():
        filtered = master_csv[
            (master_csv["ABC項目"] == abc_key)
        ]

        # ABCの各品名で合計
        total_weight = master_csv[master_csv["kg売上単価"]=="kg"].sum()
        total_sell =  master_csv[master_csv["品名"]=="売上"].sum()
        ave_sell =  total_sell / total_weight if total_sell else 0

        # 結果を master_csv に書き込み
        set_value(master_csv,"合計", "平均単価",item_key, ave_sell)
        set_value(master_csv,"合計", "kg",item_key, total_weight)
        set_value(master_csv,"合計", "売上",item_key, total_sell)

