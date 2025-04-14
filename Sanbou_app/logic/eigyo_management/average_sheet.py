import pandas as pd
from utils.config_loader import load_config
from utils.logger import app_logger
from utils.date_tools import get_weekday_japanese
from utils.rounding_tools import round_value_column


# 処理の統合
def process(dfs: dict, csv_label_map: dict) -> pd.DataFrame:
    """
    集計項目平均表（average_sheet）を生成するメイン処理関数。

    ユーザーからアップロードされた受入データ（receive）を基に、
    マスターCSVを更新し、以下の集計処理を順に実行します：

    1. ABC区分ごとの台数・重量・台数単価を集計
    2. 品目ごとの売上・重量・平均単価を計算
    3. 各品目およびABC区分の合計を集計
    4. 値の丸め処理（単価のみ小数2桁、それ以外は整数）

    Parameters
    ----------
    dfs : dict
        アップロードされたCSVファイルのDataFrame辞書（キーは "receive" など）
    csv_label_map : dict
        CSVの識別名と日本語ラベルの対応マップ

    Returns
    -------
    pd.DataFrame
        出力対象となる master_csv（Excelテンプレートに埋め込む形式）
    """
    # 設定とヘッダー情報の読み込み
    config, key, target_columns = load_config_and_headers(csv_label_map)

    # 対象CSVの読み込み
    df_receive = load_receive_data(dfs, key, target_columns)

    # マスターファイルとテンプレートの読み込み
    master_csv = load_master_and_template(config)

    # 集計処理ステップ
    master_csv = aggregate_vehicle_data(df_receive, master_csv)
    master_csv = calculate_itemwise_summary(df_receive, master_csv)
    master_csv = summarize_item_and_abc_totals(master_csv)
    master_csv = calculate_final_totals(df_receive, master_csv)
    master_csv = apply_rounding(master_csv)

    return master_csv


def load_config_and_headers(label_map):
    config = load_config()
    use_headers_path = config["main_paths"]["used_header_csv_info"]
    df_header = pd.read_csv(use_headers_path)

    key = "receive"
    header_name = label_map[key]
    target_columns = df_header[header_name].dropna().tolist()

    return config, key, target_columns


def load_receive_data(dfs, key, target_columns):
    return dfs[key][target_columns]


def load_master_and_template(config):
    master_path = config["templates"]["average_sheet"]["master_csv_path"]
    master_csv = pd.read_csv(master_path, encoding="utf-8-sig")

    return master_csv


# ---------------- ヘルパー関数：指定条件の行に値をセット ----------------
def set_value(
    master_csv, category_name: str, level1_name: str, level2_name: str, value
):
    # ABC項目は必須（空欄は許さない前提とします）
    if not category_name:
        print("⚠️ ABC項目が未指定です。スキップします。")
        return

    # --- 条件構築 ---
    cond = master_csv["大項目"] == category_name

    if level1_name in [None, ""]:
        cond &= master_csv["小項目1"].isnull()
    else:
        cond &= master_csv["小項目1"] == level1_name

    if level2_name in [None, ""]:
        cond &= master_csv["小項目2"].isnull()
    else:
        cond &= master_csv["小項目2"] == level2_name

    # --- 該当行の確認 ---
    if cond.sum() == 0:
        print(
            f"⚠️ 該当行が見つかりません（大項目: {category_name}, 小項目1: {level1_name}, 小項目2: {level2_name}）"
        )
        return

    # --- 値の代入 ---
    master_csv.loc[cond, "値"] = value


# 台数・重量・台数単価をABC区分ごとに集計
def aggregate_vehicle_data(
    df_receive: pd.DataFrame, master_csv: pd.DataFrame
) -> pd.DataFrame:
    logger = app_logger()

    # --- ABC項目と集計項目CDの対応表 ---
    abc_to_cd = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6}

    for abc_label, item_cd in abc_to_cd.items():
        # --- データ抽出 ---
        filtered = df_receive[df_receive["集計項目CD"] == item_cd]

        # --- 安全な数値変換 ---
        total_weight = pd.to_numeric(filtered["正味重量"], errors="coerce").sum()
        total_car = filtered["受入番号"].nunique()
        unit_price = total_weight / total_car if total_car > 0 else 0

        # --- 結果を master_csv に反映 ---
        set_value(master_csv, abc_label, "", "重量", total_weight)
        set_value(master_csv, abc_label, "", "台数", total_car)
        set_value(master_csv, abc_label, "", "台数単価", unit_price)

        # --- ログ出力 ---
        logger.info(
            f"[{abc_label}] 台数: {total_car}, 重量: {total_weight:.2f}, 単価: {unit_price:.2f}"
        )

        if total_car == 0:
            logger.warning(f"⚠️ {abc_label}区分で台数が0件のため、単価が0になります。")

    return master_csv


def calculate_itemwise_summary(
    df_receive: pd.DataFrame, master_csv: pd.DataFrame
) -> pd.DataFrame:
    logger = app_logger()

    # --- フィルター条件 ---
    unit_name = "kg"
    voucher_type = "売上"

    # --- 対応マップ ---
    abc_to_cd = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6}
    item_to_cd = {
        "混合廃棄物A": 1,
        "混合廃棄物B": 2,
        "混合廃棄物(焼却物)": 4,
    }

    # --- 集計ループ ---
    for abc_key, abc_cd in abc_to_cd.items():
        for item_name, item_cd in item_to_cd.items():
            filtered = df_receive[
                (df_receive["伝票区分名"] == voucher_type)
                & (df_receive["単位名"] == unit_name)
                & (df_receive["集計項目CD"] == abc_cd)
                & (df_receive["品名CD"] == item_cd)
            ]

            # 数値変換 & 集計
            total_weight = pd.to_numeric(filtered["正味重量"], errors="coerce").sum()
            total_sell = pd.to_numeric(filtered["金額"], errors="coerce").sum()
            ave_sell = total_sell / total_weight if total_weight > 0 else 0

            # master_csv に書き込み
            set_value(master_csv, abc_key, "平均単価", item_name, ave_sell)
            set_value(master_csv, abc_key, "kg", item_name, total_weight)
            set_value(master_csv, abc_key, "売上", item_name, total_sell)

            # ログ出力
            logger.info(
                f"[{abc_key}] {item_name} → 売上: {total_sell:.0f}, 重量: {total_weight:.2f}, 単価: {ave_sell:.2f}"
            )

            if total_weight == 0:
                logger.warning(
                    f"⚠️ {abc_key}・{item_name} の重量が0のため単価が0になります。"
                )

    return master_csv


def summarize_item_and_abc_totals(master_csv: pd.DataFrame) -> pd.DataFrame:
    logger = app_logger()

    abc_to_cd = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6}
    item_to_cd = {
        "混合廃棄物A": 1,
        "混合廃棄物B": 2,
        "混合廃棄物(焼却物)": 4,
    }

    # --- ① 品目ごとの合計（行: 合計 / 品目列）---
    for item_name in item_to_cd.keys():
        filtered = master_csv[master_csv["小項目2"] == item_name]

        total_weight = filtered[filtered["小項目1"] == "kg"]["値"].sum()
        total_sell = filtered[filtered["小項目1"] == "売上"]["値"].sum()
        ave_sell = total_sell / total_weight if total_weight > 0 else 0

        set_value(master_csv, "合計", "平均単価", item_name, ave_sell)
        set_value(master_csv, "合計", "kg", item_name, total_weight)
        set_value(master_csv, "合計", "売上", item_name, total_sell)

    # --- ② ABC業者ごとの "3品目合計" ---
    for abc_key in abc_to_cd.keys():
        filtered = master_csv[master_csv["大項目"] == abc_key]

        total_weight = filtered[filtered["小項目1"] == "kg"]["値"].sum()
        total_sell = filtered[filtered["小項目1"] == "売上"]["値"].sum()
        ave_sell = total_sell / total_weight if total_weight > 0 else 0

        set_value(master_csv, abc_key, "平均単価", "3品目合計", ave_sell)
        set_value(master_csv, abc_key, "kg", "3品目合計", total_weight)
        set_value(master_csv, abc_key, "売上", "3品目合計", total_sell)

    # --- ③ 全体の "3品目合計" ---
    filtered = master_csv[master_csv["小項目2"] == "3品目合計"]

    total_weight = filtered[filtered["小項目1"] == "kg"]["値"].sum()
    total_sell = filtered[filtered["小項目1"] == "売上"]["値"].sum()
    ave_sell = total_sell / total_weight if total_weight > 0 else 0

    set_value(master_csv, "合計", "平均単価", "3品目合計", ave_sell)
    set_value(master_csv, "合計", "kg", "3品目合計", total_weight)
    set_value(master_csv, "合計", "売上", "3品目合計", total_sell)

    logger.info("✅ 品目ごとの合計およびABC業者別3品目合計を集計しました。")

    return master_csv


def calculate_final_totals(
    df_receive: pd.DataFrame, master_csv: pd.DataFrame
) -> pd.DataFrame:
    logger = app_logger()

    # --- 台数・重量・台数単価の全体合計 ---
    total_car = master_csv[master_csv["小項目2"] == "台数"]["値"].sum()
    total_weight = master_csv[master_csv["小項目2"] == "重量"]["値"].sum()
    unit_price = total_weight / total_car if total_car > 0 else 0

    set_value(master_csv, "合計", "", "台数", total_car)
    set_value(master_csv, "合計", "", "重量", total_weight)
    set_value(master_csv, "合計", "", "台数単価", unit_price)

    logger.info(
        f"📊 全体合計 → 台数: {total_car}, 重量: {total_weight:.2f}, 単価: {unit_price:.2f}"
    )

    # --- 総品目合計 ---
    filtered = df_receive[
        (df_receive["伝票区分名"] == "売上") & (df_receive["単位名"] == "kg")
    ]
    total_weight_all = pd.to_numeric(filtered["正味重量"], errors="coerce").sum()
    total_sell_all = pd.to_numeric(filtered["金額"], errors="coerce").sum()
    average_price_all = total_weight_all / total_sell_all if total_sell_all > 0 else 0

    set_value(master_csv, "総品目㎏", "", "", total_weight_all)
    set_value(master_csv, "総品目売上", "", "", total_sell_all)
    set_value(master_csv, "総品目平均", "", "", average_price_all)

    # --- その他品目 = 総品目 － 3品目合計 ---
    total_sell_3items = master_csv[
        (master_csv["大項目"] == "合計")
        & (master_csv["小項目1"] == "売上")
        & (master_csv["小項目2"] == "3品目合計")
    ]["値"].sum()

    total_weight_3items = master_csv[
        (master_csv["大項目"] == "合計")
        & (master_csv["小項目1"] == "kg")
        & (master_csv["小項目2"] == "3品目合計")
    ]["値"].sum()

    other_sell = total_sell_all - total_sell_3items
    other_weight = total_weight_all - total_weight_3items
    other_avg_price = other_weight / other_sell if other_sell > 0 else 0

    set_value(master_csv, "その他品目㎏", "", "", other_weight)
    set_value(master_csv, "その他品目売上", "", "", other_sell)
    set_value(master_csv, "その他品目平均", "", "", other_avg_price)

    # --- 日付・曜日の記録 ---
    today = pd.to_datetime(df_receive["伝票日付"].dropna().iloc[0])
    weekday = get_weekday_japanese(today)

    set_value(master_csv, "日付", "", "", today.strftime("%Y/%m/%d"))
    set_value(master_csv, "曜日", "", "", weekday)

    logger.info(f"🗓 日付: {today.strftime('%Y/%m/%d')}（{weekday}）")

    return master_csv


def apply_rounding(master_csv: pd.DataFrame) -> pd.DataFrame:
    """
    値列に丸め処理を適用：
    - 「単価」の場合は小数点第2位まで
    - その他は整数
    """
    return round_value_column(master_csv)
