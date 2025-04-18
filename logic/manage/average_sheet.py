import pandas as pd

# from utils.config_loader import load_config_json
from utils.logger import app_logger
from utils.date_tools import get_weekday_japanese
from utils.rounding_tools import round_value_column
from utils.value_setter import set_value
from logic.manage.utils.load_template import load_master_and_template
from utils.config_loader import get_path_config, get_template_config


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
    key = "receive"
    key, target_columns = load_config_and_headers(csv_label_map,key)

    # 対象CSVの読み込み
    df_receive = load_receive_data(dfs, key, target_columns)

    # マスターファイルとテンプレートの読み込み
    master_path = get_template_config()["average_sheet"]["master_csv_path"]
    master_csv = load_master_and_template(master_path)

    # 集計処理ステップ
    master_csv = process_average_sheet(df_receive, master_csv)

    return master_csv


def load_config_and_headers(label_map,key):
    """
    コンフィグ設定とヘッダー定義CSVから、指定データセット（例："receive"）の必要カラムリストを取得する。

    Parameters:
        label_map (dict): データ種別（例: "receive"）に対応する日本語ラベル名の辞書。
                        例: {"receive": "受入一覧"}

    Returns:
        tuple:
            - config (dict): 読み込まれた設定情報（JSONファイルベースの辞書）
            - key (str): 使用するデータのキー（例: "receive"）
            - target_columns (list): 抽出すべきカラム名のリスト（空欄は除外済）
    """

    required_columns_definition = (
        get_path_config()["csv"]["required_columns_definition"]
    )
    df_header = pd.read_csv(required_columns_definition)

    header_name = label_map[key]
    target_columns = df_header[header_name].dropna().tolist()

    return key, target_columns


def load_receive_data(dfs, key, target_columns):
    """
    指定された辞書型DataFrameから、対象キーのDataFrameを取得し、必要なカラムのみを抽出して返す。

    Parameters:
        dfs (dict): 複数のDataFrameを格納した辞書。例: {"receive": df1, "yard": df2}
        key (str): 対象となるDataFrameのキー名。例: "receive"
        target_columns (list): 抽出するカラム名のリスト。例: ["伝票日付", "品名", "正味重量"]

    Returns:
        pd.DataFrame: 指定されたカラムのみを持つDataFrame（フィルタ済み）
    """
    return dfs[key][target_columns]


def process_average_sheet(
    df_receive: pd.DataFrame, master_csv: pd.DataFrame
) -> pd.DataFrame:
    """
    平均表テンプレート用の処理群を順に実行し、マスターCSVを完成形にする。
    """
    master_csv = aggregate_vehicle_data(df_receive, master_csv)
    master_csv = calculate_item_summary(df_receive, master_csv)
    master_csv = summarize_item_and_abc_totals(master_csv)
    master_csv = calculate_final_totals(df_receive, master_csv)
    master_csv = set_report_date_info(df_receive, master_csv)
    master_csv = apply_rounding(master_csv)
    return master_csv


# 台数・重量・台数単価をABC区分ごとに集計
def aggregate_vehicle_data(
    df_receive: pd.DataFrame, master_csv: pd.DataFrame
) -> pd.DataFrame:
    """
    受入データからABC区分ごとの台数・総重量・台数単価を集計し、
    テンプレートマスターCSVに対応する値を設定する。

    Parameters:
        df_receive (pd.DataFrame): 受入データを格納したDataFrame。
                                   「集計項目CD」「正味重量」「受入番号」などの列を含む。
        master_csv (pd.DataFrame): テンプレート構造に対応したマスター表。
                                   「大項目」「小項目1」「小項目2」「値」列を含む必要がある。

    Returns:
        pd.DataFrame: 集計結果が反映されたマスターCSV（引数と同じDataFrameに上書き）

    Notes:
        - ABC区分（A〜F）に対応する「集計項目CD」を基に台数・重量を算出。
        - 台数が0の場合は単価は0として処理。
        - ログに各区分の処理結果および注意を出力する。
    """
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


def calculate_item_summary(
    df_receive: pd.DataFrame, master_csv: pd.DataFrame
) -> pd.DataFrame:
    """
    受入データをもとに、ABC区分 × 品目ごとに売上・重量・平均単価を集計し、
    テンプレートマスターCSVに反映する。

    Parameters:
        df_receive (pd.DataFrame): 受入データ。以下の列を含む必要がある：
            - "伝票区分名"
            - "単位名"
            - "集計項目CD"
            - "品名CD"
            - "正味重量"
            - "金額"
        master_csv (pd.DataFrame): テンプレートに対応したマスター表。
                                   「大項目」「小項目1」「小項目2」「値」列を含む必要がある。

    Returns:
        pd.DataFrame: 品目別の集計結果が反映されたマスターCSV（上書き）

    Notes:
        - 売上は "伝票区分名" が "売上" のみを対象とする。
        - 単位は "kg" のみを対象とする。
        - ABC区分（A〜F）と各品目に対応したコードでフィルタし、売上・重量を合計。
        - 重量が0の場合は平均単価を0とし、警告をログ出力する。
    """
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
    """
    マスターCSVに対して、品目ごと・ABC業者ごと・全体の「3品目合計」を集計し、
    平均単価・総重量・売上をテンプレートに書き込む。

    処理ステップ：
        ① 品目別（混合廃棄物A/B/焼却物）の合計を「大項目=合計」として記入
        ② ABC業者別の3品目合計を「小項目2=3品目合計」として記入
        ③ 全体の3品目合計を「大項目=合計」「小項目2=3品目合計」として記入

    Parameters:
        master_csv (pd.DataFrame): テンプレート形式のマスター表。
                                   「大項目」「小項目1」「小項目2」「値」列が含まれていることが前提。

    Returns:
        pd.DataFrame: 集計済みのマスターCSV（対象セルに上書きされたDataFrame）

    Notes:
        - 平均単価は「売上 ÷ 重量」で計算される。
        - 重量が0の場合は平均単価を0として処理。
        - ログに集計完了メッセージを出力。
    """
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
    """
    テンプレート用マスターCSVに対し、全体の台数・重量・単価・売上情報を集計し、
    総品目・その他品目の値とともに日付・曜日も書き込む。

    Parameters:
        df_receive (pd.DataFrame): 受入データ。以下の列を含む必要あり：
            - "伝票区分名"
            - "単位名"
            - "正味重量"
            - "金額"
            - "伝票日付"

        master_csv (pd.DataFrame): テンプレート形式のマスターCSV。
                                   「大項目」「小項目1」「小項目2」「値」列が含まれていること。

    Returns:
        pd.DataFrame: 最終集計値と日付・曜日情報が反映されたマスターCSV。

    集計内容:
        - 「小項目2」が "台数" / "重量" の行をもとに、全体合計台数・重量・台数単価を算出
        - "売上" × "kg" のフィルタで総品目の重量・売上・平均単価を計算
        - 総品目 － 3品目合計 = その他品目 として差分を計算
        - df_receive の先頭日付から日付と曜日を取得し、テンプレートに書き込み

    Notes:
        - 重量や売上が0の場合は割り算を回避して平均単価を0とする。
        - 曜日は日本語（例："月", "火", ...）で `get_weekday_japanese()` により算出。
        - 日付のフォーマットは "YYYY/MM/DD"。
    """

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
    average_price_all = total_sell_all / total_weight_all if total_sell_all > 0 else 0

    set_value(master_csv, "総品目㎏", "", "", total_weight_all)
    set_value(master_csv, "総品目売上", "", "", total_sell_all)
    set_value(master_csv, "総品目平均単価", "", "", average_price_all)

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
    other_avg_price = other_sell / other_weight if other_sell > 0 else 0

    set_value(master_csv, "その他品目㎏", "", "", other_weight)
    set_value(master_csv, "その他品目売上", "", "", other_sell)
    set_value(master_csv, "その他品目平均単価", "", "", other_avg_price)

    return master_csv


def set_report_date_info(
    df_receive: pd.DataFrame, master_csv: pd.DataFrame
) -> pd.DataFrame:
    """
    受入データから最初の日付を抽出し、帳票テンプレートに「月/日」と対応する曜日を記録する。

    Parameters:
        df_receive (pd.DataFrame): 「受入一覧」のCSVデータフレーム（「伝票日付」列を含む）。
        master_csv (pd.DataFrame): 帳票テンプレートのマスターCSV。

    Returns:
        pd.DataFrame: 日付と曜日を記録したあとのマスターCSV。
    """
    logger = app_logger()
    today = pd.to_datetime(df_receive["伝票日付"].dropna().iloc[0])
    weekday = get_weekday_japanese(today)

    formatted_date = today.strftime("%m/%d")
    set_value(master_csv, "日付", "", "", formatted_date)
    set_value(master_csv, "曜日", "", "", weekday)

    logger.info(f"日付: {formatted_date}（{weekday}）")

    return master_csv


def apply_rounding(master_csv: pd.DataFrame) -> pd.DataFrame:
    """
    値列に丸め処理を適用：
    - 「単価」の場合は小数点第2位まで
    - その他は整数
    """
    return round_value_column(master_csv)
