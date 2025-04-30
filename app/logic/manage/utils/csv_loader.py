from utils.config_loader import get_csv_required_columns
from utils.logger import app_logger


def load_filtered_dataframe(dfs, key, target_columns):
    """
    指定された辞書型DataFrameから、対象キーのDataFrameを取得し、指定されたカラムのみを抽出して返す。

    Parameters:
        dfs (dict): 複数のDataFrameを格納した辞書。例: {"receive": df1, "yard": df2}
        key (str): 対象となるDataFrameのキー名。例: "receive"
        target_columns (list): 抽出するカラム名のリスト。例: ["伝票日付", "品名", "正味重量"]

    Returns:
        pd.DataFrame: 指定されたカラムのみを持つDataFrame（フィルタ済み）

    Raises:
        KeyError: 指定されたキーがdfsに存在しない場合
        ValueError: 指定カラムの一部がDataFrameに存在しない場合
    """
    logger = app_logger()
    if key not in dfs:
        raise KeyError(f"{key} はdfsに存在しません。")

    df = dfs[key]
    missing_cols = [col for col in target_columns if col not in df.columns]
    if missing_cols:
        logger.error(f"{key} に必要なカラムが不足しています: {missing_cols}")
        raise ValueError(f"{key} に次のカラムが存在しません: {missing_cols}")

    return df[target_columns]


def load_all_filtered_dataframes(
    dfs: dict,
    keys: list[str],
    template_name: str,
) -> dict:
    """
    指定された帳票テンプレートとCSVキーに基づき、必要なカラムのみ抽出して返す。

    Parameters
    ----------
    dfs : dict
        アップロード済みのCSVデータ（キー："receive"など、値：DataFrame）
    keys : list[str]
        対象とするCSVのキー一覧（例：["receive", "shipping"]）
    template_name : str
        使用するテンプレート名（帳票名）。例: "factory_report", "average_sheet"

    Returns
    -------
    dict
        フィルタ済みDataFrameの辞書（key: str → df: pd.DataFrame）
    """
    from utils.config_loader import get_csv_required_columns

    df_dict = {}
    column_defs = get_csv_required_columns(template_name)

    for key in keys:
        if key in dfs:
            target_columns = column_defs.get(key, [])
            df_dict[key] = load_filtered_dataframe(dfs, key, target_columns)

    return df_dict
