from utils.config_loader import get_required_columns_definition
from utils.logger import app_logger


def load_filtered_dataframe(dfs, key, target_columns):
    """
    指定された辞書型DataFrameから、対象キーのDataFrameを取得し、
    指定されたカラムのみを抽出して返す。
    YAML形式で型付きのdictやネストリストにも対応。

    Parameters:
        dfs (dict): 複数のDataFrameを格納した辞書。例: {"receive": df1, "yard": df2}
        key (str): 対象となるDataFrameのキー名。例: "receive"
        target_columns (list or dict): 抽出するカラム名のリスト or {カラム名: 型}

    Returns:
        pd.DataFrame: 指定されたカラムのみを持つDataFrame（フィルタ済み）
    """
    logger = app_logger()

    if key not in dfs:
        raise KeyError(f"{key} はdfsに存在しません。")

    df = dfs[key]

    # --- 型付き辞書だったらキーだけを使う
    if isinstance(target_columns, dict):
        target_columns = list(target_columns.keys())

    # --- listの中身がさらにlistなら flatten（[[...]] → [...]）
    if (
        isinstance(target_columns, list)
        and target_columns
        and isinstance(target_columns[0], list)
    ):
        target_columns = target_columns[0]

    missing_cols = [col for col in target_columns if col not in df.columns]
    if missing_cols:
        logger.error(f"{key} に必要なカラムが不足しています: {missing_cols}")
        raise ValueError(f"{key} に次のカラムが存在しません: {missing_cols}")

    return df[target_columns]


# utils/list_tools.py などに置いてもOK
def flatten_list(nested_list):
    """
    1段ネストされたリストをフラットにする。
    例: [['A', 'B'], 'C'] → ['A', 'B', 'C']
    """
    flat = []
    for item in nested_list:
        if isinstance(item, list):
            flat.extend(item)
        else:
            flat.append(item)
    return flat


def load_all_filtered_dataframes(
    dfs: dict,
    keys: list[str],
    template_name: str,
) -> dict:
    """
    指定された帳票テンプレートとCSVキーに基づき、必要なカラムのみ抽出して返す。
    """

    df_dict = {}
    column_defs = get_required_columns_definition(template_name)

    for key in keys:
        if key in dfs:
            target_columns = column_defs.get(key, [])
            # ✅ ネストされている場合に備えて flatten
            target_columns = flatten_list(target_columns)
            df_dict[key] = load_filtered_dataframe(dfs, key, target_columns)

    return df_dict
