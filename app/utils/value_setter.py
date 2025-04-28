import pandas as pd


def set_value(
    master_csv, category_name: str, level1_name: str, level2_name: str, value
):
    """
    指定された「大項目・小項目1・小項目2」の組み合わせに一致する行をマスターCSVから探し、
    該当する「値」列に指定の値を代入する。

    Parameters:
        master_csv (pd.DataFrame): テンプレートに対応したマスター表（「大項目」「小項目1」「小項目2」「値」列を含む）
        category_name (str): 大項目（例: "A", "B", "合計"など）。必須。
        level1_name (str): 小項目1（例: "売上", "kg", Noneなど）
        level2_name (str): 小項目2（例: "混合廃棄物A", Noneなど）
        value: 代入する値（数値や文字列）

    Notes:
        - `category_name` が空の場合は処理をスキップし、警告を出力します。
        - 条件に一致する行が1件も存在しない場合も警告を表示します。
        - 条件に一致するすべての行に対して「値」を上書きします。
    """
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


def set_value_fast(df, match_columns, match_values, value, value_col="値"):
    if len(match_columns) != len(match_values):
        raise ValueError("keysとvaluesの長さが一致していません")

    cond = pd.Series(True, index=df.index)
    for col, val in zip(match_columns, match_values):
        if val in [None, ""]:
            cond &= df[col].isna()
        else:
            cond &= df[col] == val

    if cond.sum() == 0:
        print(f"⚠️ 該当行なし: {dict(zip(match_columns, match_values))}")
        return

    df.loc[cond, value_col] = value
