import pandas as pd

def round_value_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    「大項目」「小項目1」「小項目2」に「単価」が含まれる行は小数点第2位に丸め、
    それ以外は整数に丸める。
    値が数値でない（文字列や日付など）の行はスキップする。
    """
    # --- 単価かどうか判定 ---
    is_tanka = (
        df["大項目"].astype(str).str.contains("単価", na=False)
        | df["小項目1"].astype(str).str.contains("単価", na=False)
        | df["小項目2"].astype(str).str.contains("単価", na=False)
    )

    # --- 値を数値に変換（できないものは NaN になる） ---
    numeric_vals = pd.to_numeric(df["値"], errors="coerce")

    # --- 値が数値である行だけ処理対象とする ---
    is_numeric = ~numeric_vals.isna()

    # --- 値の初期化（元をコピーしておく） ---
    rounded = df["値"].copy()

    # --- 単価かつ数値で0以外 → 小数点第2位 ---
    mask_tanka = is_tanka & is_numeric & (numeric_vals != 0)
    rounded.loc[mask_tanka] = numeric_vals.loc[mask_tanka].round(2)

    # --- 単価以外かつ数値 → 整数に丸め ---
    mask_non_tanka = ~is_tanka & is_numeric
    rounded.loc[mask_non_tanka] = numeric_vals.loc[mask_non_tanka].round(0).astype("Int64")

    # --- 結果を反映 ---
    df["値"] = rounded

    return df
