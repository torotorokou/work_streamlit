

def make_cell_num(master_csv, start_cell="C20", col_list=None, lock_col="セルロック"):
    import re

    if col_list is None:
        col_list = ["C", "F", "I", "L", "O"]  # デフォルト5列

    # 起点の行番号を抽出
    match = re.match(r"([A-Z]+)(\d+)", start_cell)
    if not match:
        raise ValueError("start_cell must be like 'C20'")
    
    base_row = int(match[2])
    row_step = 2

    df_target = master_csv[(master_csv[lock_col] != 1) & (master_csv["値"] != 0)].copy().reset_index(drop=True)

    df_target["セル"] = [
        f"{col_list[i % len(col_list)]}{base_row + (i // len(col_list)) * row_step}"
        for i in range(len(df_target))
    ]

    updated_csv = master_csv.copy()
    updated_csv.loc[df_target.index, "セル"] = df_target["セル"].values

    return updated_csv