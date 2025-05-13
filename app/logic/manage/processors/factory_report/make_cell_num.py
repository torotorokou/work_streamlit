import re

def make_cell_num(master_csv):
    start_cells = {
    "有価": "C14",
    "処分": "C22",
    "ヤード": "C34"
    }


    col_list = ["C", "F", "I", "L", "O"]

    df=make_cell_num_cal(master_csv, start_cells, col_list)

    return df


def make_cell_num_cal(
    master_csv,
    start_cells: dict,
    col_list=None,
    category_col="カテゴリ"
):
    """
    各カテゴリごとに指定した開始セルから、横展開（列数超えたら折り返し）する。
    
    Parameters:
    - start_cells: dict[str, str] 例: {"有価": "C14", "処分": "C30"}
    - col_list: 使用する列（横展開）
    - category_col: カテゴリ列名
    """


    # 対象データのフィルタ＆整列
    df_target = (
        master_csv[
            (master_csv["セルロック"] == 1) |
            ((master_csv["セルロック"] != 1) & (master_csv["値"] != 0))
        ]
        .copy()
        .sort_values("順番")
    )
    df_target["セル"] = ""

    for category, group in df_target.groupby(category_col):
        # --- スタートセル取得 ---
        if category not in start_cells:
            raise ValueError(f"{category} に対する start_cell が指定されていません。")
        
        start_cell = start_cells[category]
        match = re.match(r"([A-Z]+)(\d+)", start_cell)
        if not match:
            raise ValueError(f"{start_cell} はセル形式（例: C14）ではありません。")

        start_row = int(match[2])
        row_step = 2
        group_indices = list(group.index)

        for i, df_idx in enumerate(group_indices):
            col = col_list[i % len(col_list)]
            row = start_row + (i // len(col_list)) * row_step
            df_target.at[df_idx, "セル"] = f"{col}{row}"

    return df_target
