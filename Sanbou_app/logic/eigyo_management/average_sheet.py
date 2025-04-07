def generate_average_sheet(dfs, label_map):
    # 平均表用処理（仮）
    for df in dfs.values():
        df["分類"] = "ABC"
    return dfs