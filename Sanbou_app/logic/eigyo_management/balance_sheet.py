def generate_balance_sheet(dfs, label_map):
    # 搬出入収支表用の加工処理（仮）
    for df in dfs.values():
        df["集計タイプ"] = "収支表"
    return dfs