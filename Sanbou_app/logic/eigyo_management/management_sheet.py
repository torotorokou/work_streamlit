def generate_management_sheet(dfs, label_map):
    # 管理票用処理（仮）
    for df in dfs.values():
        df["用途"] = "管理票"
    return dfs