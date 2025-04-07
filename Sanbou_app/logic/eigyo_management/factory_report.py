def generate_factory_report(dfs, label_map):
    # 工場日報用の加工処理（仮）
    for df in dfs.values():
        df["レポート種別"] = "工場日報"
    return dfs