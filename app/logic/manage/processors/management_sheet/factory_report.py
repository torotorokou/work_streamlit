from utils.value_setter import set_value_fast_safe
from logic.manage.factory_report import process as process_fact


# 工場日報からの読込
def factory_report(dfs, master_csv):
    csv_fac = process_fact(dfs)

    # 有価物
    item_word = "有価物"
    yuka_word = master_csv.loc[master_csv["大項目"] == item_word, "検索ワード1"].values[
        0
    ]
    total_yuka = csv_fac.loc[csv_fac["大項目"] == yuka_word, "値"].values[0]

    master_csv = set_value_fast_safe(master_csv, ["大項目"], ["有価物"], total_yuka)

    # シュレッダー
    item_word = "シュレッダー"
    shredder_word = master_csv.loc[
        master_csv["大項目"] == item_word, "検索ワード1"
    ].values[0]
    total_shredder = csv_fac.loc[csv_fac["大項目"] == shredder_word, "値"].values[0]

    master_csv = set_value_fast_safe(
        master_csv, ["大項目"], [item_word], total_shredder
    )

    return master_csv
