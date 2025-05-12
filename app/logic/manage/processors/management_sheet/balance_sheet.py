from utils.value_setter import set_value_fast_safe
from logic.manage.balance_sheet import process as process_bal


# 工場日報からの読込
def balance_sheet(dfs, master_csv):
    csv_bal = process_bal(dfs)

    # 終了台数
    item_word = ""
    num = master_csv.loc[master_csv["大項目"] == item_word, "検索ワード1"].values[
        0
    ]
    total_yuka = csv_bal.loc[csv_bal["大項目"] == yuka_word, "値"].values[0]

    master_csv = set_value_fast_safe(master_csv, ["大項目"], ["有価物"], total_yuka)

    return
