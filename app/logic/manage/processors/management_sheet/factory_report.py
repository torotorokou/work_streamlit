from utils.value_setter import set_value_fast_safe
from logic.manage.factory_report import process as process_fact
from logic.manage.utils.dataframe_tools import apply_summary_by_item




# 工場日報からの読込
def update_from_factory_report(dfs, master_csv):
    csv_fac = process_fact(dfs)

    # 有価物
    master_csv = apply_summary_by_item(master_csv, csv_fac, "有価物")
    # シュレッダー
    master_csv = apply_summary_by_item(master_csv, csv_fac, "シュレッダー")

    return master_csv
