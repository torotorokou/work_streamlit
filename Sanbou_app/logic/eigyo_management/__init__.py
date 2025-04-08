# processors/__init__.py

from . import average_sheet, factory_report, balance_sheet, management_sheet

# テンプレートと処理の対応表
template_processors = {
    "average_sheet": average_sheet.process,
    "factory_report": factory_report.process,
    "balance_sheet": balance_sheet.process,
    "management_sheet": management_sheet.process
}
