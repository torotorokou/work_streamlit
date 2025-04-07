# processors/__init__.py

from .factory_report import generate_factory_report
from .balance_sheet import generate_balance_sheet
from .average_sheet import generate_average_sheet
from .management_sheet import generate_management_sheet

# テンプレートと処理の対応表
template_processors = {
    "factory_report": generate_factory_report,
    "balance_sheet": generate_balance_sheet,
    "average_sheet": generate_average_sheet,
    "management_sheet": generate_management_sheet,
}
