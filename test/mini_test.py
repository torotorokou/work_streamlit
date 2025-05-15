import pandas as pd
from typing import Dict
from utils.debug_tools import save_debug_parquets
from utils.logger import app_logger, debug_logger
from utils.config_loader import load_yaml


data = load_yaml("app_config", section="config_files")
print(data)
