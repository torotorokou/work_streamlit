from logic.models.csv_processor import process_csv_by_date, check_date_alignment
from logic.views.message_display import show_success, show_warning, show_error, show_date_mismatch
from utils.file_loader import load_uploaded_csv_files
from utils.preprocessor import enforce_dtypes
from utils.data_schema import load_expected_dtypes
from utils.config_loader import load_config


def prepare_csv_data(uploaded_files: dict, date_columns: dict) -> dict:
    show_success("📄 これから書類を作成します...")
    dfs = load_uploaded_csv_files(uploaded_files)

    config = load_config()
    expected_dtypes = load_expected_dtypes(config)

    for key in dfs:
        dfs[key] = enforce_dtypes(dfs[key], expected_dtypes)

    show_success("📄 CSVの日付を確認中です...")

    for key, df in dfs.items():
        date_col = date_columns.get(key)

        if not date_col:
            show_warning(f"⚠️ {key} の日付カラム定義が存在しません。")
            return {}

        if date_col not in df.columns:
            show_warning(f"⚠️ {key} のCSVに「{date_col}」列が見つかりませんでした。")
            return {}

        dfs[key] = process_csv_by_date(df, date_col)

    result = check_date_alignment(dfs, date_columns)
    if not result["status"]:
        show_error(result["error"])
        if "details" in result:
            show_date_mismatch(result["details"])
        return {}

    show_success(f"✅ すべてのCSVで日付が一致しています：{result['dates']}")
    return dfs
