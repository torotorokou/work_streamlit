# Excel出力ユーティリティモジュール

このフォルダには、PythonでDataFrameをExcelファイルに出力するためのユーティリティコードが含まれています。

## 📁 ファイル構成

### `write_excel.py`
- **機能**: Excelテンプレートファイルを使用したデータ出力
- **主要関数**:
  - `write_values_to_template()`: DataFrameのデータをExcelテンプレートに書き込み
  - `load_template_workbook()`: Excelテンプレートファイル読み込み
  - `write_dataframe_to_worksheet()`: ワークシートへのデータ書き込み
  - `safe_excel_value()`: Excel書き込み用の値変換

### `xlsxwriter_utils.py`  
- **機能**: XlsxWriterエンジンを使用した高品質なExcel出力
- **主要関数**:
  - `convert_df_to_excel_bytes()`: 日本語フォント・書式設定付きExcel出力
  - `simple_export_to_excel()`: シンプルなDataFrame→Excel変換

### `custom_button.py`
- **機能**: Streamlit用のスタイル付きボタンコンポーネント  
- **主要関数**:
  - `centered_button()`: 中央寄せボタン
  - `centered_download_button()`: ダウンロードボタン

## 🚀 基本的な使用方法

### 1. テンプレートを使用したExcel出力

```python
import pandas as pd
from write_excel import write_values_to_template

# DataFrameを準備（"セル"列と"値"列が必要）
df = pd.DataFrame({
    "セル": ["A1", "B1", "A2", "B2"],  
    "値": ["項目名", "数値", "データ1", 123]
})

# Excelテンプレートに書き込み
excel_data = write_values_to_template(
    df=df,
    template_path="template.xlsx",
    extracted_date="20250131"
)

# ファイル保存
with open("output.xlsx", "wb") as f:
    f.write(excel_data.getvalue())
```

### 2. 高品質フォーマット付きExcel出力

```python
import pandas as pd
from xlsxwriter_utils import convert_df_to_excel_bytes

# 通常のDataFrame
df = pd.DataFrame({
    "大項目": ["材料費", "人件費"],
    "中項目": ["鉄鋼", "作業員"],
    "単価": [1500.50, 2000.00],
    "台数": [10, 5]
})

# 游ゴシックフォント・書式設定付きでExcel出力
excel_bytes = convert_df_to_excel_bytes(df)

# ファイル保存
with open("formatted_output.xlsx", "wb") as f:
    f.write(excel_bytes.getvalue())
```

### 3. Streamlitでのダウンロードボタン

```python
import streamlit as st
from custom_button import centered_download_button
from xlsxwriter_utils import simple_export_to_excel

# DataFrameを準備
df = pd.DataFrame({"列1": [1, 2, 3], "列2": ["A", "B", "C"]})

# Excel変換
excel_data = simple_export_to_excel(df, "結果データ")

# ダウンロードボタン表示
centered_download_button(
    label="📥 Excelファイルをダウンロード",
    data=excel_data,
    file_name="結果.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)
```

## 🔧 必要な依存関係

このコードを使用するには、以下のパッケージが必要です：

```bash
pip install pandas openpyxl xlsxwriter streamlit
```

## 📋 DataFrameの形式について

### テンプレート出力用（`write_excel.py`）
DataFrameには以下の列が必要です：
- `セル`: Excelのセル参照（例："A1", "B2"）
- `値`: セルに書き込む値

### 通常出力用（`xlsxwriter_utils.py`） 
通常のDataFrame形式で使用可能です。

## 💡 特徴

- **書式保持**: テンプレート出力時にExcelの書式を保持
- **日本語対応**: 游ゴシックフォント設定
- **エラーハンドリング**: 結合セルや無効なセル参照の適切な処理
- **Streamlit統合**: UI用のカスタムボタンコンポーネント
- **型安全**: 型ヒント対応

## ⚠️ 注意事項

- テンプレートファイルは事前に準備が必要です
- セル参照は "A1" 形式で指定してください
- 大量データの場合はメモリ使用量にご注意ください
- 結合セルへの直接書き込みはサポートされていません
