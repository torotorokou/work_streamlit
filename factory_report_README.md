# Factory Report 移植パッケージ

このzipファイルには、`factory_report.py`とその依存関係にあるすべてのファイルが含まれています。

## 概要

`factory_report.py`は工場日報テンプレート用のメイン処理関数で、各種CSVデータを読み込み、処分・有価・ヤード等の処理を適用し、最終的な工場日報データフレームを生成します。

## 含まれているファイル

### メインファイル
- `logic/manage/factory_report.py` - メイン処理関数

### プロセッサー
- `logic/manage/processors/factory_report/` - 工場日報専用の処理モジュール群
  - `factory_report_shobun.py` - 出荷処分データ処理
  - `factory_report_yuuka.py` - 出荷有価データ処理  
  - `factory_report_yard.py` - 出荷ヤードデータ処理
  - `make_cell_num.py` - セル番号設定
  - `make_label.py` - ラベル追加
  - `summary.py` - 集計処理
  - `etc.py` - その他処理（合計・総合計行追加、日付フォーマット等）

### ユーティリティ
- `logic/manage/utils/` - 管理系ユーティリティ
  - `csv_loader.py` - CSV読み込み処理
  - `excel_tools.py` - Excel関連ツール
  - `load_template.py` - テンプレート読み込み
  - `summary_tools.py` - 集計ツール
- `utils/` - 汎用ユーティリティ
  - `config_loader.py` - 設定ファイル読み込み
  - `logger.py` - ログ処理
  - `data_cleaning.py` - データクリーニング
  - `date_tools.py` - 日付処理
  - `value_setter.py` - 値設定ツール
  - `type_converter.py` - 型変換
  - `debug_tools.py` - デバッグツール

### 設定ファイル
- `config/` - 設定ファイル群
  - `templates_config.yaml` - テンプレート設定
  - `main_paths.yaml` - パス設定
  - `required_columns_definition.yaml` - 必須カラム定義
  - その他設定ファイル

### データファイル
- `data/master/factory_report/` - マスターデータ
  - `shobun_map.csv` - 処分マッピング
  - `yuka_map.csv` - 有価マッピング
  - `yard_map.csv` - ヤードマッピング
  - `etc.csv` - その他データ
- `data/templates/factory_report.xlsx` - Excelテンプレート

## 依存関係

このコードは以下のPythonライブラリに依存しています：
- pandas
- pyyaml  
- openpyxl (Excel処理用)
- streamlit (一部のモジュールで使用)

## 移植手順

1. zipファイルを展開
2. 必要なPythonライブラリをインストール:
   ```bash
   pip install pandas pyyaml openpyxl streamlit
   ```
3. プロジェクトのベースパスを`config/main_paths.yaml`で調整
4. データファイルのパスを環境に合わせて調整

## 使用方法

```python
from logic.manage.factory_report import process

# CSVデータの辞書を準備
dfs = {
    "shipping": shipping_dataframe,
    "yard": yard_dataframe
}

# 処理実行
result_df = process(dfs)
```

## 注意事項

- このパッケージはStreamlit環境での使用を前提として開発されています
- 設定ファイルのパスは絶対パスまたは相対パスで適切に設定する必要があります
- ログファイルの出力先ディレクトリが存在することを確認してください

## 生成日時

生成日: 2025年7月31日
