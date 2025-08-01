# Excel出力ユーティリティパッケージ

このパッケージは、元のStreamlitアプリケーションからExcel出力に関連するコードを抜き出し、再利用可能なユーティリティとしてまとめたものです。

## 📦 パッケージ内容

### 🔧 核心機能
- **テンプレートベースExcel出力**: 既存のExcelテンプレートにデータを埋め込み
- **高品質フォーマット出力**: 日本語フォント・書式設定付きExcel生成
- **Streamlit統合**: UI用カスタムボタンコンポーネント

### 📁 ファイル構成
```
excel_export_utils/
├── __init__.py              # パッケージ初期化
├── write_excel.py           # テンプレート出力機能
├── xlsxwriter_utils.py      # フォーマット付き出力機能
├── custom_button.py         # Streamlitボタンコンポーネント
├── example.py               # 使用例・テストコード
├── requirements.txt         # 依存関係
└── README.md               # 詳細ドキュメント
```

## 🚀 クイックスタート

### インストール
```bash
# 依存関係をインストール
pip install -r requirements.txt
```

### 基本的な使用方法

#### 1. テンプレートを使用したExcel出力
```python
import pandas as pd
from excel_export_utils import write_values_to_template

# データ準備（セル・値形式）
df = pd.DataFrame({
    "セル": ["A1", "B1", "A2"],
    "値": ["項目", "金額", 1500]
})

# テンプレートに書き込み
excel_data = write_values_to_template(df, "template.xlsx", "20250131")
```

#### 2. フォーマット付きExcel出力
```python
from excel_export_utils import convert_df_to_excel_bytes

# 通常のDataFrame
df = pd.DataFrame({
    "項目": ["材料費", "人件費"],
    "単価": [1500.50, 2000.00]
})

# 游ゴシックフォント付きでExcel出力
excel_bytes = convert_df_to_excel_bytes(df)
```

#### 3. Streamlitでのダウンロード
```python
import streamlit as st
from excel_export_utils import centered_download_button

centered_download_button(
    label="📥 Excelダウンロード",
    data=excel_bytes,
    file_name="結果.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)
```

## 📋 元のコードからの抽出内容

このパッケージは以下の元ファイルから主要機能を抽出・統合しています：

### 抽出元ファイル
- `app/utils/write_excel.py` → `write_excel.py`
- `app/components/custom_button.py` → `custom_button.py` 
- `app/app_pages/factory_manage/pages/balance_management_table/excel_config.py` → `xlsxwriter_utils.py`
- `app/logic/excel_exporter.py` → `xlsxwriter_utils.py`（統合）

### 主要な変更点
- **依存関係の最小化**: アプリ固有の依存を削除し、標準ライブラリ中心に変更
- **エラーハンドリング強化**: より堅牢なエラー処理を追加
- **型ヒント追加**: Python 3.10+の型ヒント記法を使用
- **モジュール化**: 機能別にファイルを分割し、再利用性を向上

## 🔧 技術仕様

### 対応フォーマット
- **入力**: pandas DataFrame
- **出力**: Excel (.xlsx)
- **テンプレート**: Excel テンプレートファイル

### 主要ライブラリ
- `pandas`: データ処理
- `openpyxl`: Excelファイル読み書き（テンプレート用）
- `xlsxwriter`: 高品質Excel生成
- `streamlit`: UI コンポーネント

### システム要件
- Python 3.8+
- メモリ: 中程度のデータセット対応
- OS: Windows/Linux/macOS

## 💡 使用シーン

### 適用ケース
- **帳票システム**: 定型Excel帳票の自動生成
- **データエクスポート**: WebアプリからのExcel出力機能
- **レポート作成**: 分析結果の整形出力
- **業務システム**: 既存Excelテンプレートの活用

### メリット
- **書式保持**: 既存テンプレートの見た目を維持
- **日本語対応**: 游ゴシックフォント等の日本語環境最適化
- **Streamlit統合**: Webアプリでの即座な活用可能
- **再利用性**: モジュール化により他プロジェクトでも使用可能

## ⚠️ 注意事項

- テンプレートファイルは事前準備が必要
- 大量データ処理時はメモリ使用量に注意
- 結合セルへの直接書き込みは制限あり
- Streamlit関連機能はStreamlit環境でのみ動作

## 📞 サポート

詳細な使用方法は `README.md` および `example.py` を参照してください。

---
*元のコードから抽出・再構築 - 2025年7月*
