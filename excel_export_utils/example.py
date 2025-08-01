"""
使用例とテストコード
"""

import pandas as pd
from write_excel import write_values_to_template, safe_excel_value
from xlsxwriter_utils import convert_df_to_excel_bytes, simple_export_to_excel


def example_template_output():
    """テンプレート出力の例"""
    # サンプルデータ（セル・値形式）
    df = pd.DataFrame(
        {
            "セル": ["A1", "B1", "A2", "B2", "C2"],
            "値": ["項目名", "金額", "商品A", 1500, 2000.50],
        }
    )

    # 注意: 実際のテンプレートファイルが必要
    # excel_data = write_values_to_template(df, "template.xlsx", "20250131")
    print("テンプレート出力用データ:")
    print(df)
    print("\n各値の変換結果:")
    for idx, row in df.iterrows():
        print(f"{row['セル']}: {row['値']} -> {safe_excel_value(row['値'])}")


def example_formatted_output():
    """フォーマット付き出力の例"""
    # 通常のDataFrame
    df = pd.DataFrame(
        {
            "大項目": ["材料費", "人件費", "経費"],
            "中項目": ["鉄鋼", "作業員", "光熱費"],
            "単価": [1500.555, 2000.0, 800.25],
            "台数": [10, 5, 3],
            "合計金額": [15005.55, 10000.0, 2400.75],
        }
    )

    print("フォーマット付き出力用データ:")
    print(df)

    # Excel出力（実際のファイル作成）
    excel_bytes = convert_df_to_excel_bytes(df)

    # ファイル保存の例
    with open("example_formatted.xlsx", "wb") as f:
        f.write(excel_bytes.getvalue())
    print("ファイル 'example_formatted.xlsx' を作成しました")


def example_simple_output():
    """シンプル出力の例"""
    df = pd.DataFrame(
        {
            "ID": [1, 2, 3],
            "名前": ["田中", "佐藤", "鈴木"],
            "年齢": [25, 30, 35],
            "部署": ["営業", "開発", "総務"],
        }
    )

    print("シンプル出力用データ:")
    print(df)

    # Excel出力
    excel_data = simple_export_to_excel(df, "社員一覧")

    # ファイル保存の例
    with open("example_simple.xlsx", "wb") as f:
        f.write(excel_data)
    print("ファイル 'example_simple.xlsx' を作成しました")


if __name__ == "__main__":
    print("=== Excel出力ユーティリティ 使用例 ===\n")

    print("1. テンプレート出力の例")
    print("-" * 40)
    example_template_output()

    print("\n2. フォーマット付き出力の例")
    print("-" * 40)
    example_formatted_output()

    print("\n3. シンプル出力の例")
    print("-" * 40)
    example_simple_output()

    print("\n=== 完了 ===")
