import pytest
import pandas as pd
from pathlib import Path
import os
from test.搬入量予測AI.check_csvdata import make_df_old


@pytest.fixture
def sample_csv_files(tmp_path):
    # テスト用の一時ディレクトリを作成
    input_dir = tmp_path / "input"
    input_dir.mkdir()

    # テスト用のCSVファイルを作成
    new_data = pd.DataFrame(
        {
            "伝票日付": ["2024-05-01", "2024-05-02"],
            "正味重量": [100, 200],
            "品名": ["商品A", "商品B"],
        }
    )
    new_data.to_csv(input_dir / "20240501-20250422.csv", index=False, encoding="utf-8")

    data_2020 = pd.DataFrame(
        {
            "伝票日付": ["2020-01-01", "2020-01-02"],
            "商品": ["商品C", "商品D"],
            "正味重量_明細": [300, 400],
        }
    )
    data_2020.to_csv(input_dir / "2020顧客.csv", index=False)

    data_2021 = pd.DataFrame(
        {
            "伝票日付": ["2021-01-01", "2021-01-02"],
            "商品": ["商品E", "商品F"],
            "正味重量_明細": [500, 600],
        }
    )
    data_2021.to_csv(input_dir / "2021顧客.csv", index=False)

    data_2023 = pd.DataFrame(
        {
            "伝票日付": ["2023-01-01", "2023-01-02"],
            "商品": ["商品G", "商品H"],
            "正味重量_明細": [700, 800],
        }
    )
    data_2023.to_csv(input_dir / "2023_all.csv", index=False)

    # テスト用のconfig.yamlを作成
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    with open(config_dir / "config.yaml", "w") as f:
        f.write(f"""
directories:
    input: {str(input_dir)}
        """)

    return str(tmp_path)


def test_make_df_old(sample_csv_files, monkeypatch):
    # 環境変数を設定してconfigファイルの場所を指定
    monkeypatch.setenv(
        "CONFIG_PATH", str(Path(sample_csv_files) / "config/config.yaml")
    )

    # 関数を実行
    result = make_df_old()

    # 各DataFrameの列名を確認
    assert list(result[0].columns) == ["伝票日付", "正味重量", "品名"]
    assert list(result[1].columns) == ["伝票日付", "商品", "正味重量_明細"]
    assert list(result[2].columns) == ["伝票日付", "商品", "正味重量_明細"]
    assert list(result[3].columns) == ["伝票日付", "商品", "正味重量_明細"]

    # データの件数を確認
    assert len(result[0]) == 2  # 2024-2025のデータ
    assert len(result[1]) == 2  # 2020年のデータ
    assert len(result[2]) == 2  # 2021年のデータ
    assert len(result[3]) == 2  # 2023年のデータ
