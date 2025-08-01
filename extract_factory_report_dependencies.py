#!/usr/bin/env python3
"""
factory_report.pyとその依存関係にあるすべての.pyファイルを収集してzipファイルにパッケージするスクリプト
"""

import ast
import zipfile
from pathlib import Path
from typing import Set, List


def extract_imports_from_file(file_path: Path) -> Set[str]:
    """
    Pythonファイルから相対インポートと絶対インポートを抽出する
    """
    imports = set()

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        tree = ast.parse(content)

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.add(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.add(node.module)
    except Exception as e:
        print(f"エラー: {file_path} の解析に失敗しました: {e}")

    return imports


def find_local_modules(imports: Set[str], base_path: Path) -> Set[Path]:
    """
    インポートからローカルのPythonファイルを特定する
    """
    local_files = set()

    for imp in imports:
        # 相対パスのマッピング
        possible_paths = []

        # 絶対パス形式 (utils.logger -> utils/logger.py)
        if "." in imp:
            parts = imp.split(".")
            possible_paths.append(base_path / ("/".join(parts) + ".py"))
            possible_paths.append(base_path / "/".join(parts) / "__init__.py")
        else:
            possible_paths.append(base_path / (imp + ".py"))
            possible_paths.append(base_path / imp / "__init__.py")

        # ファイルの存在確認
        for path in possible_paths:
            if path.exists() and path.is_file():
                local_files.add(path)

    return local_files


def collect_all_dependencies(start_file: Path, base_path: Path) -> Set[Path]:
    """
    開始ファイルから再帰的にすべての依存関係を収集する
    """
    all_files = set()
    processed = set()
    to_process = [start_file]

    while to_process:
        current_file = to_process.pop()

        if current_file in processed:
            continue

        processed.add(current_file)
        all_files.add(current_file)

        print(f"処理中: {current_file}")

        # このファイルのインポートを抽出
        imports = extract_imports_from_file(current_file)
        local_modules = find_local_modules(imports, base_path)

        # 新しく見つかったファイルを処理キューに追加
        for module_file in local_modules:
            if module_file not in processed:
                to_process.append(module_file)

    return all_files


def collect_additional_files(base_path: Path) -> List[Path]:
    """
    設定ファイルやテンプレートなど、コード以外の必要なファイルを収集
    """
    additional_files = []

    # 設定ファイル
    config_patterns = [
        "config/**/*.yaml",
        "config/**/*.yml",
        "config/**/*.py",
    ]

    # データファイル（テンプレートやマスター）
    data_patterns = [
        "data/master/factory_report/**/*",
        "data/templates/factory_report.*",
    ]

    all_patterns = config_patterns + data_patterns

    for pattern in all_patterns:
        for file_path in base_path.glob(pattern):
            if file_path.is_file():
                additional_files.append(file_path)

    return additional_files


def create_dependency_zip(base_path: Path, output_zip: Path):
    """
    メイン関数: factory_report.pyの依存関係をzipファイルにパッケージ
    """
    print(f"ベースパス: {base_path}")

    # メインファイル
    main_file = base_path / "logic/manage/factory_report.py"

    if not main_file.exists():
        print(f"エラー: メインファイル {main_file} が見つかりません")
        return

    print("依存関係を収集中...")

    # すべての依存するPythonファイルを収集
    all_python_files = collect_all_dependencies(main_file, base_path)

    # 追加の設定/データファイルを収集
    additional_files = collect_additional_files(base_path)

    # 重複を除去するために、すべてのファイルを結合してセットにする
    all_files = set(all_python_files) | set(additional_files)

    print(f"\n収集されたファイル ({len(all_files)}個):")
    for f in sorted(all_files):
        print(f"  {f.relative_to(base_path)}")

    # zipファイル作成
    print(f"\nzipファイル作成中: {output_zip}")
    with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as zipf:
        # すべてのファイルを追加
        for file_path in all_files:
            arcname = file_path.relative_to(base_path)
            zipf.write(file_path, arcname)

    print(f"✅ 完了: {output_zip}")
    print(f"📦 総ファイル数: {len(all_files)}")


if __name__ == "__main__":
    # パス設定
    base_path = Path("/work/app")
    output_zip = Path("/work/factory_report_dependencies.zip")

    # 既存のzipファイルがあれば削除
    if output_zip.exists():
        output_zip.unlink()

    # 実行
    create_dependency_zip(base_path, output_zip)
