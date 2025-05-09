# Python 3.10 slim イメージをベースに使用
FROM python:3.10-slim

# 作業ディレクトリを作成・設定
WORKDIR /work

# 日本語表示対応のためフォントとgitをインストール
RUN apt-get update && \
    apt-get install -y \
    fonts-noto-cjk \
    fonts-noto \
    git && \
    rm -rf /var/lib/apt/lists/*

# requirements.txt だけ先にコピーして、キャッシュを活かす
COPY requirements.txt .

# 必要なPythonライブラリをインストール（プロジェクト依存分）
RUN pip install --no-cache-dir -r requirements.txt

# 開発ツール：整形＆静的解析ツールを追加インストール
RUN pip install --no-cache-dir \
    pre-commit \
    black \
    ruff \
    isort

# プロジェクトファイルをすべてコピー
COPY . /work

# Streamlit のポートを開放（デフォルト8501）
EXPOSE 8501

# アプリ起動コマンド（port=8501, 0.0.0.0で外部公開）
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
