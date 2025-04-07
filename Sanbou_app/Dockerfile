FROM python:3.10-slim

# 作業ディレクトリの設定
WORKDIR /work

# 必要ファイルをコピー
COPY . /work

# フォント追加（Google Notoフォントなど日本語対応）
RUN apt-get update && apt-get install -y \
    fonts-noto-cjk \
    fonts-noto \
    && rm -rf /var/lib/apt/lists/*

# Pythonライブラリのインストール
RUN pip install --no-cache-dir -r requirements.txt

# Streamlitのデフォルトポートを開放
EXPOSE 8501

# アプリ起動コマンド
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
