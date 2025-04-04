# Python 3.9 の軽量版イメージをベースにする
FROM python:3.9-slim

# 作業ディレクトリの指定
WORKDIR /work

# requirements.txt をコピー
COPY requirements.txt /work/

# パッケージアップグレードと依存関係のインストール
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# アプリのソースコードをすべてコピー
COPY . /work/

# ポート開放（Streamlitデフォルト）
EXPOSE 8501

# 実行するStreamlitファイルをlogin.pyに変更（ここが重要！）
CMD bash
