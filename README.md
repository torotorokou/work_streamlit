# CSV事務処理アプリ

このアプリはCSVを読み込み、Excelテンプレートに出力し、カレンダー表示も行うStreamlitアプリです。


## 🚀 Dockerでの実行方法

### 1. Dockerイメージをビルド
```bash
docker build -t my-streamlit-app .
```

### 2. アプリを起動
```bash
docker run -p 8501:8501 my-streamlit-app
```

### (オプション) docker-compose で起動
```bash
docker-compose up
```
