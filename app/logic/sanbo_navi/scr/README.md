# Sanbo Navi API

これは、Reactフロントエンドと連携するためのFastAPIバックエンドです。

## 起動手順

1. **依存関係のインストール:**
   ```bash
   pip install -r requirements.txt
   ```

2. **APIサーバーの起動:**
   ```bash
   uvicorn main:app --reload
   ```
   サーバーは `http://127.0.0.1:8000` で起動します。

## .env ファイルの設定

プロジェクトのルートに `.env` ファイルを作成し、OpenAIのAPIキーを設定してください。

```
OPENAI_API_KEY=sk-your_actual_api_key
```

## APIエンドポイント

### POST /api/generate-answer

ユーザーの質問とカテゴリに基づいてAIが回答を生成します。

**リクエストボディ:**

```json
{
  "query": "日本の首都は？",
  "category": "地理"
}
```

**レスポンス (成功時):**

```json
{
  "answer": "日本の首都は東京です。",
  "sources": [
    ["Wikipedia", "https://ja.wikipedia.org/wiki/%E6%9D%B1%E4%BA%AC%E9%83%BD"],
    ["政府公式サイト", "https://www.metro.tokyo.lg.jp/"]
  ]
}
```

## CORS (Cross-Origin Resource Sharing)

このバックエンドは、`http://localhost:3000` からのリクエストを許可するように設定されています。これは、開発中にReactフロントエンド（通常ポート3000で実行される）がAPIと通信できるようにするためです。

## ディレクトリ構成

```
scr/
├── api/
│   ├── endpoints.py  # APIエンドポイントの定義
│   └── models.py     # Pydanticデータモデル
├── main.py           # FastAPIアプリケーションのエントリポイント
├── requirements.txt  # プロジェクトの依存関係
├── .env              # 環境変数（APIキーなど）
├── README.md         # このファイル
└── ... (既存のロジックファイル)
```
