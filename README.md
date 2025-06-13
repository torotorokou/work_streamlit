# CSV事務処理アプリ

Streamlitバージョンの参謀くんWebアプリです。


# 開発環境セットアップ手順

## 1. makeコマンドのインストール

### Windowsの場合

[Chocolatey](https://chocolatey.org/)を使う場合:

```sh
choco install make
```

[Git for Windows](https://gitforwindows.org/)をインストールすると、Git Bash内で`make`が利用できる場合があります。

---

## 2. 開発用コンテナのビルド

以下を実行してください。

```sh
make dev_rebuild
```

## 3. コンテナでの作業

`make dev_rebuild` 実行後、`sanbou_dev-app` という名前のコンテナが作成されます。

作業はこのコンテナ内で行います。入るには:

```sh
docker exec -it sanbou_dev-app bash
```

あとはコンテナ内で開発作業を進めてください。

---

## 4. Streamlitアプリの起動（ポート指定）

以下のコマンドで、Streamlitアプリをポート8504で起動できます。

```sh
make st-up
```

このコマンドは、ポート8504を使用中のプロセスを自動で停止し、  
`app/app.py` を `0.0.0.0:8504` で起動します。

**便利な点:**
- ポート競合を自動で解消できるため、手動でプロセスを探して停止する手間が省けます。
- 複数のStreamlitアプリを異なるポートで同時に起動・開発できます。
- `0.0.0.0`で起動するため、同じネットワーク内の他の端末からもアクセスできます。

ブラウザで [http://localhost:8504](http://localhost:8504) にアクセスしてください。
