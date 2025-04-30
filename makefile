# --- 環境設定 ---
PROJECT_NAME=sanbou_app

# --- コマンド ---

# 開発環境起動
dev:
	docker-compose -p sanbou_dev -f docker/docker-compose.dev.yml up

# ステージング環境起動
staging:
	docker-compose -p sanbou_staging -f docker/docker-compose.staging.yml up

# 本番環境起動
# .envファイルから環境変数を読み込んで本番環境起動

# 環境ファイルのパス
ENV_FILE=.env

# プロジェクト名
PROJECT_NAME=sanbou_prod

# Docker Composeファイル
COMPOSE_FILE=docker/docker-compose.prod.yml

# Dockerfileの場所（buildに使う）
DOCKERFILE=docker/prod.Dockerfile

# Dockerイメージ名
IMAGE_NAME=sanboukun:prod

prod:
	@echo "🔄 本番環境を再構築して起動します..."
	@echo "📦 .env を読み込み中..."
	@set -a && . $(ENV_FILE) && set +a && \
	docker build \
		--build-arg GITHUB_TOKEN=$$GITHUB_TOKEN \
		--build-arg REPO_TAG=$$REPO_TAG \
		-f $(DOCKERFILE) \
		-t $(IMAGE_NAME) .

	docker-compose -p $(PROJECT_NAME) -f $(COMPOSE_FILE) down || true
	docker-compose -p $(PROJECT_NAME) -f $(COMPOSE_FILE) up -d

# コンテナ停止（sanbou_app系）
down:
	docker-compose -p $(PROJECT_NAME) down

# sanbou_app系のコンテナ・ボリューム・イメージを削除
clean:
	docker-compose -p $(PROJECT_NAME) -f docker/docker-compose.dev.yml down --volumes --rmi local
	docker-compose -p $(PROJECT_NAME) -f docker/docker-compose.prod.yml down --volumes --rmi local
	docker system prune -a -f

# --- Streamlit管理 ---

# 開発コンテナ内でStreamlit起動
st-up:
	streamlit run app/app.py --server.port=8504 --server.address=0.0.0.0 --server.headless=false --server.enableCORS=false

# ポート8504を使ってるプロセスをkillする（確実）
st-kill:
	@fuser -k 8504/tcp || true
