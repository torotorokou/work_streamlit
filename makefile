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
prod:
	docker-compose -p sanbou_dev -f docker/docker-compose.dev.yml down || true
	docker-compose -p sanbou_prod -f docker/docker-compose.prod.yml up

# コンテナ停止（sanbou_app系）
down:
	docker-compose -p $(PROJECT_NAME) down

# sanbou_app系のコンテナ・ボリューム・イメージを削除
clean:
	docker-compose -p $(PROJECT_NAME) -f docker/docker-compose.dev.yml down --volumes --rmi local
	docker-compose -p $(PROJECT_NAME) -f docker/docker-compose.prod.yml down --volumes --rmi local
	docker system prune -a -f
