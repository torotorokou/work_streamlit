# --- 共通設定 ---
ENV_FILE=.env
PROJECT_NAME=sanbou_prod
COMPOSE_FILE=docker/docker-compose.prod.yml
DOCKERFILE=docker/prod.Dockerfile
IMAGE_NAME=sanboukun:prod

# --- 環境起動 ---

dev:
	docker-compose -p sanbou_dev -f docker/docker-compose.dev.yml up

dev_rebuild:
	docker-compose -p sanbou_dev -f docker/docker-compose.dev.yml up --build --force-recreate


staging:
	docker-compose -p sanbou_staging -f docker/docker-compose.staging.yml up


# 本番ビルド＆起動
prod:
	@echo "Starting production environment rebuild..."
	@echo "Loading .env and starting services..."

	powershell -Command "$$envs = Get-Content .env | Where-Object { $$_ -match '^[^#].*=.*' } | ForEach-Object { $$kv = $$_ -split '=', 2; Set-Item -Path Env:$$($$kv[0].Trim()) -Value $$($$kv[1].Trim()) }; \
	docker-compose -p sanbou_prod -f docker/docker-compose.prod.yml build --build-arg GITHUB_TOKEN=$$env:GITHUB_TOKEN --build-arg REPO_TAG=$$env:REPO_TAG --build-arg REPO_URL=$$env:REPO_URL; \
	docker-compose -p sanbou_prod -f docker/docker-compose.prod.yml down -v || true; \
	docker-compose -p sanbou_prod -f docker/docker-compose.prod.yml up -d"

# 再ビルド（キャッシュ無効化）
prod_rebuild:
	@echo "Starting full rebuild with --no-cache..."
	@echo "Reloading .env and rebuilding Docker image..."

	powershell -Command "$$envs = Get-Content .env | Where-Object { $$_ -match '^[^#].*=.*' } | ForEach-Object { $$kv = $$_ -split '=', 2; Set-Item -Path Env:$$($$kv[0].Trim()) -Value $$($$kv[1].Trim()) }; \
	docker-compose -p sanbou_prod -f docker/docker-compose.prod.yml build --no-cache --build-arg GITHUB_TOKEN=$$env:GITHUB_TOKEN --build-arg REPO_TAG=$$env:REPO_TAG --build-arg REPO_URL=$$env:REPO_URL; \
	try { docker-compose -p sanbou_prod -f docker/docker-compose.prod.yml down -v } catch { Write-Host 'down failed (ignored)' }; \
	docker-compose -p sanbou_prod -f docker/docker-compose.prod.yml up -d"
down:
	docker-compose -p $(PROJECT_NAME) down

clean:
	docker-compose -p sanbou_dev -f docker/docker-compose.dev.yml down --volumes --rmi local
	docker-compose -p $(PROJECT_NAME) -f $(COMPOSE_FILE) down --volumes --rmi local
	docker system prune -a -f

# --- Streamlit操作 ---

st-up:
	streamlit run app/app.py --server.port=8504 --server.address=0.0.0.0 --server.headless=false --server.enableCORS=false

st-kill:
	@fuser -k 8504/tcp || true


# --- モニタリング・ログ用ユーティリティ ---

logs-prod:
	docker-compose -p sanbou_prod -f docker/docker-compose.prod.yml logs -f

status-prod:
	docker ps --filter "name=sanbou_prod"

restart-prod:
	docker-compose -p sanbou_prod -f docker/docker-compose.prod.yml restart

logs-dev:
	docker-compose -p sanbou_dev -f docker/docker-compose.dev.yml logs -f

status-dev:
	docker ps --filter "name=sanbou_dev"

restart-dev:
	docker-compose -p sanbou_dev -f docker/docker-compose.dev.yml restart

logs-staging:
	docker-compose -p sanbou_staging -f docker/docker-compose.staging.yml logs -f

status-staging:
	docker ps --filter "name=sanbou_staging"

restart-staging:
	docker-compose -p sanbou_staging -f docker/docker-compose.staging.yml restart


push-all-tags:
	git push origin --tags


commit:
	@git add .
	@read -p "Enter commit message: " msg; \
	git commit -m "$$msg" --no-verify && git push origin HEAD

precommit:
	pre-commit clean
	pre-commit run --all-files > logs/precommit_run.log 2>&1
