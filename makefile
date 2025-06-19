# --- 共通設定 ---
ENV_FILE=.env_file/.env.prod
PROJECT_NAME=sanbou_prod
COMPOSE_FILE=docker/docker-compose.prod.yml
DOCKERFILE=docker/prod.Dockerfile
IMAGE_NAME=sanboukun:prod

# --- 実行環境判定（WSL2判定強化版） ---
IS_WSL := $(shell grep -i microsoft /proc/version > /dev/null && echo 1 || echo 0)

# --- Linuxシェルを使わせる ---
SHELL := /bin/bash
.ONESHELL:

# --- 環境起動 ---


dev_rebuild:
	@echo "Starting dev rebuild with --no-cache..."
	docker compose -p sanbou_dev -f docker/docker-compose.dev.yml down -v
	docker compose -p sanbou_dev -f docker/docker-compose.dev.yml build --no-cache
	docker compose -p sanbou_dev -f docker/docker-compose.dev.yml up -d


# --- Streamlit操作 ---

st-up:
	@echo "🔌 Killing any process using port 8504..."
	@fuser -k 8504/tcp || true
	@echo "🚀 Starting Streamlit app on port 8504..."
	streamlit run app/app.py \
		--server.port=8504 \
		--server.address=0.0.0.0 \
		--server.headless=false \
		--server.enableCORS=false \
		--server.enableXsrfProtection=false


# dev:
# 	docker-compose -p sanbou_dev -f docker/docker-compose.dev.yml up

# ステージング再ビルド（キャッシュ無効化）
# staging_rebuild:
# 	@echo "Starting full staging rebuild with --no-cache..."
# 	@echo "Reloading .env_file/.env.staging and rebuilding Docker image..."

# 	powershell -Command "$$envs = Get-Content .env_file/.env.staging | Where-Object { $$_ -match '^[^#].*=.*' } | ForEach-Object { $$kv = $$_ -split '=', 2; Set-Item -Path Env:$$($$kv[0].Trim()) -Value $$($$kv[1].Trim()) }; \
# 	docker-compose -p sanbou_staging -f docker/docker-compose.prod.yml build --no-cache \
# 		--build-arg GITHUB_TOKEN=$$env:GITHUB_TOKEN \
# 		--build-arg REPO_TAG=$$env:REPO_TAG \
# 		--build-arg REPO_URL=$$env:REPO_URL \
# 		--build-arg STAGE_ENV=staging \
# 		--build-arg ENV_FILE=.env_file/.env.staging; \
# 	try { docker-compose -p sanbou_staging -f docker/docker-compose.prod.yml down -v } catch { Write-Host 'down failed (ignored)' }; \
# 	docker-compose -p sanbou_staging -f docker/docker-compose.prod.yml up -d"


# ステージング再ビルド（キャッシュ無効化）
staging_rebuild:
ifeq ($(IS_WSL),1)
	@echo "WSL2 Linux環境で実行中"
	@echo "Loading environment variables from .env_file/.env.staging..."

	@if [ -f .env_file/.env.staging ]; then \
		set -a && source .env_file/.env.staging && set +a && \
		docker compose -p sanbou_staging -f docker/docker-compose.prod.yml build --no-cache \
			--build-arg GITHUB_TOKEN=$${GITHUB_TOKEN} \
			--build-arg REPO_TAG=$${REPO_TAG} \
			--build-arg REPO_URL=$${REPO_URL} \
			--build-arg STAGE_ENV=staging \
			--build-arg ENV_FILE=.env_file/.env.staging && \
		docker compose -p sanbou_staging -f docker/docker-compose.prod.yml down -v || true && \
		docker compose -p sanbou_staging -f docker/docker-compose.prod.yml up -d; \
	else \
		echo "Error: Environment file .env_file/.env.staging not found."; \
		exit 1; \
	fi
else
	@echo "Windows (Docker Desktop) 環境で実行中"
	powershell -Command "$$envs = Get-Content .env_file/.env.staging | Where-Object { $$_ -match '^[^#].*=.*' } | ForEach-Object { $$kv = $$_ -split '=', 2; Set-Item -Path Env:$$($$kv[0].Trim()) -Value $$($$kv[1].Trim()) }; \
	docker-compose -p sanbou_staging -f docker/docker-compose.prod.yml build --no-cache \
		--build-arg GITHUB_TOKEN=$$env:GITHUB_TOKEN \
		--build-arg REPO_TAG=$$env:REPO_TAG \
		--build-arg REPO_URL=$$env:REPO_URL \
		--build-arg STAGE_ENV=staging \
		--build-arg ENV_FILE=.env_file/.env.staging; \
	try { docker-compose -p sanbou_staging -f docker/docker-compose.prod.yml down -v } catch { Write-Host 'down failed (ignored)' }; \
	docker-compose -p sanbou_staging -f docker/docker-compose.prod.yml up -d"
endif



# ステージング再ビルド（キャッシュ無効化）
another_staging_rebuild:
	@echo "Starting full another_staging rebuild with --no-cache..."
	@echo "Reloading .env_file/.env.another_staging and rebuilding Docker image..."

	powershell -Command "$$envs = Get-Content .env_file/.env.another_staging | Where-Object { $$_ -match '^[^#].*=.*' } | ForEach-Object { $$kv = $$_ -split '=', 2; Set-Item -Path Env:$$($$kv[0].Trim()) -Value $$($$kv[1].Trim()) }; \
	docker-compose -p sanbou_another_staging -f docker/docker-compose.prod.yml build --no-cache \
		--build-arg GITHUB_TOKEN=$$env:GITHUB_TOKEN \
		--build-arg REPO_TAG=$$env:REPO_TAG \
		--build-arg REPO_URL=$$env:REPO_URL \
		--build-arg STAGE_ENV=another_staging \
		--build-arg ENV_FILE=.env_file/.env.another_staging; \
	try { docker-compose -p sanbou_another_staging -f docker/docker-compose.prod.yml down -v } catch { Write-Host 'down failed (ignored)' }; \
	docker-compose -p sanbou_another_staging -f docker/docker-compose.prod.yml up -d"


# 本番ビルド＆起動（キャッシュあり）
prod:
	@echo "Starting production environment rebuild..."
	@echo "Loading .env.prod and starting services..."


	powershell -Command "$$envs = Get-Content .env.prod | Where-Object { $$_ -match '^[^#].*=.*' } | ForEach-Object { $$kv = $$_ -split '=', 2; Set-Item -Path Env:$$($$kv[0].Trim()) -Value $$($$kv[1].Trim()) }; \
	docker-compose -p sanbou_prod -f docker/docker-compose.prod.yml build --build-arg GITHUB_TOKEN=$$env:GITHUB_TOKEN --build-arg REPO_TAG=$$env:REPO_TAG --build-arg REPO_URL=$$env:REPO_URL; \
	docker-compose -p sanbou_prod -f docker/docker-compose.prod.yml down -v || true; \
	docker-compose -p sanbou_prod -f docker/docker-compose.prod.yml up -d"


prod_rebuild:
ifeq ($(IS_WSL),1)
	@echo "WSL2 Linux環境で実行中"
	@echo "Loading environment variables from $(ENV_FILE)..."

	@if [ -f $(ENV_FILE) ]; then \
		set -a && source $(ENV_FILE) && set +a && \
		docker compose -p $(PROJECT_NAME) -f $(COMPOSE_FILE) build --no-cache \
			--build-arg GITHUB_TOKEN=$${GITHUB_TOKEN} \
			--build-arg REPO_TAG=$${REPO_TAG} \
			--build-arg REPO_URL=$${REPO_URL} \
			--build-arg ENV_FILE=$(ENV_FILE) && \
		docker compose -p $(PROJECT_NAME) -f $(COMPOSE_FILE) down -v || true && \
		docker compose -p $(PROJECT_NAME) -f $(COMPOSE_FILE) up -d; \
	else \
		echo "Error: Environment file $(ENV_FILE) not found."; \
		exit 1; \
	fi
else
	@echo "Windows (Docker Desktop) 環境で実行中"
	powershell -Command "$$envs = Get-Content $(ENV_FILE) | Where-Object { $$_ -match '^[^#].*=.*' } | ForEach-Object { $$kv = $$_ -split '=', 2; Set-Item -Path Env:$$($$kv[0].Trim()) -Value $$($$kv[1].Trim()) }; \
	docker-compose -p $(PROJECT_NAME) -f $(COMPOSE_FILE) build --no-cache \
		--build-arg GITHUB_TOKEN=$$env:GITHUB_TOKEN \
		--build-arg REPO_TAG=$$env:REPO_TAG \
		--build-arg REPO_URL=$$env:REPO_URL \
		--build-arg ENV_FILE=$(ENV_FILE); \
	try { docker-compose -p $(PROJECT_NAME) -f $(COMPOSE_FILE) down -v } catch { Write-Host 'down failed (ignored)' }; \
	docker-compose -p $(PROJECT_NAME) -f $(COMPOSE_FILE) up -d"
endif

st_debug:
	streamlit run app/app_debug/upload_test.py --server.port=8505


down:
	docker-compose -p $(PROJECT_NAME) down

clean:
	docker-compose -p sanbou_dev -f docker/docker-compose.dev.yml down --volumes --rmi local
	docker-compose -p $(PROJECT_NAME) -f $(COMPOSE_FILE) down --volumes --rmi local
	docker system prune -a -f



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
	make black
	@git add .
	@read -p "Enter commit message: " msg; \
	git commit -m "$$msg" --no-verify && git push origin HEAD

precommit:
	pre-commit clean
	pre-commit run --all-files > logs/precommit_run.log 2>&1

black:
	black app --verbose