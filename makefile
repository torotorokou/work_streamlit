# --- 共通設定 ---
ENV_FILE=.env
PROJECT_NAME=sanbou_prod
COMPOSE_FILE=docker/docker-compose.prod.yml
DOCKERFILE=docker/prod.Dockerfile
IMAGE_NAME=sanboukun:prod

# --- 環境起動 ---


dev_rebuild:
	@echo "🔧 このコマンドはローカル開発環境でのみ実行してください。"
	@echo "🛠️  sanbou_dev 用の Docker コンテナを --no-cache で再構築します..."
	docker-compose -p sanbou_dev -f docker/docker-compose.dev.yml down -v
	docker-compose -p sanbou_dev -f docker/docker-compose.dev.yml build --no-cache
	docker-compose -p sanbou_dev -f docker/docker-compose.dev.yml up -d

# --- Streamlit操作 ---

st-up:
	@echo "🐳 このコマンドは Docker コンテナ内で実行してください。"
	@echo "📌 Streamlit アプリ（app/app.py）をポート 8504 で起動します。"
	@echo "⚠️  起動時にエラーになる場合がありますが、その場合は数回再実行してください。"
	@echo "🔌 ポート 8504 を使用中のプロセスを強制終了します..."
	@fuser -k 8504/tcp || true
	@echo "🚀 Streamlit アプリを起動中..."
	streamlit run app/app.py \
		--server.port=8504 \
		--server.address=0.0.0.0 \
		--server.headless=false \
		--server.enableCORS=false \
		--server.enableXsrfProtection=false


# dev:
# 	docker-compose -p sanbou_dev -f docker/docker-compose.dev.yml up


# ステージングビルド＆起動（キャッシュあり）
# staging:
# 	@echo "Starting staging environment rebuild..."
# 	@echo "Loading .env.staging and starting services..."

# 	powershell -Command "$$envs = Get-Content .env.staging | Where-Object { $$_ -match '^[^#].*=.*' } | ForEach-Object { $$kv = $$_ -split '=', 2; Set-Item -Path Env:$$($$kv[0].Trim()) -Value $$($$kv[1].Trim()) }; \
# 	docker-compose -p sanbou_staging -f docker/docker-compose.prod.yml build \
# 		--build-arg GITHUB_TOKEN=$$env:GITHUB_TOKEN \
# 		--build-arg REPO_TAG=$$env:REPO_TAG \
# 		--build-arg REPO_URL=$$env:REPO_URL \
# 		--build-arg STAGE_ENV=staging \
# 		--build-arg ENV_FILE=.env.staging; \
# 	docker-compose -p sanbou_staging -f docker/docker-compose.prod.yml down -v || true; \
# 	docker-compose -p sanbou_staging -f docker/docker-compose.prod.yml up -d"

# ステージング再ビルド（キャッシュ無効化）
staging_rebuild:
	@echo "Starting full staging rebuild with --no-cache..."
	@echo "Reloading .env_file/.env.staging and rebuilding Docker image..."

	powershell -Command "$$envs = Get-Content .env_file/.env.staging | Where-Object { $$_ -match '^[^#].*=.*' } | ForEach-Object { $$kv = $$_ -split '=', 2; Set-Item -Path Env:$$($$kv[0].Trim()) -Value $$($$kv[1].Trim()) }; \
	docker-compose -p sanbou_staging -f docker/docker-compose.prod.yml build --no-cache \
		--build-arg GITHUB_TOKEN=$$env:GITHUB_TOKEN \
		--build-arg REPO_TAG=$$env:REPO_TAG \
		--build-arg REPO_URL=$$env:REPO_URL \
		--build-arg STAGE_ENV=staging \
		--build-arg ENV_FILE=.env_file/.env.staging; \
	try { docker-compose -p sanbou_staging -f docker/docker-compose.prod.yml down -v } catch { Write-Host 'down failed (ignored)' }; \
	docker-compose -p sanbou_staging -f docker/docker-compose.prod.yml up -d"


# ステージング再ビルド（キャッシュ無効化）Linux用
staging_rebuild_linux:
	@echo "Starting full staging rebuild with --no-cache... (Linux)"
	@echo "Reloading .env_file/.env.staging and rebuilding Docker image..."

	set -a; \
	. .env_file/.env.staging; \
	set +a; \
	docker-compose -p sanbou_staging -f docker/docker-compose.prod.yml build --no-cache \
		--build-arg GITHUB_TOKEN=$$GITHUB_TOKEN \
		--build-arg REPO_TAG=$$REPO_TAG \
		--build-arg REPO_URL=$$REPO_URL \
		--build-arg STAGE_ENV=staging \
		--build-arg ENV_FILE=.env_file/.env.staging; \
	docker-compose -p sanbou_staging -f docker/docker-compose.prod.yml down -v || echo 'down failed (ignored)'; \
	docker-compose -p sanbou_staging -f docker/docker-compose.prod.yml up -d


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


# 本番再ビルド（キャッシュ無効化）
prod_rebuild:
	@echo "Starting full production rebuild with --no-cache..."
	@echo "Reloading .env_file/.env.prod and rebuilding Docker image..."


	powershell -Command "$$envs = Get-Content .env_file/.env.prod | Where-Object { $$_ -match '^[^#].*=.*' } | ForEach-Object { $$kv = $$_ -split '=', 2; Set-Item -Path Env:$$($$kv[0].Trim()) -Value $$($$kv[1].Trim()) }; \
	docker-compose -p sanbou_prod -f docker/docker-compose.prod.yml build --no-cache \
		--build-arg GITHUB_TOKEN=$$env:GITHUB_TOKEN \
		--build-arg REPO_TAG=$$env:REPO_TAG \
		--build-arg REPO_URL=$$env:REPO_URL \
		--build-arg ENV_FILE=.env_file/.env.prod; \
	try { docker-compose -p sanbou_prod -f docker/docker-compose.prod.yml down -v } catch { Write-Host 'down failed (ignored)' }; \
	docker-compose -p sanbou_prod -f docker/docker-compose.prod.yml up -d"


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