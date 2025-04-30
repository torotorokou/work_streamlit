# --- 共通設定 ---
ENV_FILE=.env
PROJECT_NAME=sanbou_prod
COMPOSE_FILE=docker/docker-compose.prod.yml
DOCKERFILE=docker/prod.Dockerfile
IMAGE_NAME=sanboukun:prod

# --- 環境起動 ---

dev:
	docker-compose -p sanbou_dev -f docker/docker-compose.dev.yml up

staging:
	docker-compose -p sanbou_staging -f docker/docker-compose.staging.yml up

prod:
	@echo "Starting production environment rebuild..."
	@echo "Loading .env with PowerShell..."

	powershell -Command "Get-Content .env | Where-Object { $$_ -match '^[^#].*=.*' } | ForEach-Object { $$parts = $$_ -split '=', 2; Set-Item -Path Env:$$($$parts[0].Trim()) -Value $$($$parts[1].Trim()) }; docker-compose -p sanbou_prod -f docker/docker-compose.prod.yml build --build-arg GITHUB_TOKEN=$$env:GITHUB_TOKEN --build-arg REPO_TAG=$$env:REPO_TAG --build-arg REPO_URL=$$env:REPO_URL"

	docker-compose -p sanbou_prod -f docker/docker-compose.prod.yml down || true
	docker-compose -p sanbou_prod -f docker/docker-compose.prod.yml up -d


# --- 停止・削除 ---

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
