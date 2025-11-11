@echo off
REM Slope Stability Prediction - Docker Startup Script (Windows)

echo ==========================================
echo Slope Stability Prediction - Docker Setup
echo ==========================================
echo.

REM Check if Docker is installed
docker --version >nul 2>&1
if errorlevel 1 (
    echo X Docker is not installed!
    echo Please install Docker Desktop from: https://www.docker.com/products/docker-desktop
    pause
    exit /b 1
)

REM Check if Docker Compose is available
docker compose version >nul 2>&1
if errorlevel 1 (
    echo X Docker Compose is not available!
    echo Please update Docker Desktop to the latest version
    pause
    exit /b 1
)

echo [OK] Docker is installed
echo.

REM Check if models exist
if not exist "web-app\backend\models\best_model_gradient_boosting.pkl" (
    echo [!] Warning: Model files not found in web-app\backend\models\
    echo Please ensure the following files exist:
    echo   - best_model_gradient_boosting.pkl
    echo   - best_model_xgboost.pkl
    echo   - scaler.pkl
    echo.
    set /p continue="Continue anyway? (y/n): "
    if /i not "%continue%"=="y" exit /b 1
)

echo [*] Starting Slope Stability Prediction application...
echo.

REM Stop any existing containers
echo [*] Stopping existing containers (if any)...
docker compose down 2>nul

REM Build and start containers
echo [*] Building Docker images (this may take a few minutes on first run)...
docker compose up -d --build

REM Wait for containers to be healthy
echo.
echo [*] Waiting for services to be ready...
timeout /t 5 /nobreak >nul

REM Check status
echo.
echo [*] Container Status:
docker compose ps

REM Show logs
echo.
echo [*] Recent Logs:
docker compose logs --tail=20

echo.
echo ==========================================
echo [OK] Application is running!
echo ==========================================
echo.
echo [*] Access the application:
echo    Frontend: http://localhost:3000
echo    Backend API: http://localhost:5000
echo.
echo [*] Useful commands:
echo    View logs: docker compose logs -f
echo    Stop app: docker compose down
echo    Restart: docker compose restart
echo.
echo [*] For more info, see DOCKER_GUIDE.md
echo ==========================================
echo.
pause
