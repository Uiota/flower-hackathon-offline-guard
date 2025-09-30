@echo off
echo 🚀 Starting Federated Learning Environment
echo ==========================================

REM Load environment
if exist .env (
    echo ✅ Environment file found
) else (
    echo ⚠️  No .env file found, using defaults
    set OFFLINE_MODE=1
)

REM Activate virtual environment
if exist .venv\Scripts\activate.bat (
    call .venv\Scripts\activate.bat
    echo ✅ Virtual environment activated
) else (
    echo ❌ Virtual environment not found
    echo Run: python setup_environment.py
    pause
    exit /b 1
)

REM Check dependencies
echo 🔍 Checking dependencies...
python -c "import sys; print(f'Python: {sys.version}')"

REM Start the dashboard
echo 🎯 Starting FL dashboard...
python dashboard_with_agents.py

pause
