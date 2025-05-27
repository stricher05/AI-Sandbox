@echo off
REM ML Forecasting Platform Setup Script for Windows
REM This script sets up the complete environment

echo 🚀 Setting up ML Forecasting Platform...

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is not installed. Please install Python 3.8 or higher.
    pause
    exit /b 1
)

echo ✅ Python found

REM Create virtual environment
echo 📦 Creating virtual environment...
if not exist "venv" (
    python -m venv venv
)

REM Activate virtual environment
echo 🔥 Activating virtual environment...
call venv\Scripts\activate.bat

REM Upgrade pip
echo ⬆️ Upgrading pip...
python -m pip install --upgrade pip

REM Install requirements
echo 📋 Installing Python dependencies...
pip install -r requirements.txt

REM Create necessary directories
echo 📁 Creating directory structure...
if not exist "data" mkdir data
if not exist "models" mkdir models
if not exist "logs" mkdir logs

REM Generate sample data
echo 🔢 Generating sample data...
python -c "import sys; sys.path.append('src'); from data.preprocessing import generate_sample_data; generate_sample_data('data/sample_energy_data.csv'); print('✅ Sample data generated')"

REM Create batch scripts
echo 📝 Creating startup scripts...

REM Training script
echo @echo off > run_training.bat
echo echo 🚀 Starting ML training pipeline... >> run_training.bat
echo call venv\Scripts\activate.bat >> run_training.bat
echo set PYTHONPATH=%CD%\src;%PYTHONPATH% >> run_training.bat
echo python src\train_model.py --config config\default_config.yaml >> run_training.bat
echo echo ✅ Training completed! >> run_training.bat
echo pause >> run_training.bat

REM API startup script
echo @echo off > run_api.bat
echo echo 🚀 Starting ML forecasting API... >> run_api.bat
echo call venv\Scripts\activate.bat >> run_api.bat
echo set PYTHONPATH=%CD%\src;%PYTHONPATH% >> run_api.bat
echo python src\api\forecast_api.py >> run_api.bat
echo pause >> run_api.bat

echo.
echo ✅ Setup completed successfully! 🎉
echo.
echo 📋 Next steps:
echo   1. Run training: run_training.bat
echo   2. Start API: run_api.bat
echo   3. Or use Docker: docker-compose up
echo.
echo 🌟 Your production-ready ML forecasting platform is now ready!
echo.
pause
