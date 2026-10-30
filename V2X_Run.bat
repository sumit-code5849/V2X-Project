@echo off
setlocal

REM Launch the V2X Streamlit demo using the existing trained models.
REM Keep this file beside V2X_All_In_One.py, data/, and models_realistic/.

cd /d "%~dp0"
title V2X Consolidated Project

echo ==============================================
echo       V2X CONSOLIDATED PROJECT - LAUNCHER
echo ==============================================
echo.

if not exist "V2X_All_In_One.py" (
    echo ERROR: V2X_All_In_One.py was not found.
    echo Keep V2X_Run.bat in the project root folder.
    echo.
    pause
    exit /b 1
)

if not exist "models_realistic\" (
    echo WARNING: models_realistic folder was not found.
    echo The application may not be able to load your trained models.
    echo.
)

where py >nul 2>nul
if %errorlevel%==0 (
    set "PYTHON=py"
) else (
    where python >nul 2>nul
    if errorlevel 1 (
        echo ERROR: Python was not found. Install Python and add it to PATH.
        echo.
        pause
        exit /b 1
    )
    set "PYTHON=python"
)

echo Starting Streamlit. Your existing model files will be loaded;
echo this launcher does not start model training.
echo.
%PYTHON% -m streamlit run "V2X_All_In_One.py"

if errorlevel 1 (
    echo.
    echo The application could not be started.
    echo Try installing dependencies with:
    echo   %PYTHON% -m pip install -r requirements.txt
    echo.
    pause
    exit /b 1
)

endlocal
