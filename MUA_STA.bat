@echo off
title MUA_STA - multi-unit spike-triggered averages

:: 1. Find conda
set CONDA_PATH=%USERPROFILE%\anaconda3\Scripts\activate.bat
if not exist "%CONDA_PATH%" set CONDA_PATH=%USERPROFILE%\miniconda3\Scripts\activate.bat
if not exist "%CONDA_PATH%" set CONDA_PATH=%ProgramData%\anaconda3\Scripts\activate.bat
if not exist "%CONDA_PATH%" set CONDA_PATH=%ProgramData%\miniconda3\Scripts\activate.bat
if not exist "%CONDA_PATH%" (
    echo Conda not found. Install Anaconda or Miniconda, then run this again.
    pause
    exit /b 1
)

:: 2. This .bat's own folder ("%~dp0"), so the launcher works from anywhere
cd /d "%~dp0"

:: 3. Activate conda, creating the environment the first time
call "%CONDA_PATH%"
conda env list | findstr /B /C:"mua_sta " >nul
if errorlevel 1 (
    echo Environment 'mua_sta' not found: creating it from environment.yml ^(a few minutes^)...
    call conda env create -f environment.yml
    if errorlevel 1 (
        echo Creating the environment failed.
        pause
        exit /b 1
    )
)
call conda activate mua_sta
if errorlevel 1 (
    echo Could not activate the 'mua_sta' environment.
    pause
    exit /b 1
)

:: 4. Launch through the environment's own interpreter rather than whatever "pythonw" finds
::    on PATH: another environment's folder early on PATH would otherwise win silently.
::    pythonw = no console window.
if not exist "%CONDA_PREFIX%\pythonw.exe" (
    echo The 'mua_sta' environment has no pythonw.exe at "%CONDA_PREFIX%".
    pause
    exit /b 1
)
start "" "%CONDA_PREFIX%\pythonw.exe" -m mua_sta
exit
