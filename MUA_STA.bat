@echo off
TITLE MUA_STA
cd /d "%~dp0"

call conda activate mua_sta
if %errorlevel% neq 0 (
    echo Environment 'mua_sta' not found: creating it from environment.yml ^(a few minutes^)...
    call conda env create -f environment.yml
    if %errorlevel% neq 0 (
        echo Environment creation failed.
        pause
        exit /b
    )
    call conda activate mua_sta
)

if not exist "%USERPROFILE%\.mua_sta" mkdir "%USERPROFILE%\.mua_sta"
start "" pythonw -m mua_sta
