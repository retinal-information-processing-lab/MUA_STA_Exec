@echo off
TITLE Analyse MUA STA
echo ---------------------------------------------------------
echo ACTIVATION DE L'ENVIRONNEMENT MUA_STA
echo ---------------------------------------------------------

:: Charge Conda (Remplacez par votre chemin si Conda n'est pas dans le PATH)
call conda activate MUA_STA

:: Vérifie si l'activation a fonctionné
if %errorlevel% neq 0 (
    echo Erreur : L'environnement MUA_STA n'a pas pu etre active.
    pause
    exit /b
)

echo LANCEMENT DU CODE PYTHON...
python "STA_MU_Exec.py"

echo ---------------------------------------------------------
echo ANALYSE THERMINEE.
pause