@echo off
title Analyse Pannes - Serveur
echo ============================================================
echo   Demarrage de l'application Analyse des Pannes...
echo ============================================================
echo.
echo   Acces depuis ce PC     : http://localhost:8501
echo   Acces depuis le reseau : http://192.168.13.109:8501
echo.
echo   Laissez cette fenetre ouverte pendant l'utilisation.
echo   Fermez-la pour arreter le serveur.
echo ============================================================

cd /d "%~dp0"
".venv\Scripts\python.exe" -m streamlit run app.py ^
    --server.address 0.0.0.0 ^
    --server.port 8501 ^
    --server.headless true ^
    --browser.gatherUsageStats false

pause
