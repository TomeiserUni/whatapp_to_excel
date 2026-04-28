@echo off
cd /d "%~dp0.."
if exist "windows\python\python.exe" (
    windows\python\python.exe src\app.py
) else (
    echo ERRO: Python nao instalado. Corre primeiro windows\instalar.ps1
    pause
)
