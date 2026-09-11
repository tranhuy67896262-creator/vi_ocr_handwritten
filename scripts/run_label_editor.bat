@echo off
rem Mo OCR Label Editor (HF + anh ca nhan) tren http://127.0.0.1:9000
cd /d "%~dp0.."
set "PY=python"
if exist ".venv\Scripts\python.exe" set "PY=.venv\Scripts\python.exe"
"%PY%" -m scripts.labeling serve --port 9000 %*
