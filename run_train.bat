@echo off
setlocal
cd /d "%~dp0"

echo ==========================================
echo  Vi-OCR-Handwritten - Train QLoRA (Windows)
echo ==========================================

REM Resolve absolute python: prefer .venv, else python on PATH.
REM Clear VIRTUAL_ENV so uv does not install into the shell's active venv.
set "VIRTUAL_ENV="
set "PY="
for /f "delims=" %%i in ('where python 2^>nul') do if not defined PY set "PY=%%i"
if not defined PY set "PY=python"
if exist ".venv\Scripts\python.exe" set "PY=.venv\Scripts\python.exe"

REM Install uv if missing
where uv >nul 2>&1
if errorlevel 1 (
    echo Installing uv...
    "%PY%" -m pip install uv
    if errorlevel 1 exit /b 1
)

"%PY%" --version

REM Check torch has CUDA + torchvision (PyPI default is CPU build)
"%PY%" -c "import torch, torchvision; assert torch.cuda.is_available()" >nul 2>&1
if errorlevel 1 (
    echo Installing torch + torchvision CUDA...
    uv pip install --python "%PY%" torch torchvision --index-url https://download.pytorch.org/whl/cu128
    if errorlevel 1 exit /b 1
)

REM Check GPU
"%PY%" -c "import torch; assert torch.cuda.is_available()" >nul 2>&1
if errorlevel 1 (
    echo [WARN] No GPU - torch is CPU build?
    echo   - Real training should run on Colab GPU via run_train.sh
    echo   - Model 7B needs 16GB+ VRAM; 3B needs ~8GB
)

REM Install missing deps
"%PY%" -c "import transformers, peft, bitsandbytes" >nul 2>&1
if errorlevel 1 (
    echo Installing dependencies...
    uv pip install --python "%PY%" -r requirements.txt
    if errorlevel 1 exit /b 1
)

echo Start training...
echo   Smoke test: %~nx0 --max-samples 100
echo   Real train: run on Colab GPU via run_train.sh
echo(
"%PY%" scripts\train_qlora.py %*
set EXIT_CODE=%ERRORLEVEL%

echo(
echo Done. Results at: models\qwen25vl-7b-vi-hwr-lora\  (see training_metadata.json + training.log)
exit /b %EXIT_CODE%