@echo off
setlocal
cd /d "%~dp0"

REM Portable qua may khac: uv copy thay vi hardlink (tat warning
REM "Failed to hardlink files" khi cache/project khac o dia).
if not defined UV_LINK_MODE set "UV_LINK_MODE=copy"

REM Cache HF trong project de de mang theo qua may khac (giong run_train.sh).
if not defined HF_HOME set "HF_HOME=%CD%\.hf_cache"
echo HF cache: %HF_HOME%

echo ==========================================
echo  Vi-OCR-Handwritten - Train QLoRA (Windows)
echo ==========================================

REM Nhan HF_TOKEN tu doi so dau tien (neu khong bat dau bang --), giong run_train.sh
set "A1=%~1"
if defined A1 if not "%A1:~0,2%"=="--" (
    > .env.dev echo HF_TOKEN = %A1%
    echo Da ghi HF_TOKEN vao .env.dev
    shift
)

REM Canh bao neu chua co token - giong run_train.sh
set "HAS_TOKEN="
if defined HF_TOKEN set "HAS_TOKEN=1"
if not exist .env.dev goto :tokenwarn
findstr /R /C:"^HF_TOKEN" .env.dev >nul 2>&1
if not errorlevel 1 set "HAS_TOKEN=1"
:tokenwarn
if defined HAS_TOKEN goto :hastoken
echo [WARN] Chua co HF_TOKEN. Truyen token lam doi so dau:
echo   %~nx0 hf_xxxxxxxx --train --max-samples 100
:hastoken

REM Mac dinh: chi cai toi thieu + mo UI Gradio (khong train, khong tai data).
REM --train = train that (cai full torch-CUDA + requirements).
REM --ui = giong mac dinh (giu de tuong thich cu).
set "DO_UI=1"

REM Thu gom cac doi so con lai (vi %*% khong doi sau shift).
REM Bo --ui / --eval[=N] ra khoi REST (co rieng cua wrapper, giong run_train.sh).
set "DO_EVAL="
set "EVAL_NUM=100"
set "REST="
:argloop
if "%~1"=="" goto :argsdone
if "%~1"=="--ui" goto :nextarg
if "%~1"=="--train" set "DO_UI="
if "%~1"=="--train" goto :nextarg
echo %~1 | findstr /R /C:"^--eval" >nul 2>&1
if errorlevel 1 goto :notflag
set "DO_EVAL=1"
shift
if "%~1"=="" goto :argloop
set "ISNUM=1"
for /f "delims=0123456789" %%c in ("%~1") do set "ISNUM="
if not defined ISNUM goto :argloop
set "EVAL_NUM=%~1"
goto :nextarg
:notflag
set "REST=%REST% %~1"
:nextarg
shift
goto :argloop
:argsdone

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
"%PY%" -c "import sys; assert sys.version_info>=(3,10)" >nul 2>&1
if errorlevel 1 echo [WARN] Can Python ^>= 3.10 - da test ky tren 3.13.x

REM Che do chi mo UI --ui: len UI ngay, KHONG tai torch-CUDA/requirements/data.
REM Chi can gradio + dotenv. Nut Train/OCR trong UI se bao loi neu thieu deps.
if defined DO_UI (
    echo Opening UI Gradio - light mode, chua tai torch/data...
    "%PY%" -c "import gradio, dotenv" >nul 2>&1
    if errorlevel 1 (
        uv pip install --python "%PY%" gradio python-dotenv
        if errorlevel 1 exit /b 1
    )
    "%PY%" scripts\ui.py
    exit /b 0
)

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
echo   Smoke test: %~nx0 --train --max-samples 100
echo   Real train: run on Colab GPU via run_train.sh --train
echo(

"%PY%" scripts\train_qlora.py %REST%
set EXIT_CODE=%ERRORLEVEL%
if not "%EXIT_CODE%"=="0" exit /b %EXIT_CODE%

REM Sao luu adapter + log - tuy chon, giong run_train.sh
REM vd: set BACKUP_DIR=D:\backup\vi_ocr_models
if defined BACKUP_DIR if exist models (
    if not exist "%BACKUP_DIR%" mkdir "%BACKUP_DIR%"
    xcopy models "%BACKUP_DIR%\" /E /I /Y >nul
    echo Da sao luu ket qua vao: %BACKUP_DIR%
)

REM --eval[=N]: chay eval CER/WER sau train (giong run_train.sh)
if defined DO_EVAL (
    echo ===== Eval CER/WER tren %EVAL_NUM% mau test =====
    "%PY%" scripts\eval_ocr.py --num-test "%EVAL_NUM%"
    set EXIT_CODE=%ERRORLEVEL%
)

echo(
echo Done. Results at: models\qwen25vl-7b-vi-hwr-lora\  (see training_metadata.json + training.log)
exit /b %EXIT_CODE%