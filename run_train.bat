@echo off
setlocal
cd /d "%~dp0"

echo ==========================================
echo  Vi-OCR-Handwritten - Train QLoRA (Windows)
echo ==========================================

REM Cài uv nếu chưa có (thay pip/venv — nhanh hơn nhiều)
where uv >nul 2>&1
if errorlevel 1 (
    echo Dang cai uv...
    python -m pip install uv
    if errorlevel 1 exit /b 1
)

REM Chọn python TUYỆT ĐỐI: ưu tiên .venv, không thì python trên PATH.
REM Xoá VIRTUAL_ENV để uv không cài nhầm vào venv đang active của shell.
REM KHÔNG dùng goto trong block ( ) — cmd bị lỗi ". was unexpected at this time".
set "VIRTUAL_ENV="
set "PY="
for /f "delims=" %%i in ('where python 2^>nul') do if not defined PY set "PY=%%i"
if not defined PY set "PY=python"
if exist ".venv\Scripts\python.exe" set "PY=.venv\Scripts\python.exe"
"%PY%" --version

REM Kiểm tra torch có GPU + torchvision (PyPI mặc định là bản CPU)
"%PY%" -c "import torch, torchvision; assert torch.cuda.is_available()" >nul 2>&1
if errorlevel 1 (
    echo Dang cai torch + torchvision ban CUDA...
    uv pip install --python "%PY%" torch torchvision --index-url https://download.pytorch.org/whl/cu128
    if errorlevel 1 exit /b 1
)

REM Kiểm tra GPU
"%PY%" -c "import torch; assert torch.cuda.is_available()" >nul 2>&1
if errorlevel 1 (
    echo [WARN] Khong thay GPU (torch dang ban CPU?).
    echo   - Train that nen chay tren Colab GPU (run_train.sh).
    echo   - Model 7B mac dinh can 16GB+ VRAM; 3B can ~8GB.
)

REM Cài deps còn thiếu
"%PY%" -c "import transformers, peft, bitsandbytes" >nul 2>&1
if errorlevel 1 (
    echo Dang cai dependencies...
    uv pip install --python "%PY%" -r requirements.txt
    if errorlevel 1 exit /b 1
)

echo Bat dau train...
echo   Smoke test: %~nx0 --max-samples 100
echo   Train that : chay tren Colab GPU (run_train.sh)
echo.
"%PY%" scripts\train_qlora.py %*
set EXIT_CODE=%ERRORLEVEL%

echo.
echo Xong. Ket qua o: models\qwen25vl-7b-vi-hwr-lora\  (xem training_metadata.json + training.log)
exit /b %EXIT_CODE%