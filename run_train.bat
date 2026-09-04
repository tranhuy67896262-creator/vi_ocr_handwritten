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

REM Chọn python: ưu tiên venv .venv (tạo bằng uv), không thì dùng python hệ thống
if exist ".venv\Scripts\python.exe" (
    set "PYTHON=.venv\Scripts\python.exe"
) else (
    set "PYTHON=python"
)
"%PYTHON%" --version

REM Kiểm tra torch có GPU + torchvision (PyPI mặc định là bản CPU)
"%PYTHON%" -c "import torch, torchvision; assert torch.cuda.is_available()" >nul 2>&1
if errorlevel 1 (
    echo Dang cai torch + torchvision ban CUDA...
    uv pip install --python "%PYTHON%" torch torchvision --index-url https://download.pytorch.org/whl/cu128
    if errorlevel 1 exit /b 1
)

REM Kiểm tra GPU
"%PYTHON%" -c "import torch; assert torch.cuda.is_available()" >nul 2>&1
if errorlevel 1 (
    echo [WARN] Khong thay GPU (torch dang ban CPU?).
    echo   - Train that nen chay tren Colab GPU (run_train.sh) hoac may co CUDA.
    echo   - Model 3B mac dinh can ~8GB VRAM; 7B can 16GB+.
)

REM Cài deps còn thiếu
"%PYTHON%" -c "import transformers, peft, bitsandbytes" >nul 2>&1
if errorlevel 1 (
    echo Dang cai dependencies...
    uv pip install --python "%PYTHON%" -r requirements.txt
    if errorlevel 1 exit /b 1
)

echo Bat dau train...
echo   Smoke test: %~nx0 --max-samples 100
echo   Train that : chay tren Colab GPU (run_train.sh)
echo.

"%PYTHON%" scripts\train_qlora.py %*
set EXIT_CODE=%ERRORLEVEL%

echo.
echo Xong. Ket qua o: models\qwen25vl-3b-vi-hwr-lora\  (xem training_metadata.json + training.log)
exit /b %EXIT_CODE%