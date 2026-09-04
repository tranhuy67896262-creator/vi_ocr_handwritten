@echo off
setlocal
cd /d "%~dp0"

echo ==========================================
echo  Vi-OCR-Handwritten - Train QLoRA (Windows)
echo ==========================================

REM Kiểm tra torch có GPU + torchvision (PyPI mặc định là bản CPU)
python -c "import torch, torchvision; assert torch.cuda.is_available()" >nul 2>&1
if errorlevel 1 (
    echo Dang cai torch + torchvision ban CUDA...
    python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
    if errorlevel 1 exit /b 1
)

REM Kiểm tra GPU
python -c "import torch; assert torch.cuda.is_available()" >nul 2>&1
if errorlevel 1 (
    echo [WARN] Khong thay GPU (torch dang ban CPU?).
    echo   - May local 6GB VRAM khong train duoc 7B.
    echo   - Train 7B that nen chay tren Colab H100 bang run_train.sh
    echo   - Muon train tren may nay: doi MODEL_NAME=Qwen/Qwen2.5-VL-3B-Instruct trong configs\configs.py
)

REM Cài deps còn thiếu
python -m pip show transformers >nul 2>&1
if errorlevel 1 (
    echo Dang cai dependencies...
    python -m pip install -r requirements.txt
    if errorlevel 1 exit /b 1
)

echo Bat dau train...
echo   Smoke test: %~nx0 --max-samples 100
echo   Train 7B  : chay tren Colab H100 (run_train.sh)
echo.

python scripts\train_qlora.py %*
set EXIT_CODE=%ERRORLEVEL%

echo.
echo Xong. Ket qua o: models\qwen25vl-7b-vi-hwr-lora\  (xem training_metadata.json + training.log)
exit /b %EXIT_CODE%