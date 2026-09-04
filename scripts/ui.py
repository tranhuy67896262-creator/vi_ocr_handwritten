import re
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import gradio as gr

from configs.configs import Configs

SCRIPT = PROJECT_ROOT / "scripts"


def _run(cmd, log=""):
    """Chạy 1 script con, stream output realtime vào log."""
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        cwd=str(PROJECT_ROOT),
    )
    for line in proc.stdout:
        log += line
        yield log
    proc.wait()
    yield log + f"\n[Thoát với mã: {proc.returncode}]"


# ---------------- Train ----------------

def train_ui(dataset, model, max_samples, epochs, lr, lora_r, lora_alpha,
             batch_size, max_seq_len, no_kl, push, hub_repo):
    cmd = [sys.executable, str(SCRIPT / "train_qlora.py")]
    if dataset:
        cmd += ["--dataset", dataset]
    if model:
        cmd += ["--model", model]
    if max_samples:
        cmd += ["--max-samples", str(int(max_samples))]
    if epochs:
        cmd += ["--epochs", str(int(epochs))]
    if lr:
        cmd += ["--lr", str(lr)]
    if lora_r:
        cmd += ["--lora-r", str(int(lora_r))]
    if lora_alpha:
        cmd += ["--lora-alpha", str(int(lora_alpha))]
    if batch_size:
        cmd += ["--batch-size", str(int(batch_size))]
    if max_seq_len:
        cmd += ["--max-seq-len", str(int(max_seq_len))]
    if no_kl:
        cmd += ["--no-kl"]
    if push:
        cmd += ["--push"]
    if hub_repo:
        cmd += ["--hub-repo", hub_repo]
    yield from _run(cmd, "> " + " ".join(cmd) + "\n")


# ---------------- OCR ----------------

_OCR_CACHE = {}


def _get_ocr(config, adapter):
    key = (config.MODEL_NAME, adapter)
    if key not in _OCR_CACHE:
        from src.infer.predict import load_ocr_model
        _OCR_CACHE[key] = load_ocr_model(config, adapter)
    return _OCR_CACHE[key]


def ocr_ui(image, adapter):
    if image is None:
        return "Chưa có ảnh. Hãy upload 1 ảnh chữ viết tay."
    config = Configs()
    adapter = adapter.strip() or str(config.ADAPTER_DIR)
    model, processor = _get_ocr(config, adapter)
    from src.infer.predict import predict_image
    return predict_image(config, model, processor, image)


# ---------------- Eval ----------------

def eval_ui(num_test, adapter):
    cmd = [sys.executable, str(SCRIPT / "eval_ocr.py"), "--num-test", str(int(num_test))]
    if adapter and adapter.strip():
        cmd += ["--adapter", adapter.strip()]
    yield from _run(cmd)


# ---------------- Export ----------------

def export_ui(adapter):
    cmd = [sys.executable, str(SCRIPT / "export_merged.py")]
    if adapter and adapter.strip():
        cmd += ["--adapter", adapter.strip()]
    yield from _run(cmd)


# ---------------- Settings (HF token) ----------------

def _token_path():
    return PROJECT_ROOT / ".env.dev"


def token_status():
    path = _token_path()
    if path.exists():
        m = re.search(r"HF_TOKEN\s*=\s*(\S+)", path.read_text(encoding="utf-8"))
        if m and m.group(1):
            tok = m.group(1)
            return f"✅ Đã có token: `{tok[:6]}...{tok[-4:]}`"
    return "❌ Chưa có token. Nhập vào ô bên dưới rồi bấm Lưu."


def save_token(token):
    tok = (token or "").strip()
    if not tok:
        return "Token rỗng — chưa lưu."
    _token_path().write_text(f"HF_TOKEN = {tok}\n", encoding="utf-8")
    return f"✅ Đã lưu token vào {_token_path()}"


# ---------------- App ----------------

def build_app():
    cfg = Configs()
    with gr.Blocks(title="Vi-OCR-Handwritten UI") as demo:
        gr.Markdown(
            "# 🚀 Vi-OCR-Handwritten — QLoRA fine-tune Qwen2.5-VL\n"
            "Train / OCR / Eval / Export. Log hiển thị realtime."
        )

        with gr.Tab("Train"):
            with gr.Row():
                dataset = gr.Textbox(value=cfg.DATASET_NAME, label="Dataset")
                model = gr.Textbox(value=cfg.MODEL_NAME, label="Model")
            with gr.Row():
                max_samples = gr.Number(value=1000, precision=0, label="max-samples (0 = full)")
                epochs = gr.Number(value=cfg.NUM_EPOCHS, precision=0, label="epochs")
                lr = gr.Number(value=cfg.LEARNING_RATE, label="learning rate")
            with gr.Row():
                lora_r = gr.Number(value=cfg.LORA_R, precision=0, label="lora-r")
                lora_alpha = gr.Number(value=cfg.LORA_ALPHA, precision=0, label="lora-alpha")
                batch_size = gr.Number(value=cfg.BATCH_SIZE, precision=0, label="batch-size")
                max_seq_len = gr.Number(value=cfg.MAX_SEQ_LEN, precision=0, label="max-seq-len")
            with gr.Row():
                no_kl = gr.Checkbox(value=not cfg.KL_REGULARIZATION, label="--no-kl (tắt KL, tiết kiệm VRAM)")
                push = gr.Checkbox(value=False, label="--push lên Hub")
                hub_repo = gr.Textbox(label="--hub-repo (owner/repo)")
            train_btn = gr.Button("▶ Train", variant="primary")
            train_log = gr.Textbox(label="Log", lines=20, max_lines=100)
            train_btn.click(
                train_ui,
                inputs=[dataset, model, max_samples, epochs, lr, lora_r, lora_alpha,
                        batch_size, max_seq_len, no_kl, push, hub_repo],
                outputs=train_log,
            )

        with gr.Tab("OCR 1 ảnh"):
            image = gr.Image(type="pil", image_mode="RGB", label="Ảnh chữ viết tay")
            adapter_in = gr.Textbox(value=str(cfg.ADAPTER_DIR), label="Adapter (đường dẫn hoặc owner/repo)")
            ocr_btn = gr.Button("🔍 OCR", variant="primary")
            ocr_out = gr.Textbox(label="Kết quả")
            ocr_btn.click(ocr_ui, inputs=[image, adapter_in], outputs=ocr_out)

        with gr.Tab("Eval CER/WER"):
            with gr.Row():
                num_test = gr.Number(value=100, precision=0, label="Số mẫu test")
                eval_adapter = gr.Textbox(value=str(cfg.ADAPTER_DIR), label="Adapter")
            eval_btn = gr.Button("📊 Eval", variant="primary")
            eval_log = gr.Textbox(label="Log", lines=20, max_lines=100)
            eval_btn.click(eval_ui, inputs=[num_test, eval_adapter], outputs=eval_log)

        with gr.Tab("Export"):
            export_adapter = gr.Textbox(value=str(cfg.ADAPTER_DIR), label="Adapter")
            export_btn = gr.Button("📦 Export full model", variant="primary")
            export_log = gr.Textbox(label="Log", lines=20, max_lines=100)
            export_btn.click(export_ui, inputs=[export_adapter], outputs=export_log)

        with gr.Tab("Settings"):
            tok_status = gr.Markdown(value=token_status())
            token_in = gr.Textbox(label="HF_TOKEN", type="password",
                                  placeholder="hf_xxxx... (máy Colab: thêm qua biểu tượng key 🔑 hoặc dán vào đây)")
            save_btn = gr.Button("💾 Lưu token vào .env.dev", variant="primary")
            tok_msg = gr.Markdown()
            save_btn.click(save_token, inputs=[token_in], outputs=[tok_msg, tok_status])
            gr.Markdown(
                "Ghi chú: token được lưu vào `.env.dev` (git-ignored). "
                "Mọi nút Train/OCR/Eval/Export đều đọc token này khi chạy."
            )

    return demo


if __name__ == "__main__":
    build_app().launch(server_name="0.0.0.0", share=True)