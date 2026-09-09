"""UI Gradio: Fine-tune / OCR / Eval / Export / Settings."""
import os
import re
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import gradio as gr

from configs.configs import Configs

SCRIPT = PROJECT_ROOT / "scripts"

MODEL_CHOICES = [
    "Qwen/Qwen2.5-VL-3B-Instruct",
    "Qwen/Qwen2.5-VL-7B-Instruct",
]


def _adapter_for_model(model_name):
    from configs.configs import _adapter_tag
    return str(PROJECT_ROOT / "models" / f"qwen25vl-{_adapter_tag(model_name)}-vi-hwr-lora")


def sync_adapter(model_name, current):
    """Đổi base model -> tự trỏ adapter về thư mục mặc định tương ứng.
    Giữ nguyên nếu user đã gõ path/repo id riêng."""
    cur = (current or "").strip()
    known = {_adapter_for_model(m) for m in MODEL_CHOICES}
    if not cur or cur in known:
        return _adapter_for_model(model_name or MODEL_CHOICES[0])
    return cur


def _run(cmd, log=""):
    """Chạy 1 script con, stream output realtime vào log."""
    with subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        cwd=str(PROJECT_ROOT),
    ) as proc:
        for line in proc.stdout:
            log += line
            yield log
        proc.wait()
        yield log + f"\n[Thoát với mã: {proc.returncode}]"


# ---------------- Train ----------------

def train_ui(dataset, model, data_size, resume):
    """Chạy train_qlora.py với tham số từ UI, log realtime; xong thì xóa cache OCR."""
    _free_gpu()
    cmd = [sys.executable, str(SCRIPT / "train_qlora.py")]
    if dataset:
        cmd += ["--dataset", dataset]
    if model:
        cmd += ["--model", model]
    if data_size and int(data_size) > 0:
        cmd += ["--max-samples", str(int(data_size))]
    if resume:
        cmd += ["--resume"]
    yield from _run(cmd, "> " + " ".join(cmd) + "\n")
    # Train xong -> xoa cache model OCR để lần OCR sau load đúng adapter mới nhất.
    _OCR_CACHE.clear()


# ---------------- OCR ----------------

_OCR_CACHE = {}


def _free_gpu():
    """Nhả model OCR đang cache trước khi chạy tiến trình nặng (Train/Eval/Export).

    UI và subprocess (train/eval/export) là 2 process riêng nhưng chung VRAM —
    không nhả thì export/merge phải offload ra disk (chậm) hoặc OOM.
    """
    _OCR_CACHE.clear()
    try:
        import gc
        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _get_ocr(config, adapter, model):
    key = (model, adapter)
    if key not in _OCR_CACHE:
        from src.infer.predict import load_ocr_model
        _OCR_CACHE[key] = load_ocr_model(config, adapter, model)
    return _OCR_CACHE[key]


def _resolve_ocr(config, adapter, model):
    adapter = (adapter or "").strip() or str(config.ADAPTER_DIR)
    model = (model or "").strip() or config.MODEL_NAME
    return _get_ocr(config, adapter, model)


def ocr_ui(image, adapter, model):
    """OCR 1 ảnh PIL từ UI."""
    if image is None:
        return "Chưa có ảnh. Hãy upload 1 ảnh chữ viết tay."
    config = Configs()
    model_obj, processor = _resolve_ocr(config, adapter, model)
    from src.infer.predict import predict_image
    return predict_image(config, model_obj, processor, image)


def ocr_file_ui(file_path, adapter, model):
    """OCR file nhiều trang: PDF scan / Word (.docx) / ảnh lẻ."""
    if not file_path:
        return "Chưa có file. Hãy upload PDF, Word (.docx) hoặc ảnh."
    config = Configs()
    model_obj, processor = _resolve_ocr(config, adapter, model)
    try:
        suf = Path(file_path).suffix.lower()
        if suf == ".pdf":
            from src.infer.predict import ocr_pdf
            return ocr_pdf(config, model_obj, processor, file_path)
        if suf == ".docx":
            from src.infer.predict import ocr_docx
            return ocr_docx(config, model_obj, processor, file_path)
        from PIL import Image as _PIL
        from src.infer.predict import predict_image
        return predict_image(config, model_obj, processor, _PIL.open(file_path).convert("RGB"))
    except ImportError as exc:
        return str(exc)


# ---------------- Eval ----------------

def eval_ui(num_test, adapter, model):
    """Chạy eval_ocr.py, log realtime."""
    _free_gpu()
    cmd = [sys.executable, str(SCRIPT / "eval_ocr.py"), "--num-test", str(int(num_test))]
    if adapter and adapter.strip():
        cmd += ["--adapter", adapter.strip()]
    if model and model.strip():
        cmd += ["--model", model.strip()]
    yield from _run(cmd)


# ---------------- Export ----------------

def export_ui(adapter, model):
    """Chạy export_merged.py, log realtime."""
    _free_gpu()
    cmd = [sys.executable, str(SCRIPT / "export_merged.py")]
    if adapter and adapter.strip():
        cmd += ["--adapter", adapter.strip()]
    if model and model.strip():
        cmd += ["--model", model.strip()]
    yield from _run(cmd)


def _latest_gguf():
    """File .gguf mới nhất (ưu tiên bản Q4_K_M), hoặc None nếu chưa có."""
    files = sorted((PROJECT_ROOT / "models" / "gguf").glob("*.gguf"))
    if not files:
        return None
    q4 = [f for f in files if "Q4_K_M" in f.name]
    pick = q4[-1] if q4 else files[-1]
    return str(pick)


def _merge_is_fresh(merge_dir, adapter):
    """Merged còn dùng được nếu có safetensors và mới hơn adapter local.
    Adapter là repo Hub (không có local) thì luôn merge lại cho chắc."""
    m = Path(merge_dir)
    if not m.exists() or not list(m.glob("*.safetensors")):
        return False
    src = Path(adapter)
    if not src.exists():
        return False

    def _newest(p):
        fs = [f for f in p.rglob("*") if f.is_file()]
        return max((f.stat().st_mtime for f in fs), default=0)

    return _newest(m) >= _newest(src)


def export_gguf_ui(adapter, model):
    """Nút Download .gguf: bỏ qua merge nếu thư mục merged còn mới,
    rồi convert + quantize. Chỉ Linux/Colab.

    YIELD tuple 3 phần tử (log, download, status) — Gradio bắt lỗi nếu lệch.
    """
    _hidden = gr.DownloadButton(visible=False)
    if sys.platform == "win32":
        yield (("Export GGUF cần Linux/Colab (build llama.cpp) — "
                "không chạy trên Windows."), _hidden, "")
        return
    _free_gpu()
    config = Configs()
    adapter = (adapter or "").strip() or str(config.ADAPTER_DIR)
    model = (model or "").strip() or config.MODEL_NAME
    merge_dir = str(PROJECT_ROOT / "models" / f"{Path(adapter).name}-merged")
    log = ""
    if _merge_is_fresh(merge_dir, adapter):
        log = f"Dùng merged có sẵn (mới hơn adapter): {merge_dir}\n"
        yield log, _hidden, "⏳ Đang convert GGUF (xem Log)..."
    else:
        cmd1 = [sys.executable, str(SCRIPT / "export_merged.py"),
                "--adapter", adapter, "--model", model, "--output", merge_dir]
        for chunk in _run(cmd1, "> " + " ".join(cmd1) + "\n"):
            log = chunk
            yield log, _hidden, "⏳ Đang merge adapter (vài phút)..."
    cmd2 = ["bash", str(SCRIPT / "export_gguf.sh"), merge_dir]
    for chunk in _run(cmd2, log):
        log = chunk
        yield (log, _hidden,
               "⏳ Đang build/convert GGUF — lần đầu lâu (10–20 phút). "
               "Xong sẽ hiện nút tải file bên dưới.")
    gguf = _latest_gguf()
    if gguf:
        yield (log + f"\n✅ GGUF: {gguf}",
               gr.DownloadButton(value=gguf, visible=True,
                                 label=f"⬇ Tải {Path(gguf).name}"),
               "✅ Xong — bấm nút tải file bên dưới.")
    else:
        yield (log + "\n[WARN] Không thấy file .gguf — xem log convert.",
               _hidden, "⚠️ Thất bại — xem Log.")


def push_gguf_ui(hub_repo):
    """Push file .gguf mới nhất lên Hugging Face Hub để có link tải nhanh/ổn định."""
    hub_repo = (hub_repo or "").strip()
    if "/" not in hub_repo:
        return "Nhập repo id dạng `owner/repo` (vd `username/qwen25vl-3b-vi-hwr-gguf`)."
    config = Configs()
    if not config.HF_TOKEN:
        return "Chưa có HF_TOKEN — sang tab Settings lưu token trước."
    gguf = _latest_gguf()
    if not gguf:
        return "Chưa có file .gguf — bấm Download file .gguf để export trước."
    try:
        from huggingface_hub import HfApi
        api = HfApi(token=config.HF_TOKEN)
        api.create_repo(hub_repo, exist_ok=True, token=config.HF_TOKEN)
        name = Path(gguf).name
        api.upload_file(path_or_fileobj=gguf, path_in_repo=name,
                        repo_id=hub_repo, token=config.HF_TOKEN)
        return (f"✅ Đã push: https://huggingface.co/{hub_repo}/blob/main/{name}\n"
                f"Tải từ link đó (nhanh, hỗ trợ resume).")
    except Exception as exc:
        return _push_error(exc)


def push_adapter_ui(hub_repo, adapter):
    """Push thư mục adapter local lên Hugging Face Hub."""
    hub_repo = (hub_repo or "").strip()
    if "/" not in hub_repo:
        return "Nhập repo id dạng `owner/repo` (vd `username/qwen25vl-3b-vi-hwr-lora`)."
    config = Configs()
    if not config.HF_TOKEN:
        return "Chưa có HF_TOKEN — sang tab Settings lưu token trước."
    adapter = (adapter or "").strip() or str(config.ADAPTER_DIR)
    src = Path(adapter)
    if not src.exists():
        return (f"Adapter `{adapter}` không có local (repo Hub hoặc sai path) — "
                f"không có gì để push.")
    try:
        from huggingface_hub import HfApi
        api = HfApi(token=config.HF_TOKEN)
        api.create_repo(hub_repo, exist_ok=True, token=config.HF_TOKEN)
        api.upload_folder(folder_path=str(src), repo_id=hub_repo,
                          token=config.HF_TOKEN)
        return (f"✅ Đã push adapter: https://huggingface.co/{hub_repo}\n"
                f"Dùng trực tiếp ở ô Adapter bằng repo id `{hub_repo}`.")
    except Exception as exc:
        return _push_error(exc)


def _fmt_size(n):
    """Định dạng bytes thành chuỗi đọc được."""
    unit = "B"
    for u in ("KB", "MB", "GB"):
        if n < 1024:
            break
        n /= 1024
        unit = u
    return f"{n:.1f} {unit}"


def _dir_size(p):
    p = Path(p)
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


def _check_hf_cached(repo_id, token):
    """Repo HF đã cache đủ local chưa (không tải thêm). Trả về (ok, bytes)."""
    try:
        from huggingface_hub import snapshot_download
        p = Path(snapshot_download(repo_id, local_files_only=True,
                                   token=token or None))
    except Exception:
        return False, 0
    return True, _dir_size(p)


def check_model_ui(model, adapter):
    """Kiểm tra base model + adapter đã có trong máy chưa (không tải gì thêm)."""
    config = Configs()
    model = (model or "").strip() or config.MODEL_NAME
    adapter = (adapter or "").strip() or str(config.ADAPTER_DIR)
    lines = []
    ok, size = _check_hf_cached(model, config.HF_TOKEN)
    if ok:
        lines.append(f"✅ Base `{model}` — đã có ({_fmt_size(size)} trong cache).")
    else:
        lines.append(f"❌ Base `{model}` — chưa đủ trong máy, lần dùng đầu sẽ phải tải.")
    ap = Path(adapter)
    if ap.exists():
        lines.append(f"✅ Adapter local `{adapter}` ({_fmt_size(_dir_size(ap))}).")
    elif "/" in adapter:
        ok2, size2 = _check_hf_cached(adapter, config.HF_TOKEN)
        if ok2:
            lines.append(f"✅ Adapter Hub `{adapter}` — đã có ({_fmt_size(size2)} trong cache).")
        else:
            lines.append(f"❌ Adapter Hub `{adapter}` — chưa tải, lần dùng đầu sẽ phải tải.")
    else:
        lines.append(f"❌ Adapter `{adapter}` — không thấy local, cũng không phải repo id.")
    return "\n".join(lines)


# ---------------- Settings (HF token) ----------------

def _token_path():
    return PROJECT_ROOT / ".env.dev"


def token_status():
    """Chuỗi trạng thái token HF (che bớt giữa)."""
    path = _token_path()
    if path.exists():
        m = re.search(r"HF_TOKEN\s*=\s*(\S+)", path.read_text(encoding="utf-8"))
        if m and m.group(1):
            tok = m.group(1)
            return f"✅ Đã có token: `{tok[:6]}...{tok[-4:]}`"
    return "❌ Chưa có token. Nhập vào ô bên dưới rồi bấm Lưu."


def _token_username(token):
    """Tên tài khoản sở hữu token (None nếu không lấy được, vd offline)."""
    try:
        from huggingface_hub import HfApi
        info = HfApi(token=token).whoami()
        return info.get("name") if isinstance(info, dict) else getattr(info, "name", None)
    except Exception:
        return None


def _push_error(exc):
    s = str(exc)
    if "403" in s or "Forbidden" in s:
        return (f"[ERROR] Push thất bại (403 - không có quyền): {s}\n"
                f"Nguyên nhân thường gặp: repo owner khác tài khoản của token, "
                f"hoặc token loại Read (cần Write). Xem tên tài khoản ở tab Settings.")
    return f"[ERROR] Push thất bại: {exc}"


def save_token(token):
    """Lưu HF_TOKEN vào .env.dev, trả (message, status mới)."""
    tok = (token or "").strip()
    if not tok:
        return "Token rỗng — chưa lưu.", token_status()
    _token_path().write_text(f"HF_TOKEN = {tok}\n", encoding="utf-8")
    user = _token_username(tok)
    extra = (f" Tài khoản: `{user}` — push repo phải dùng owner là `{user}`."
             if user else "")
    return f"✅ Đã lưu token vào {_token_path()}.{extra}", token_status()


# ---------------- App ----------------

def build_app():
    """Dựng giao diện Gradio 5 tab."""
    cfg = Configs()
    with gr.Blocks(title="Vi-OCR-Handwritten UI") as demo:
        gr.Markdown(
            "# 🚀 Vi-OCR-Handwritten — QLoRA fine-tune Qwen2.5-VL\n"
            "Fine-tune / OCR / Eval / Export. Log hiển thị realtime.\n"
            "☁️ Dùng ké GPU Colab: "
            "[mở notebook Colab]"
            "(https://colab.research.google.com/notebook"
            "#fileId=https%3A//huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct.ipynb)"
        )

        with gr.Tab("Fine-tune"):
            gr.Markdown(
                "Dataset nguồn có **50k+ ảnh**. Chọn **Full** để train toàn bộ (lâu nhất), "
                "**35k / 10k** để train nhanh hơn."
            )
            with gr.Row():
                dataset = gr.Textbox(value=cfg.DATASET_NAME, label="Dataset")
                model = gr.Dropdown(
                    choices=MODEL_CHOICES,
                    value=cfg.MODEL_NAME, label="Model",
                    allow_custom_value=True,
                )
            data_size = gr.Radio(
                choices=[
                    ("Full (50k+ ảnh)", 0),
                    ("35k ảnh", 35000),
                    ("10k ảnh", 10000),
                    ("100 ảnh", 100),
                    ("10 ảnh (test pipeline)", 10),
                ],
                value=0, label="Cỡ data train",
            )
            resume_ckpt = gr.Checkbox(
                value=False,
                label="Tiếp tục từ checkpoint (tick khi chạy lại sau đứt giữa chừng)",
            )
            gr.Markdown(
                "🔗 Dataset: "
                "[5CD-AI/Viet-Handwriting-OCR-v2](https://huggingface.co/datasets/5CD-AI/Viet-Handwriting-OCR-v2) | "
                "Models: [Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) · "
                "[Qwen2.5-VL-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct)"
            )
            train_btn = gr.Button("▶ Fine-tune", variant="primary")
            train_log = gr.Textbox(label="Log", lines=20, max_lines=30, autoscroll=True, elem_classes=["log-scroll"])
            train_btn.click(
                train_ui,
                inputs=[dataset, model, data_size, resume_ckpt],
                outputs=train_log,
            )

        with gr.Tab("OCR 1 ảnh"):
            image = gr.Image(type="pil", image_mode="RGB", label="Ảnh chữ viết tay")
            with gr.Row():
                ocr_model = gr.Dropdown(choices=MODEL_CHOICES, value=cfg.MODEL_NAME,
                                        label="Base model", allow_custom_value=True)
                adapter_in = gr.Textbox(value=str(cfg.ADAPTER_DIR), label="Adapter (đường dẫn hoặc owner/repo)")
            ocr_model.change(sync_adapter, inputs=[ocr_model, adapter_in], outputs=adapter_in)
            ocr_btn = gr.Button("🔍 OCR", variant="primary")
            ocr_out = gr.Textbox(label="Kết quả")
            ocr_btn.click(ocr_ui, inputs=[image, adapter_in, ocr_model], outputs=ocr_out)

            gr.Markdown("### 📄 OCR file nhiều trang (PDF scan / Word .docx / ảnh)")
            pdf_in = gr.File(
                label="File PDF, Word (.docx) hoặc ảnh",
                file_types=[".pdf", ".docx", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"],
            )
            pdf_btn = gr.Button("🔍 OCR file", variant="primary")
            pdf_out = gr.Textbox(label="Kết quả (gộp theo trang/ảnh)", lines=20, max_lines=30,
                                 autoscroll=True, elem_classes=["log-scroll"])
            pdf_btn.click(ocr_file_ui, inputs=[pdf_in, adapter_in, ocr_model], outputs=pdf_out)

        with gr.Tab("Eval CER/WER"):
            gr.Markdown(
                "Dataset nguồn có **50k+ ảnh**. Chọn **100 mẫu** để test nhanh (vài phút), "
                "**10k / 35k** để kết quả chắc hơn, **Full** để đánh giá toàn bộ test split "
                "(rất lâu — eval OCR từng ảnh). "
                "**CER/WER càng thấp càng tốt.**"
            )
            with gr.Row():
                num_test = gr.Radio(
                    choices=[
                        ("100 mẫu (nhanh)", 100),
                        ("10k mẫu", 10000),
                        ("35k mẫu", 35000),
                        ("Full test split (rất lâu!)", 1000000000),
                    ],
                    value=100, label="Số mẫu test",
                )
                eval_adapter = gr.Textbox(value=str(cfg.ADAPTER_DIR), label="Adapter")
                eval_model = gr.Dropdown(choices=MODEL_CHOICES, value=cfg.MODEL_NAME,
                                         label="Base model", allow_custom_value=True)
            eval_model.change(sync_adapter, inputs=[eval_model, eval_adapter], outputs=eval_adapter)
            eval_btn = gr.Button("📊 Eval", variant="primary")
            eval_log = gr.Textbox(label="Log", lines=20, max_lines=30, autoscroll=True, elem_classes=["log-scroll"])
            eval_btn.click(eval_ui, inputs=[num_test, eval_adapter, eval_model], outputs=eval_log)

        with gr.Tab("Export"):
            export_adapter = gr.Textbox(value=str(cfg.ADAPTER_DIR), label="Adapter")
            export_model = gr.Dropdown(choices=MODEL_CHOICES, value=cfg.MODEL_NAME,
                                       label="Base model", allow_custom_value=True)
            export_model.change(sync_adapter, inputs=[export_model, export_adapter], outputs=export_adapter)
            export_btn = gr.Button("📦 Export ra thư mục (full model)", variant="primary")
            export_log = gr.Textbox(label="Log", lines=20, max_lines=30, autoscroll=True, elem_classes=["log-scroll"])
            export_btn.click(export_ui, inputs=[export_adapter, export_model], outputs=export_log)

            gguf_btn = gr.Button("⬇ Download file .gguf", variant="primary")
            gguf_status = gr.Markdown()
            _gguf0 = _latest_gguf()
            dl_gguf = gr.DownloadButton(
                f"⬇ Tải {Path(_gguf0).name}" if _gguf0 else "⬇ Tải file GGUF",
                value=_gguf0, visible=bool(_gguf0))
            gguf_btn.click(export_gguf_ui, inputs=[export_adapter, export_model],
                           outputs=[export_log, dl_gguf, gguf_status])

            gr.Markdown("### ⬆ Push GGUF lên Hub (link tải nhanh, ổn định, vĩnh viễn)")
            hub_repo_in = gr.Textbox(label="Repo Hub (owner/repo)",
                                     placeholder="username/qwen25vl-3b-vi-hwr-gguf")
            push_btn = gr.Button("⬆ Push file .gguf lên Hub", variant="secondary")
            push_msg = gr.Markdown()
            push_btn.click(push_gguf_ui, inputs=[hub_repo_in], outputs=push_msg)

            push_ad_btn = gr.Button("⬆ Push adapter lên Hub", variant="secondary")
            push_ad_msg = gr.Markdown()
            push_ad_btn.click(push_adapter_ui, inputs=[hub_repo_in, export_adapter],
                              outputs=push_ad_msg)

        with gr.Tab("Settings"):
            tok_status = gr.Markdown(value=token_status())
            token_in = gr.Textbox(label="HF_TOKEN", type="password",
                                  placeholder="hf_xxxx... (máy Colab: thêm qua biểu tượng key 🔑 hoặc dán vào đây)")
            save_btn = gr.Button("💾 Lưu token vào .env.dev", variant="primary")
            tok_msg = gr.Markdown()
            save_btn.click(save_token, inputs=[token_in], outputs=[tok_msg, tok_status])
            gr.Markdown(
                "Ghi chú: token được lưu vào `.env.dev` (git-ignored). "
                "Mọi nút Fine-tune/OCR/Eval/Export đều đọc token này khi chạy."
            )

            gr.Markdown("### 💾 Kiểm tra model/adapter đã tải về máy chưa (không tải thêm)")
            check_model = gr.Dropdown(choices=MODEL_CHOICES, value=cfg.MODEL_NAME,
                                      label="Base model", allow_custom_value=True)
            check_adapter = gr.Textbox(value=str(cfg.ADAPTER_DIR), label="Adapter")
            check_btn = gr.Button("🔍 Kiểm tra", variant="secondary")
            check_msg = gr.Markdown()
            check_btn.click(check_model_ui, inputs=[check_model, check_adapter],
                            outputs=check_msg)

    return demo


if __name__ == "__main__":
    # GRADIO_SHARE=0 khi chay offline (khong tao link public). Mac dinh 1.
    share = os.getenv("GRADIO_SHARE", "1") == "1"
    build_app().launch(
        server_name="0.0.0.0",
        share=share,
        css=".log-scroll textarea { max-height: 500px !important; overflow-y: auto !important; }",
    )
