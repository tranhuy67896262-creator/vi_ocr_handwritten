"""UI Gradio: Fine-tune / OCR / Eval / Export / Settings."""
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import gradio as gr

from configs.configs import Configs

SCRIPT = PROJECT_ROOT / "scripts"

MODEL_CHOICES = [
    "tranhuy67896262/Qwen2.5-VL-3B-Instruct-private",
    "tranhuy67896262/Qwen2.5-VL-7B-Instruct-private",
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


def _run(cmd, log="", cwd=None, env=None):
    """Chạy 1 script con, stream output realtime vào log (cwd mặc định project).

    ``env``: dict biến môi trường cộng thêm (vd USE_4BIT/BATCH_SIZE cho stage_train.sh).
    """
    merged_env = dict(os.environ)
    if env:
        merged_env.update({k: str(v) for k, v in env.items() if v is not None})
    with subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        cwd=str(cwd or PROJECT_ROOT),
        env=merged_env,
    ) as proc:
        for line in proc.stdout:
            log += line
            yield log
        proc.wait()
        yield log + f"\n[Thoát với mã: {proc.returncode}]"


STAGE_MODELS = ["qwenvl-3b", "qwenvl-7b"]

STAGE_DATASETS = [
    "tranhuy67896262/Viet-Handwriting-OCR-v2-local",
    "tranhuy67896262/vietnamese-ocr-dataset-line-local",
    "tranhuy67896262/vietnamese-ocr-dataset-crop-card",
]

# Batch preset theo model: (USE_4BIT, BATCH_SIZE, GRAD_ACCUM); None = để script tự chọn.
BATCH_PRESETS = {
    "auto": {},
    "fast": {"qwenvl-3b": ("0", "16", "2"), "qwenvl-7b": ("0", "8", "4")},
    "saver": {"qwenvl-3b": ("1", "2", "8"), "qwenvl-7b": ("0", "2", "8")},
}


def _stage_model_full(short):
    """qwenvl-3b -> tên model HF đầy đủ (để suy repo mặc định + kiểm tra cache)."""
    s = (short or "").strip().lower()
    if "3b" in s:
        return MODEL_CHOICES[0]
    if "7b" in s:
        return MODEL_CHOICES[1]
    return s or MODEL_CHOICES[0]


def _hub_user():
    """User của token trong .env.dev (None nếu chưa có/không đọc được)."""
    config = Configs()
    if not config.HF_TOKEN:
        return None
    return _token_username(config.HF_TOKEN)


def _resolve_hub_repo(repo):
    """Repo trần (không owner) -> ghép user của token; trả (repo_full, cảnh báo)."""
    repo = (repo or "").strip()
    if not repo:
        return "", "Chưa nhập repo (dạng `owner/repo`)."
    if "/" in repo:
        return repo, ""
    user = _hub_user()
    if not user:
        return repo, "Repo thiếu owner mà chưa có HF_TOKEN — sang tab Settings lưu token trước."
    return f"{user}/{repo}", ""


def _list_stage_branches(repo):
    """Nhánh stage-* của repo Hub, sắp xếp theo K tăng dần (cần HF_TOKEN)."""
    config = Configs()
    if not config.HF_TOKEN:
        return None, "Chưa có HF_TOKEN — sang tab Settings lưu token trước."
    try:
        from huggingface_hub import HfApi
        refs = HfApi(token=config.HF_TOKEN).list_repo_refs(repo_id=repo)
    except Exception as exc:
        return None, f"Không đọc được repo `{repo}`: {exc}"
    found = []
    for br in refs.branches:
        name = br.name
        if not name.startswith("stage-"):
            continue
        core = name[len("stage-"):]
        knum = core.split("k")[0] if "k" in core else ""
        if knum.isdigit():
            found.append((int(knum), name))
    found.sort()
    return found, ""


def stage_progress_ui(repo, model):
    """Nút kiểm tra: repo đã train tới mốc nào (không train gì cả)."""
    repo_full, warn = _resolve_hub_repo(repo)
    if not repo_full or "/" not in repo_full:
        return warn
    found, err = _list_stage_branches(repo_full)
    if found is None:
        return err
    lines = [f"Repo `{repo_full}` (model `{model or 'qwenvl-3b'}`):"]
    if not found:
        lines.append("— Chưa có mốc stage nào → lần chạy tới sẽ **train trắng từ mẫu 0**.")
    else:
        lines.append("— Các mốc đã có: " + ", ".join(f"`{n}`" for _, n in found))
        top_k, top_name = found[-1]
        lines.append(f"— Lần chạy tới sẽ **train tiếp từ mẫu {top_k * 1000}** "
                     f"với adapter `{repo_full}@{top_name}`.")
    if warn:
        lines.append(f"⚠️ {warn}")
    return "\n".join(lines)


def _stage_env(model, preset):
    """Env batch/precision cho stage_train.sh theo preset đã chọn."""
    label = (preset or "")
    if label.startswith("Nhanh"):
        key = "fast"
    elif label.startswith("Ti"):
        key = "saver"
    else:
        key = "auto"
    if key == "auto":
        return {}
    short = (model or "").strip().lower()
    tag = "qwenvl-7b" if "7b" in short else "qwenvl-3b"
    use_4bit, batch, accum = BATCH_PRESETS[key][tag]
    return {"USE_4BIT": use_4bit, "BATCH_SIZE": batch, "GRAD_ACCUM": accum}


def _stage_extra_env(dataset, force_start, epochs):
    """Env thêm cho stage_train.sh: đổi dataset / ép start / số epoch mỗi lát."""
    env = {}
    if dataset and dataset.strip():
        env["DATASET"] = dataset.strip()
    if force_start is not None and str(force_start).strip() != "":
        env["FORCE_START"] = str(force_start).strip()
    try:
        ep = int(epochs) if epochs is not None else 1
    except (TypeError, ValueError):
        ep = 1
    if ep != 1:
        env["EPOCHS"] = str(ep)
    return env


def stage_dryrun_ui(repo, model, count, save_steps, preset, dataset, force_start, epochs):
    """Chạy stage_train.sh với DRY_RUN=1: xem kế hoạch start/init/rev, không train."""
    repo_full, warn = _resolve_hub_repo(repo)
    if not repo_full or "/" not in repo_full:
        yield warn
        return
    _free_gpu()
    cmd = ["bash", str(SCRIPT / "stage_train.sh"),
           (model or "qwenvl-3b").strip(), repo_full,
           (count or "5000").strip(), str(int(save_steps or 50))]
    env = {"DRY_RUN": "1"}
    env.update(_stage_env(model, preset))
    env.update(_stage_extra_env(dataset, force_start, epochs))
    if warn:
        yield warn + "\n"
    yield from _run(cmd, "> " + " ".join(cmd) + "\n", env=env)


def stage_train_ui(repo, model, count, save_steps, preset, dataset, force_start, epochs):
    """Chạy stage_train.sh thật: 1 lát, nền nohup, log vào train-<run>.log."""
    repo_full, warn = _resolve_hub_repo(repo)
    if not repo_full or "/" not in repo_full:
        yield warn
        return
    _free_gpu()
    cmd = ["bash", str(SCRIPT / "stage_train.sh"),
           (model or "qwenvl-3b").strip(), repo_full,
           (count or "5000").strip(), str(int(save_steps or 50))]
    env = _stage_env(model, preset)
    env.update(_stage_extra_env(dataset, force_start, epochs))
    if warn:
        yield warn + "\n"
    yield from _run(cmd, "> " + " ".join(cmd) + "\n", env=env)
    # Train chạy nền xong (hoặc đứt) -> xóa cache để OCR sau load adapter mới nhất.
    _OCR_CACHE.clear()
    yield ("Xem tiến độ bằng `tail -n 5 train-<run>.log` (không `tail -f`). "
           "Đứt giữa chừng: bấm lại nút này (tự resume checkpoint + nối nhánh Hub).")


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

def eval_ui(num_test, adapter, model, revision):
    """Chạy eval_ocr.py, log realtime (revision: eval đúng mốc staged, vd stage-10k)."""
    _free_gpu()
    cmd = [sys.executable, str(SCRIPT / "eval_ocr.py"), "--num-test", str(int(num_test))]
    if adapter and adapter.strip():
        cmd += ["--adapter", adapter.strip()]
    if model and model.strip():
        cmd += ["--model", model.strip()]
    if revision and revision.strip():
        cmd += ["--adapter-revision", revision.strip()]
    yield from _run(cmd)


# ---------------- Export ----------------

def export_ui(adapter, model, revision):
    """Chạy export_merged.py, log realtime (revision: merge đúng mốc staged)."""
    _free_gpu()
    cmd = [sys.executable, str(SCRIPT / "export_merged.py")]
    if adapter and adapter.strip():
        cmd += ["--adapter", adapter.strip()]
    if model and model.strip():
        cmd += ["--model", model.strip()]
    if revision and revision.strip():
        cmd += ["--adapter-revision", revision.strip()]
    yield from _run(cmd)


def _latest_gguf():
    """File .gguf mới nhất (ưu tiên bản Q6_K), hoặc None nếu chưa có."""
    files = sorted((PROJECT_ROOT / "models" / "gguf").glob("*.gguf"))
    if not files:
        return None
    q6 = [f for f in files if "Q6_K" in f.name]
    pick = q6[-1] if q6 else files[-1]
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


def export_gguf_ui(adapter, model, revision):
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
    revision = (revision or "").strip()
    merge_dir = str(PROJECT_ROOT / "models" / f"{Path(adapter).name}-merged")
    log = ""
    if _merge_is_fresh(merge_dir, adapter):
        log = f"Dùng merged có sẵn (mới hơn adapter): {merge_dir}\n"
        yield log, _hidden, "⏳ Đang convert GGUF (xem Log)..."
    else:
        cmd1 = [sys.executable, str(SCRIPT / "export_merged.py"),
                "--adapter", adapter, "--model", model, "--output", merge_dir]
        if revision:
            cmd1 += ["--adapter-revision", revision]
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


def _mmproj_sibling(gguf_path):
    """File mmproj cùng thư mục với gguf (nếu có) — Ollama/llama.cpp cần để OCR ảnh."""
    cands = sorted(Path(gguf_path).parent.glob("*mmproj*.gguf"))
    picked = [c for c in cands if str(c) != str(gguf_path)]
    return str(picked[0]) if picked else None


def _default_ollama_name():
    """Tên model Ollama gợi ý từ file gguf mới nhất (bỏ hậu tố quant)."""
    gguf = _latest_gguf()
    if not gguf:
        return "qwen25vl-3b-vi-hwr"
    stem = Path(gguf).stem
    for suf in ("-Q6_K", "-Q4_K_M", "-Q4_0", "-Q8_0", "-f16"):
        if stem.endswith(suf):
            stem = stem[: -len(suf)]
            break
    return stem


def import_ollama_ui(model_name):
    """Import gguf + Modelfile vào Ollama đang chạy (`ollama create`)."""
    name = (model_name or "").strip() or _default_ollama_name()
    gguf = _latest_gguf()
    if not gguf:
        msg = "Chưa có file .gguf — bấm Download file .gguf để export trước."
        yield msg, msg
        return
    if shutil.which("ollama") is None:
        msg = ("Chưa có Ollama CLI — cài tại https://ollama.com/download "
               "(Colab: curl -fsSL https://ollama.com/install.sh | sh).")
        yield msg, msg
        return
    gguf_dir = str(Path(gguf).parent)
    if not Path(gguf_dir, "Modelfile").exists():
        msg = f"Thiếu {gguf_dir}/Modelfile — bấm Download file .gguf để sinh lại."
        yield msg, msg
        return
    try:
        r = subprocess.run(["ollama", "list"], capture_output=True, timeout=30, check=False)
        server_ok = r.returncode == 0
    except Exception:
        server_ok = False
    if not server_ok:
        msg = "Ollama server chưa chạy — chạy `ollama serve` trước (Colab chạy nền). "
        yield msg, msg
        return
    log = ""
    for chunk in _run(["ollama", "create", name, "-f", "Modelfile"],
                      f"> ollama create {name} -f Modelfile\n", cwd=gguf_dir):
        log = chunk
        yield log, "⏳ Đang import vào Ollama..."
    yield (log, f"✅ Xong — test: `ollama run {name} \"Đọc chữ trong ảnh\" -- /path/to/anh.jpg`")


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
        urls = [f"https://huggingface.co/{hub_repo}/blob/main/{name}"]
        mm = _mmproj_sibling(gguf)
        if mm:
            mname = Path(mm).name
            api.upload_file(path_or_fileobj=mm, path_in_repo=mname,
                            repo_id=hub_repo, token=config.HF_TOKEN)
            urls.append(f"https://huggingface.co/{hub_repo}/blob/main/{mname}")
        modelfile = Path(gguf).parent / "Modelfile"
        if modelfile.exists():
            api.upload_file(path_or_fileobj=str(modelfile), path_in_repo="Modelfile",
                            repo_id=hub_repo, token=config.HF_TOKEN)
            urls.append(f"https://huggingface.co/{hub_repo}/blob/main/Modelfile")
        return ("✅ Đã push:\n" + "\n".join(urls) + "\n"
                "Tải cả 3 file (.gguf text + mmproj + Modelfile) về cùng thư mục rồi "
                "`ollama create <ten> -f Modelfile`.")
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
            "#fileId=https%3A//huggingface.co/tranhuy67896262/Qwen2.5-VL-7B-Instruct-private.ipynb)"
        )

        with gr.Tab("Fine-tune"):
            gr.Markdown(
                "Mỗi lần chạy train tiếp **1 lát data mới** (`count` mẫu) từ đúng mốc repo "
                "đang có — script tự dò nhánh `stage-*` cao nhất trên Hub, kéo adapter về "
                "train tiếp, push nhánh mới. Không cần nhớ start/init."
            )
            with gr.Row():
                stage_repo = gr.Textbox(
                    label="Repo adapter (owner/repo)",
                    placeholder="username/qwen25vl-3b-vi-hwr-lora (thiếu owner → lấy user của token)",
                )
                stage_model = gr.Dropdown(
                    choices=STAGE_MODELS,
                    value="qwenvl-3b", label="Model",
                    allow_custom_value=True,
                )
            with gr.Row():
                stage_count = gr.Textbox(
                    value="5000", label="Số mẫu mỗi lát (count)",
                    placeholder="5000 hoặc 10k",
                )
                stage_save = gr.Number(
                    value=50, label="Push snapshot mỗi N step (save_steps)",
                    precision=0,
                )
                stage_preset = gr.Radio(
                    choices=[
                        "Auto theo model (3B: batch 8 QLoRA · 7B: batch 4 bf16)",
                        "Nhanh nhất (A100 80GB — 3B bf16 batch 16 · 7B bf16 batch 8)",
                        "Tiết kiệm VRAM (batch 2)",
                    ],
                    value="Auto theo model (3B: batch 8 QLoRA · 7B: batch 4 bf16)",
                    label="Chế độ batch",
                )
            with gr.Row():
                stage_dataset = gr.Dropdown(
                    choices=STAGE_DATASETS,
                    value=STAGE_DATASETS[0], label="Dataset train",
                    allow_custom_value=True,
                )
            with gr.Accordion("Nâng cao: đổi dataset giữa chừng / data nhỏ", open=False):
                gr.Markdown(
                    "Đổi dataset trên cùng adapter (vd train xong bộ line, chuyển sang bộ "
                    "đoạn văn): nhập `0` vào ô ép start — init vẫn lấy weight mốc mới nhất. "
                    "Bộ ≤2k mẫu thì tăng epoch lên 2-3 cho đủ step."
                )
                with gr.Row():
                    stage_force = gr.Textbox(
                        label="Ép start (FORCE_START)",
                        placeholder="trống = tự dò từ Hub; 0 = về đầu dataset mới",
                    )
                    stage_epochs = gr.Number(
                        value=1, label="Epoch mỗi lát (EPOCHS)",
                        precision=0, minimum=1,
                    )
            with gr.Row():
                progress_btn = gr.Button("🔍 Kiểm tra repo tới đâu", variant="secondary")
                dry_btn = gr.Button("📋 Xem kế hoạch (không train)", variant="secondary")
                stage_btn = gr.Button("▶ Fine-tune tiếp 1 lát", variant="primary")
            stage_progress = gr.Markdown()
            train_log = gr.Textbox(label="Log", lines=20, max_lines=30, autoscroll=True, elem_classes=["log-scroll"])
            progress_btn.click(
                stage_progress_ui,
                inputs=[stage_repo, stage_model],
                outputs=stage_progress,
            )
            dry_btn.click(
                stage_dryrun_ui,
                inputs=[stage_repo, stage_model, stage_count, stage_save, stage_preset,
                        stage_dataset, stage_force, stage_epochs],
                outputs=train_log,
            )
            stage_btn.click(
                stage_train_ui,
                inputs=[stage_repo, stage_model, stage_count, stage_save, stage_preset,
                        stage_dataset, stage_force, stage_epochs],
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
            eval_rev = gr.Textbox(label="Revision adapter (tùy chọn)",
                                  placeholder="stage-10k — để trống = nhánh main/mặc định")
            eval_model.change(sync_adapter, inputs=[eval_model, eval_adapter], outputs=eval_adapter)
            eval_btn = gr.Button("📊 Eval", variant="primary")
            eval_log = gr.Textbox(label="Log", lines=20, max_lines=30, autoscroll=True, elem_classes=["log-scroll"])
            eval_btn.click(eval_ui, inputs=[num_test, eval_adapter, eval_model, eval_rev], outputs=eval_log)

        with gr.Tab("Export"):
            export_adapter = gr.Textbox(value=str(cfg.ADAPTER_DIR), label="Adapter")
            export_model = gr.Dropdown(choices=MODEL_CHOICES, value=cfg.MODEL_NAME,
                                       label="Base model", allow_custom_value=True)
            export_rev = gr.Textbox(label="Revision adapter (tùy chọn)",
                                    placeholder="stage-10k — để trống = nhánh main/mặc định")
            export_model.change(sync_adapter, inputs=[export_model, export_adapter], outputs=export_adapter)
            export_btn = gr.Button("📦 Export ra thư mục (full model)", variant="primary")
            export_log = gr.Textbox(label="Log", lines=20, max_lines=30, autoscroll=True, elem_classes=["log-scroll"])
            export_btn.click(export_ui, inputs=[export_adapter, export_model, export_rev], outputs=export_log)

            gguf_btn = gr.Button("⬇ Download file .gguf", variant="primary")
            gguf_status = gr.Markdown()
            _gguf0 = _latest_gguf()
            dl_gguf = gr.DownloadButton(
                f"⬇ Tải {Path(_gguf0).name}" if _gguf0 else "⬇ Tải file GGUF",
                value=_gguf0, visible=bool(_gguf0))
            gguf_btn.click(export_gguf_ui, inputs=[export_adapter, export_model, export_rev],
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

            gr.Markdown("### 🦙 Import vào Ollama đang chạy trên máy này")
            ollama_name_in = gr.Textbox(value=_default_ollama_name(),
                                        label="Tên model Ollama")
            import_btn = gr.Button("🦙 Import .gguf vào Ollama", variant="secondary")
            import_msg = gr.Markdown()
            import_btn.click(import_ollama_ui, inputs=[ollama_name_in],
                             outputs=[export_log, import_msg])

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
