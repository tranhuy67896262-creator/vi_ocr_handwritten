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
    "tranhuy67896262/Qwen2.5-VL-3B-Instruct-private",
    "tranhuy67896262/Qwen2.5-VL-7B-Instruct-private",
]


def _adapter_for_model(model_name):
    """Giá trị hiện sẵn trong ô Adapter: path local nếu đã có trọng số,
    ngược lại repo id Hub (máy Colab mới không phải gõ tay)."""
    from configs.configs import default_adapter_dir, default_hub_repo
    local = default_adapter_dir(PROJECT_ROOT / "models", model_name or MODEL_CHOICES[0])
    if (local / "adapter_config.json").is_file():
        return str(local)
    return default_hub_repo(model_name or MODEL_CHOICES[0])


def sync_adapter(model_name, current):
    """Đổi base model -> tự trỏ adapter về giá trị mặc định tương ứng
    (path local nếu có weight, ngược lại repo id Hub).
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
    """Env batch/precision + mode 7B cho stage_train.sh theo preset đã chọn.

    Mode 7B khác 3B: luôn gửi LORA_R=64/LORA_ALPHA=128 (script chỉ dùng khi
    train trắng, nối chain cũ thì bỏ qua). KL giữ mặc định cả 2 mode.
    """
    label = (preset or "")
    if label.startswith("Nhanh"):
        key = "fast"
    elif label.startswith("Ti"):
        key = "saver"
    else:
        key = "auto"
    short = (model or "").strip().lower()
    tag = "qwenvl-7b" if "7b" in short else "qwenvl-3b"
    env = {}
    if "7b" in short:
        env["LORA_R"] = "64"
        env["LORA_ALPHA"] = "128"
    if key == "auto":
        return env
    use_4bit, batch, accum = BATCH_PRESETS[key][tag]
    env.update({"USE_4BIT": use_4bit, "BATCH_SIZE": batch, "GRAD_ACCUM": accum})
    return env


# ---------------- Repo của tôi (chọn repo để train tiếp) ----------------

_REPO_MODEL_CACHE = {}


def _repo_latest_stage(repo, token):
    """(max_k, tên nhánh stage-*) của repo; (None, None) nếu chưa có mốc nào."""
    from huggingface_hub import HfApi
    refs = HfApi(token=token).list_repo_refs(repo_id=repo)
    best_k, best_name = None, None
    for br in refs.branches:
        name = br.name
        if not name.startswith("stage-"):
            continue
        core = name[len("stage-"):]
        knum = core.split("k")[0] if "k" in core else ""
        if knum.isdigit() and (best_k is None or int(knum) > best_k):
            best_k, best_name = int(knum), name
    return best_k, best_name


def _repo_model_tag(repo, token, rev):
    """qwenvl-3b/7b của repo: đọc base model trong adapter_config.json, fallback tên repo."""
    key = (repo, rev or "main")
    if key in _REPO_MODEL_CACHE:
        return _REPO_MODEL_CACHE[key]
    tag = ""
    try:
        import json
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(repo_id=repo, filename="adapter_config.json",
                               revision=rev, token=token)
        with open(path, encoding="utf-8") as fh:
            base = json.load(fh).get("base_model_name_or_path", "")
        low = str(base).lower()
        if "3b" in low:
            tag = "qwenvl-3b"
        elif "7b" in low:
            tag = "qwenvl-7b"
    except Exception:
        pass
    if not tag:
        low = repo.lower()
        if "3b" in low:
            tag = "qwenvl-3b"
        elif "7b" in low:
            tag = "qwenvl-7b"
    _REPO_MODEL_CACHE[key] = tag
    return tag


def _repo_stage_dataset(repo, token, rev):
    """Dataset ngắn gọn của mốc staged (đọc stage_progress.json; \"?\" nếu không có)."""
    key = (repo, rev or "main", "dataset")
    if key in _REPO_MODEL_CACHE:
        return _REPO_MODEL_CACHE[key]
    short = "?"
    try:
        import json
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(repo_id=repo, filename="stage_progress.json",
                               revision=rev, token=token)
        with open(path, encoding="utf-8") as fh:
            full = json.load(fh).get("dataset", "")
        if full:
            short = full.rsplit("/", 1)[-1]
    except Exception:
        pass
    _REPO_MODEL_CACHE[key] = short
    return short


def refresh_my_repos():
    """Nút tải repo adapter của user: yield (dropdown, state, bảng, trạng thái)."""
    config = Configs()
    if not config.HF_TOKEN:
        yield gr.update(choices=[], value=None), {}, "Chưa có HF_TOKEN — sang tab Settings lưu token trước.", ""
        return
    user = _token_username(config.HF_TOKEN)
    if not user:
        yield gr.update(choices=[], value=None), {}, "Không đọc được user từ token (offline?).", ""
        return
    try:
        from huggingface_hub import HfApi
        repos = list(HfApi(token=config.HF_TOKEN).list_models(author=user, limit=50))
    except Exception as exc:
        yield gr.update(choices=[], value=None), {}, f"Không liệt kê được repo: {exc}", ""
        return
    if not repos:
        yield gr.update(choices=[], value=None), {}, f"Tài khoản `{user}` chưa có model repo nào.", ""
        return
    state, rows, ids = {}, [], [r.id for r in repos]
    for pos, rid in enumerate(ids, 1):
        yield gr.skip(), {}, "\n".join(rows), f"⏳ Đang đọc {rid} ({pos}/{len(ids)})..."
        try:
            maxk, stagename = _repo_latest_stage(rid, config.HF_TOKEN)
        except Exception:
            maxk, stagename = None, None
        tag = _repo_model_tag(rid, config.HF_TOKEN, stagename)
        model_txt = {"qwenvl-3b": "3B", "qwenvl-7b": "7B"}.get(tag, "?")
        if maxk is None:
            rows.append(f"| `{rid}` | {model_txt} | — | trắng (từ 0) | — |")
            trained = 0
        else:
            data_txt = _repo_stage_dataset(rid, config.HF_TOKEN, stagename)
            rows.append(f"| `{rid}` | {model_txt} | `{stagename}` | ~{maxk}k mẫu | `{data_txt}` |")
            trained = maxk
        state[rid] = {"repo": rid, "model": tag or "qwenvl-3b",
                      "trained_k": trained, "stage": stagename or ""}
    table = ("| Repo | Model | Mốc mới nhất | Đã train | Data |\n"
             "|---|---|---|---|---|\n" + "\n".join(rows))
    done_msg = f"✅ {len(ids)} repo của `{user}` — chọn 1 dòng ở ô dưới để nối tiếp."
    yield gr.update(choices=ids), state, table, done_msg


def on_repo_select(repo_id, state):
    """Chọn repo cũ -> set model + báo mốc nối tiếp; gõ tên mới -> giữ model, báo trắng."""
    repo_id = (repo_id or "").strip()
    if not repo_id:
        return gr.skip(), ""
    info = (state or {}).get(repo_id, {})
    if not info:
        return gr.skip(), ("Repo mới gõ tay → sẽ **train trắng từ 0** "
                            "(nhập đúng `owner/repo` nếu repo đã có trên Hub).")
    model_update = gr.update(value=info["model"]) if info.get("model") else gr.skip()
    if info.get("stage"):
        return model_update, (f"▶ Sẽ **nối tiếp từ `{info['stage']}`** "
                               f"(~{info['trained_k']}k mẫu) — bấm ▶ để chạy.")
    return model_update, "Repo chưa có mốc stage → sẽ **train trắng từ 0**."


def _stage_extra_env(dataset, force_start, epochs, model=None):
    """Env thêm cho stage_train.sh: đổi dataset / ép start / số epoch mỗi lát.

    Epoch mặc định theo mode (3B=1, 7B=2) — chỉ gửi env khi user đổi khác default.
    """
    env = {}
    if dataset and dataset.strip():
        env["DATASET"] = dataset.strip()
    if force_start is not None and str(force_start).strip() != "":
        env["FORCE_START"] = str(force_start).strip()
    default_ep = 2 if "7b" in (model or "").strip().lower() else 1
    try:
        ep = int(epochs) if epochs is not None else default_ep
    except (TypeError, ValueError):
        ep = default_ep
    if ep != default_ep:
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
    env.update(_stage_extra_env(dataset, force_start, epochs, model))
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
    env.update(_stage_extra_env(dataset, force_start, epochs, model))
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

# ---------------- Export 1 nút (UI chỉ orchestrate, logic ở src/export/) ----------------

def _stream_service(target, log_lines):
    """Chạy 1 service call trong thread, stream log realtime vào log_lines.

    ``target`` nhận callback emit(line). Yield nội dung log sau mỗi dòng mới,
    return kết quả của target (lấy qua ``yield from``).
    """
    import queue
    import threading
    box, que = {}, queue.Queue()

    def _emit(line):
        que.put(line)

    def _wrap():
        try:
            box["result"] = target(_emit)
        except Exception as exc:
            que.put(f"[ERR] Lỗi không mong đợi: {exc}")
            box["result"] = None

    thread = threading.Thread(target=_wrap, daemon=True)
    thread.start()
    while thread.is_alive():
        try:
            log_lines.append(que.get(timeout=2))
            yield "\n".join(log_lines)
        except Exception:
            pass
    thread.join()
    while not que.empty():
        log_lines.append(que.get_nowait())
    return box.get("result")


def export_import_ollama_ui(adapter, model, revision, ollama_name, ollama_hub):
    """1 NÚT duy nhất: Ollama sẵn sàng -> build GGUF -> import (+ push Hub nếu điền)."""
    from src.export.bundle import GgufBundle
    from src.export.ollama_env import OllamaEnv
    from src.export.ollama_import import OllamaImporter, default_ollama_name

    if sys.platform == "win32":
        yield "Export GGUF cần Linux/Colab (build llama.cpp) — không chạy trên Windows."
        return
    _free_gpu()
    config = Configs()
    adapter = (adapter or "").strip() or str(config.ADAPTER_DIR)
    model = (model or "").strip() or config.MODEL_NAME
    revision = (revision or "").strip()
    name = (ollama_name or "").strip() or default_ollama_name(adapter, revision)
    hub_name = (ollama_hub or "").strip()
    log_lines = [f"Adapter: {adapter}", f"Model: {model}",
                 f"Revision: {revision or '(mặc định)'}", f"Ollama: {name}",
                 f"Ollama Hub: {hub_name or '(bỏ qua)'}"]
    yield "\n".join(log_lines) + "\n⏳ Bước 0/3 — kiểm tra Ollama (Colab tự cài + serve nếu thiếu)..."
    env_ok = yield from _stream_service(OllamaEnv().ensure, log_lines)
    if not env_ok:
        yield "\n".join(log_lines) + "\n[ERR] Ollama chưa sẵn sàng — dừng."
        return
    log_lines.append("⏳ Bước 1-2/3 — merge adapter + convert GGUF (lần đầu 10–20 phút)...")
    yield "\n".join(log_lines)
    bundle = GgufBundle()
    gguf_dir = yield from _stream_service(
        lambda emit: bundle.build(
            adapter=adapter, model=model, revision=revision,
            models_dir=str(config.MODELS_DIR), python_exe=sys.executable,
            scripts_dir=str(SCRIPT), emit=emit),
        log_lines)
    if not gguf_dir:
        yield "\n".join(log_lines) + "\n[ERR] Không build được gói GGUF."
        return
    log_lines.append(f"⏳ Bước 3/3 — import vào Ollama (`ollama create {name}`)...")
    yield "\n".join(log_lines)
    importer = OllamaImporter()
    imported = yield from _stream_service(
        lambda emit: importer.import_bundle(gguf_dir, name, emit),
        log_lines)
    if not imported:
        yield "\n".join(log_lines) + "\n[ERR] Import Ollama thất bại."
        return
    if hub_name:
        log_lines.append(f"⏳ Bước 4/3 — push lên Ollama Hub (`ollama push {hub_name}`)...")
        yield "\n".join(log_lines)
        pushed = yield from _stream_service(
            lambda emit: importer.push_model(hub_name, emit),
            log_lines)
        if not pushed:
            yield ("\n".join(log_lines) + "\n[ERR] Push Ollama Hub thất bại "
                   "(lần đầu cần `ollama signin` 1 lần trên máy này).")
            return
    yield ("\n".join(log_lines) + f"\n✅ Xong — test: `ollama run {name} "
           f"\"Đọc chữ trong ảnh\" -- /path/to/anh.jpg`")


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
            gr.Markdown("### 📂 Repo adapter — chọn repo cũ để nối tiếp, gõ tên mới để train trắng")
            with gr.Row():
                refresh_btn = gr.Button("🔄 Tải danh sách repo của tôi", variant="secondary")
            repo_state = gr.State({})
            repo_table = gr.Markdown()
            repo_status = gr.Markdown()
            with gr.Row():
                stage_repo = gr.Dropdown(
                    choices=[], label="Repo adapter (chọn cũ / gõ mới: owner/repo)",
                    allow_custom_value=True,
                )
                stage_model = gr.Dropdown(
                    choices=STAGE_MODELS,
                    value="qwenvl-3b", label="Model",
                    allow_custom_value=True,
                )
            refresh_btn.click(
                refresh_my_repos,
                outputs=[stage_repo, repo_state, repo_table, repo_status],
            )
            stage_repo.change(
                on_repo_select,
                inputs=[stage_repo, repo_state],
                outputs=[stage_model, repo_status],
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
                        "Auto theo model (bf16 batch 8 cho cả 3B/7B)",
                        "Nhanh nhất (A100 80GB — 3B bf16 batch 16 · 7B bf16 batch 8)",
                        "Tiết kiệm VRAM (3B QLoRA batch 2)",
                    ],
                    value="Auto theo model (bf16 batch 8 cho cả 3B/7B)",
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
                        value=1, label="Epoch mỗi lát (EPOCHS — mặc định 3B:1, 7B:2)",
                        precision=0, minimum=1,
                    )
            # Đổi model -> tự gợi ý epoch theo mode (user vẫn sửa tay được).
            stage_model.change(
                lambda m: gr.update(value=2 if "7b" in (m or "").lower() else 1),
                inputs=stage_model,
                outputs=stage_epochs,
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
                adapter_in = gr.Textbox(value=_adapter_for_model(cfg.MODEL_NAME),
                                          label="Adapter (đường dẫn hoặc owner/repo)")
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
                eval_adapter = gr.Textbox(value=_adapter_for_model(cfg.MODEL_NAME), label="Adapter")
                eval_model = gr.Dropdown(choices=MODEL_CHOICES, value=cfg.MODEL_NAME,
                                         label="Base model", allow_custom_value=True)
            eval_rev = gr.Textbox(label="Revision adapter (tùy chọn)",
                                  placeholder="stage-10k — để trống = nhánh main/mặc định")
            eval_model.change(sync_adapter, inputs=[eval_model, eval_adapter], outputs=eval_adapter)
            eval_btn = gr.Button("📊 Eval", variant="primary")
            eval_log = gr.Textbox(label="Log", lines=20, max_lines=30, autoscroll=True, elem_classes=["log-scroll"])
            eval_btn.click(eval_ui, inputs=[num_test, eval_adapter, eval_model, eval_rev], outputs=eval_log)

        with gr.Tab("Export"):
            gr.Markdown("1 nút duy nhất: merge adapter → convert GGUF → import vào Ollama "
                        "(Colab tự cài + chạy Ollama server nếu thiếu). Chỉ Linux/Colab.")
            export_adapter = gr.Textbox(value=_adapter_for_model(cfg.MODEL_NAME), label="Adapter")
            export_model = gr.Dropdown(choices=MODEL_CHOICES, value=cfg.MODEL_NAME,
                                       label="Base model", allow_custom_value=True)
            export_rev = gr.Textbox(label="Revision adapter (tùy chọn)",
                                    placeholder="stage-10k — để trống = nhánh main/mặc định")
            export_model.change(sync_adapter, inputs=[export_model, export_adapter], outputs=export_adapter)
            ollama_name_in = gr.Textbox(label="Tên model Ollama (trống = tự đặt theo adapter)",
                                        placeholder="vd qwen25vl-3b-vi-hwr")
            ollama_hub_in = gr.Textbox(label="Đẩy lên Ollama Hub (tùy chọn)",
                                       placeholder="owner/model — trống = bỏ qua (cần ollama signin 1 lần)")
            one_btn = gr.Button("🦙 Export GGUF + Import vào Ollama", variant="primary")
            export_log = gr.Textbox(label="Log", lines=20, max_lines=30, autoscroll=True, elem_classes=["log-scroll"])
            one_btn.click(export_import_ollama_ui,
                          inputs=[export_adapter, export_model, export_rev,
                                  ollama_name_in, ollama_hub_in],
                          outputs=export_log)

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
            check_adapter = gr.Textbox(value=_adapter_for_model(cfg.MODEL_NAME), label="Adapter")
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
