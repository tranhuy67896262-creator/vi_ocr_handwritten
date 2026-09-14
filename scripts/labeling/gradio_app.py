"""Label editor bằng **Gradio** (thay cho HTTP server + template tự viết).

Giữ nguyên tầng nghiệp vụ (``LabelService`` / sources / ``merge``) — chỉ thay
web layer bằng Gradio (đã dùng ở ``scripts/ui.py``).
"""
from __future__ import annotations

from io import BytesIO
from pathlib import Path

import gradio as gr
from PIL import Image

from .domain import LabelError
from .service import LabelService

PAGE_SIZE = 12
SPLIT_CHOICES = [("Tất cả", ""), ("train", "train"), ("test", "test")]


def _to_pil(data: bytes) -> Image.Image:
    return Image.open(BytesIO(data)).convert("RGB")


def _placeholder() -> Image.Image:
    return Image.new("RGB", (160, 48), (235, 235, 235))


class LabelEditor:
    """Bọc ``LabelService`` cho UI Gradio."""

    def __init__(self, service: LabelService):
        self.service = service

    def page(self, source, query, split, offset):
        """Nạp 1 trang bản ghi: gallery + danh sách uid + info + state."""
        offset = max(int(offset or 0), 0)
        try:
            data = self.service.list_records(source=source, query=query or "",
                                             split=split or "", offset=offset, limit=PAGE_SIZE)
        except LabelError as exc:
            return [], gr.update(choices=[], value=None), f"❌ {exc}", 0, {}

        gallery, choices = [], []
        for row in data["rows"]:
            try:
                content, _ = self.service.get_image(row["uid"])
                gallery.append((_to_pil(content), row["image_ref"]))
            except (LabelError, OSError):
                gallery.append((_placeholder(), row["image_ref"]))
            choices.append((row["image_ref"], row["uid"]))
        start = data["offset"] + 1 if data["rows"] else 0
        info = f"Nguồn **{data['source']}** — {start}–{data['offset'] + len(data['rows'])}/{data['total']}"
        state = {"rows": {row["uid"]: row for row in data["rows"]}}
        return gallery, gr.update(choices=choices, value=choices[0][1] if choices else None), \
            info, data["offset"], state

    def first_page(self, source, query, split, _offset):
        """Tìm kiếm -> về trang đầu."""
        return self.page(source, query, split, 0)

    def next_page(self, source, query, split, offset):
        """Trang sau."""
        return self.page(source, query, split, (offset or 0) + PAGE_SIZE)

    def prev_page(self, source, query, split, offset):
        """Trang trước."""
        return self.page(source, query, split, max((offset or 0) - PAGE_SIZE, 0))

    def select(self, uid, state):
        """Chọn 1 bản ghi -> ảnh + text + split."""
        row = (state or {}).get("rows", {}).get(uid)
        if not row:
            return None, "", "train", "Chọn một ảnh trong danh sách."
        try:
            content, _ = self.service.get_image(uid)
            image = _to_pil(content)
        except (LabelError, OSError):
            image = _placeholder()
        notes = []
        if not row.get("can_edit_text"):
            notes.append("không sửa text")
        if not row.get("can_edit_split"):
            notes.append("không đổi split")
        if not row.get("can_delete"):
            notes.append("không xoá")
        return image, row.get("text", ""), row.get("split", "train"), \
            f"`{uid}` — {' | '.join(notes) or 'sửa được đầy đủ'}"

    def save_text(self, uid, text):
        """Lưu text cho bản ghi."""
        if not uid:
            return "Chưa chọn ảnh."
        try:
            self.service.update_text(uid, text or "")
            return "✅ Đã lưu text."
        except LabelError as exc:
            return f"❌ {exc}"

    def save_split(self, uid, split):
        """Đổi split cho bản ghi."""
        if not uid:
            return "Chưa chọn ảnh."
        try:
            self.service.update_split(uid, split)
            return "✅ Đã đổi split."
        except LabelError as exc:
            return f"❌ {exc}"

    def delete(self, uid):
        """Xoá bản ghi."""
        if not uid:
            return "Chưa chọn ảnh."
        try:
            self.service.delete_record(uid)
            return f"✅ Đã xoá `{uid}`."
        except LabelError as exc:
            return f"❌ {exc}"

    def add(self, source, file_path, text, split):
        """Thêm ảnh mới vào nguồn (chỉ nguồn mutable)."""
        del source  # chọn nguồn qua dropdown; thêm luôn vào nguồn local
        if not file_path:
            return "Chưa chọn file."
        try:
            path = Path(file_path)
            record = self.service.add_record(
                source="local", filename=path.name,
                content=path.read_bytes(), text=text or "", split=split or "train",
            )
            return f"✅ Đã thêm `{record['image_ref']}`."
        except (LabelError, OSError) as exc:
            return f"❌ {exc}"


def build_app(service: LabelService) -> gr.Blocks:
    """Dựng app Gradio cho label editor."""
    editor = LabelEditor(service)
    src_choices = [(src["label"], src["name"]) for src in service.describe_sources()]
    default_src = src_choices[0][1] if src_choices else None

    with gr.Blocks(title="OCR Label Editor") as demo:
        gr.Markdown("# 🏷️ OCR Label Editor\nNguồn HF chỉ sửa text; nguồn local sửa/thêm/xoá đầy đủ.")
        offset = gr.State(0)
        state = gr.State({})
        with gr.Row():
            source = gr.Dropdown(choices=src_choices, value=default_src, label="Nguồn", scale=1)
            split = gr.Dropdown(choices=SPLIT_CHOICES, value="", label="Split", scale=1)
            query = gr.Textbox(label="Tìm (ảnh/nhãn)", scale=2)
            search = gr.Button("🔍 Tìm", variant="primary")
        with gr.Row():
            prev = gr.Button("⬅ Trước")
            info = gr.Markdown("")
            nxt = gr.Button("Sau ➡")
        with gr.Row():
            gallery = gr.Gallery(label="Kết quả", columns=6, height=260, allow_preview=False)
            uid = gr.Dropdown(label="Chọn ảnh", choices=[])
        with gr.Row():
            preview = gr.Image(label="Ảnh", type="pil", interactive=False)
            with gr.Column():
                text = gr.Textbox(label="Nhãn (text)", lines=6)
                row_split = gr.Radio(choices=["train", "test"], label="Split")
                with gr.Row():
                    save = gr.Button("💾 Lưu text", variant="primary")
                    setsp = gr.Button("↔ Đổi split")
                    delete = gr.Button("🗑️ Xoá", variant="stop")
                status = gr.Markdown("")
        gr.Markdown("### ➕ Thêm ảnh vào nguồn local")
        with gr.Row():
            upload = gr.File(label="Ảnh mới", file_types=["image"])
            add_text = gr.Textbox(label="Nhãn", scale=2)
            add_split = gr.Radio(choices=["train", "test"], value="train", label="Split")
            add_btn = gr.Button("Thêm")
        add_status = gr.Markdown("")

        page_outputs = [gallery, uid, info, offset, state]
        inputs = [source, query, split, offset]
        search.click(editor.first_page, inputs, page_outputs)
        nxt.click(editor.next_page, inputs, page_outputs)
        prev.click(editor.prev_page, inputs, page_outputs)
        demo.load(editor.first_page, inputs, page_outputs)  # pylint: disable=no-member
        uid.change(editor.select, [uid, state], [preview, text, row_split, status])
        save.click(editor.save_text, [uid, text], status)
        setsp.click(editor.save_split, [uid, row_split], status)
        delete.click(editor.delete, [uid], status)
        add_btn.click(editor.add, [source, upload, add_text, add_split], add_status)
    return demo
