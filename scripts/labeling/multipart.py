"""Parser multipart/form-data tối giản (chỉ stdlib) cho upload ảnh."""

from __future__ import annotations

import re


def parse_multipart(body: bytes, boundary: bytes) -> tuple[dict[str, str], dict[str, dict[str, object]]]:
    """Tách body multipart thành ``(fields, files)``.

    ``fields``: tên -> giá trị text. ``files``: tên -> {filename, content}.
    """
    fields: dict[str, str] = {}
    files: dict[str, dict[str, object]] = {}
    for chunk in body.split(b"--" + boundary):
        if b"Content-Disposition" not in chunk:
            continue
        head, _, content = chunk.partition(b"\r\n\r\n")
        if content.endswith(b"\r\n"):
            content = content[:-2]
        name_match = re.search(rb'name="([^"]*)"', head)
        if not name_match:
            continue
        name = name_match.group(1).decode("utf-8", "replace")
        file_match = re.search(rb'filename="([^"]*)"', head)
        if file_match:
            files[name] = {
                "filename": file_match.group(1).decode("utf-8", "replace"),
                "content": content,
            }
        else:
            fields[name] = content.decode("utf-8", "replace")
    return fields, files
