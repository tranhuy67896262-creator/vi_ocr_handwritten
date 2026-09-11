"""Tầng HTTP: router mỏng gọi ``LabelService`` và trả JSON/ảnh/HTML.

DIP: handler phụ thuộc abstraction ``LabelService`` (được tiêm từ ngoài),
không tự tạo nguồn dữ liệu.
"""

from __future__ import annotations

import json
import re
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .domain import LabelError, RecordNotFound, SourceUnavailable, ValidationError
from .multipart import parse_multipart
from .service import LabelService

MAX_REQUEST_BYTES = 25 * 1024 * 1024


def _status_for(error: LabelError) -> HTTPStatus:
    """Ánh xạ lỗi nghiệp vụ sang mã HTTP."""
    if isinstance(error, SourceUnavailable):
        return HTTPStatus.SERVICE_UNAVAILABLE
    if isinstance(error, RecordNotFound):
        return HTTPStatus.NOT_FOUND
    if isinstance(error, ValidationError):
        return HTTPStatus.BAD_REQUEST
    return HTTPStatus.BAD_REQUEST


def create_handler(service: LabelService, template_path: Path):
    """Tạo lớp handler đã gắn service + template (factory injection)."""
    template = Path(template_path)

    class LabelRequestHandler(BaseHTTPRequestHandler):
        """HTTP handler cho label editor (mỗi instance phục vụ một request)."""

        def log_message(self, fmt: str, *args: object) -> None:  # pylint: disable=arguments-differ
            """Ghi log request ra stdout."""
            print(f"[{self.log_date_time_string()}] {fmt % args}")

        def do_GET(self) -> None:
            """Điều phối GET."""
            self._dispatch("GET")

        def do_PUT(self) -> None:
            """Điều phối PUT."""
            self._dispatch("PUT")

        def do_POST(self) -> None:
            """Điều phối POST."""
            self._dispatch("POST")

        def do_DELETE(self) -> None:
            """Điều phối DELETE."""
            self._dispatch("DELETE")

        # ------------------------------------------------------------------ core
        def _dispatch(self, method: str) -> None:
            try:
                parsed = urlparse(self.path)
                route = self._routes().get(method, {}).get(parsed.path)
                if route is None:
                    self.send_error(HTTPStatus.NOT_FOUND)
                    return
                route(parsed)
            except LabelError as error:
                self._send_json({"error": str(error)}, _status_for(error))
            except (ValueError, KeyError, json.JSONDecodeError) as error:
                self._send_json({"error": str(error)}, HTTPStatus.BAD_REQUEST)
            except Exception as error:  # noqa: BLE001 - UI hiển thị lỗi thay vì crash
                self._send_json({"error": str(error)}, HTTPStatus.INTERNAL_SERVER_ERROR)

        def _routes(self) -> dict[str, dict[str, object]]:
            return {
                "GET": {
                    "/": self._serve_editor,
                    "/api/sources": self._serve_sources,
                    "/api/rows": self._serve_rows,
                    "/api/image": self._serve_image,
                },
                "PUT": {
                    "/api/rows": self._update_row,
                },
                "POST": {
                    "/api/rows/upload": self._upload_row,
                },
                "DELETE": {
                    "/api/rows": self._delete_row,
                },
            }

        # ------------------------------------------------------------------ GET
        def _serve_editor(self, _parsed) -> None:
            self._send_bytes(template.read_bytes(), "text/html; charset=utf-8")

        def _serve_sources(self, _parsed) -> None:
            self._send_json({"sources": service.describe_sources()})

        def _serve_rows(self, parsed) -> None:
            params = self._params(parsed)
            payload = service.list_records(
                source=params.get("source", ""),
                query=params.get("query", ""),
                split=params.get("split", ""),
                offset=self._int(params.get("offset"), 0),
                limit=self._int(params.get("limit"), 50),
            )
            self._send_json(payload)

        def _serve_image(self, parsed) -> None:
            uid = self._params(parsed).get("uid", "")
            data, content_type = service.get_image(uid)
            self._send_bytes(data, content_type)

        # ------------------------------------------------------------------ PUT
        def _update_row(self, _parsed) -> None:
            payload = self._read_json()
            uid = str(payload.get("uid") or "")
            if not uid:
                raise ValidationError("Thiếu uid")
            if "text" in payload:
                service.update_text(uid, str(payload.get("text") or ""))
            if "split" in payload:
                service.update_split(uid, str(payload.get("split") or ""))
            self._send_json({"saved": True, "uid": uid})

        # ----------------------------------------------------------------- POST
        def _upload_row(self, _parsed) -> None:
            match = re.search(r"boundary=([^;]+)", self.headers.get("Content-Type", ""))
            if not match:
                raise ValidationError("Thiếu multipart boundary")
            length = int(self.headers.get("Content-Length", "0"))
            if length > MAX_REQUEST_BYTES:
                raise ValidationError("Request quá lớn")
            fields, files = parse_multipart(
                self.rfile.read(length),
                match.group(1).strip().strip('"').encode("ascii", "ignore"),
            )
            if "image" not in files:
                raise ValidationError("Thiếu file ảnh")
            upload = files["image"]
            record = service.add_record(
                source=fields.get("source", ""),
                filename=str(upload["filename"]),
                content=bytes(upload["content"]),
                text=fields.get("text", ""),
                split=fields.get("split", "train"),
            )
            self._send_json(record, HTTPStatus.CREATED)

        # --------------------------------------------------------------- DELETE
        def _delete_row(self, parsed) -> None:
            uid = self._params(parsed).get("uid", "")
            self._send_json(service.delete_record(uid))

        # --------------------------------------------------------------- helpers
        @staticmethod
        def _params(parsed) -> dict[str, str]:
            return {key: values[0] for key, values in parse_qs(parsed.query).items()}

        @staticmethod
        def _int(value: str | None, default: int) -> int:
            try:
                return int(value)
            except (TypeError, ValueError):
                return default

        def _read_json(self) -> dict:
            length = int(self.headers.get("Content-Length", "0"))
            if not length:
                return {}
            return json.loads(self.rfile.read(length).decode("utf-8"))

        def _send_json(self, payload: object, status: HTTPStatus = HTTPStatus.OK) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self._send_bytes(body, "application/json; charset=utf-8", status)

        def _send_bytes(self, data: bytes, content_type: str, status: HTTPStatus = HTTPStatus.OK) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    return LabelRequestHandler
