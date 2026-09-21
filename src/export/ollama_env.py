"""Đảm bảo môi trường Ollama: CLI + server (tự cài trên Colab/Linux)."""
import shutil
import subprocess
import sys
import time

from .proc import stream_command

APT_TIMEOUT = 300
INSTALL_TIMEOUT = 600
SERVE_LOG = "/tmp/ollama-vi-ocr.log"
SERVE_WAIT_ROUNDS = 36
SERVE_WAIT_SECONDS = 5


class OllamaEnv:
    """Quản lý vòng đời Ollama runtime; không biết gì về Gradio/train."""

    def __init__(self, runner=stream_command):
        self._run = runner

    def cli_available(self):
        """Máy đã có lệnh ollama chưa."""
        return shutil.which("ollama") is not None

    def server_up(self):
        """Ollama server có đang trả lời không."""
        try:
            probe = subprocess.run(["ollama", "list"], capture_output=True,
                                   timeout=30, check=False)
            return probe.returncode == 0
        except Exception:
            return False

    def install(self, emit):
        """Cài Ollama CLI (apt + install.sh). Trả True nếu xong (không hỗ trợ Windows)."""
        if sys.platform == "win32":
            emit("Chưa có Ollama CLI — cài tay tại https://ollama.com/download .")
            return False
        emit("Chưa có Ollama CLI — tự cài (apt + install.sh, vài phút)...")
        if self._run(["sudo", "apt-get", "update"], emit) != 0:
            emit("[ERR] apt-get update thất bại (cần sudo không mật khẩu như Colab).")
            return False
        if self._run(["sudo", "apt-get", "install", "-y", "zstd", "curl"], emit) != 0:
            emit("[ERR] Không cài được zstd/curl.")
            return False
        code = self._run(["bash", "-c", "curl -fsSL https://ollama.com/install.sh | sh"], emit)
        if code != 0 or not self.cli_available():
            emit("[ERR] Cài Ollama thất bại.")
            return False
        emit("Đã cài Ollama CLI.")
        return True

    def start_server(self, emit):
        """Chạy `ollama serve` nền + đợi server lên. Trả True nếu lên."""
        if sys.platform == "win32":
            emit("Ollama server chưa chạy — mở terminal chạy `ollama serve` trước.")
            return False
        emit(f"Khởi động Ollama server nền (nohup, log {SERVE_LOG})...")
        try:
            # Process tách nền phải sống sau khi hàm trả về nên không dùng `with`
            # (context manager sẽ đợi process kết thúc).
            # pylint: disable=consider-using-with
            subprocess.Popen(
                ["bash", "-c", f"nohup ollama serve > {SERVE_LOG} 2>&1 &"],
                start_new_session=True,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception as exc:
            emit(f"[ERR] Không khởi động được serve: {exc}")
            return False
        for _ in range(SERVE_WAIT_ROUNDS):
            time.sleep(SERVE_WAIT_SECONDS)
            if self.server_up():
                emit("Ollama server đã lên.")
                return True
        emit(f"[ERR] Ollama server không lên — xem {SERVE_LOG}.")
        return False

    def ensure(self, emit):
        """Đảm bảo CLI + server đều sẵn sàng (tự cài/chạy khi thiếu)."""
        if not self.cli_available() and not self.install(emit):
            return False
        if self.server_up():
            emit("Ollama server đang chạy.")
            return True
        return self.start_server(emit)
