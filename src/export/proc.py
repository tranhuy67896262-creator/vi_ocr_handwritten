"""Primitive chạy tiến trình: stream stdout realtime qua callback emit."""
import subprocess


def stream_command(cmd, emit, cwd=None):
    """Chạy cmd, gọi emit(line) theo từng dòng; trả returncode (None nếu không chạy được)."""
    try:
        with subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace", bufsize=1,
            cwd=str(cwd) if cwd else None,
        ) as proc:
            try:
                for line in proc.stdout:
                    emit(line.rstrip("\n"))
                proc.wait()
                return proc.returncode
            except Exception as exc:
                emit(f"[ERR] Lỗi khi đọc output: {exc}")
                return 1
    except Exception as exc:
        emit(f"[ERR] Không chạy được {' '.join(cmd)}: {exc}")
        return None
