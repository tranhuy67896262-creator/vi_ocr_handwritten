"""Entry cho pre-commit hook: chay pylint neu tim thay, khong thi bo qua.

Tim python theo thu tu: python dang chay hook -> .venv cua project
(Windows + Linux). Khong bao gio chan commit chi vi thieu tool;
chi chan khi code loi that (pylint exit != 0).
"""

import subprocess
import sys
from pathlib import Path

TARGETS = ["configs", "src", "scripts", "main.py"]


def _has_pylint(python):
    try:
        r = subprocess.run(
            [python, "-c", "import pylint"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False,
        )
        return r.returncode == 0
    except OSError:
        return False


def _candidates():
    yield sys.executable
    root = Path(__file__).resolve().parent.parent
    yield str(root / ".venv" / "Scripts" / "python.exe")
    yield str(root / ".venv" / "bin" / "python")


def main():
    """Tim python co pylint roi chay, khong thay thi exit 0 (khong chan commit)."""
    seen = set()
    for python in _candidates():
        if not python or python in seen:
            continue
        seen.add(python)
        if _has_pylint(python):
            r = subprocess.run([python, "-m", "pylint", *TARGETS], check=False)
            return r.returncode
    print("Bo qua pylint: khong tim thay python co pylint "
          "(cai vao .venv: uv pip install pylint). Commit van di tiep.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
