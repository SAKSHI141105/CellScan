"""Single-command launcher: starts the FastAPI backend and the Vite dev
server together, waits for both to actually answer, then opens the browser.

Exists because "run two things in two terminals" is exactly the kind of
instruction people skip half of, then report the app as broken when it's
really just the API process that was never started. This removes that
failure mode entirely.

    python run.py

Ctrl+C stops both processes. Use --no-browser on a headless box / CI.
"""
from __future__ import annotations

import argparse
import shutil
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
API_PORT = 8000
FRONTEND_PORT = 5173
API_HEALTH_URL = f"http://localhost:{API_PORT}/api/health"
FRONTEND_URL = f"http://localhost:{FRONTEND_PORT}"

# npm on Windows is npm.cmd — plain "npm" via subprocess without shell=True
# fails silently there, this is the actual fix, not just defensive paranoia
NPM_EXECUTABLE = shutil.which("npm") or shutil.which("npm.cmd")


def _venv_python() -> str:
    venv_python = PROJECT_ROOT / (".venv/Scripts/python.exe" if sys.platform == "win32" else ".venv/bin/python")
    return str(venv_python) if venv_python.exists() else sys.executable


def _wait_until_up(url: str, label: str, timeout: float = 45.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            urllib.request.urlopen(url, timeout=1.5)
            return True
        except (urllib.error.URLError, ConnectionError, OSError):
            time.sleep(0.5)
    print(f"[run.py] {label} didn't come up within {timeout:.0f}s — check the process output above.")
    return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-browser", action="store_true", help="don't auto-open the browser")
    args = parser.parse_args()

    if NPM_EXECUTABLE is None:
        print("[run.py] npm not found on PATH — install Node.js first (see README setup).")
        sys.exit(1)
    if not (PROJECT_ROOT / "frontend" / "node_modules").exists():
        print("[run.py] frontend/node_modules missing — run `npm install` inside frontend/ first.")
        sys.exit(1)

    print("[run.py] starting API on :8000 ...")
    api_proc = subprocess.Popen(
        [_venv_python(), "-m", "uvicorn", "src.api.main:app", "--port", str(API_PORT), "--reload"],
        cwd=PROJECT_ROOT,
    )

    print("[run.py] starting frontend on :5173 ...")
    frontend_proc = subprocess.Popen(
        [NPM_EXECUTABLE, "run", "dev"],
        cwd=PROJECT_ROOT / "frontend",
    )

    procs = [api_proc, frontend_proc]

    def _shutdown(*_):
        print("\n[run.py] shutting down...")
        for p in procs:
            if p.poll() is None:
                p.terminate()
        for p in procs:
            try:
                p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                p.kill()
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)

    api_ready = _wait_until_up(API_HEALTH_URL, "API")
    frontend_ready = _wait_until_up(FRONTEND_URL, "Frontend")

    if api_ready and frontend_ready:
        print(f"\n[run.py] ready — {FRONTEND_URL}\n")
        if not args.no_browser:
            webbrowser.open(FRONTEND_URL)
    else:
        print("\n[run.py] one or both services failed to start — leaving processes running so you can read their logs above.")

    # block here until Ctrl+C; if either child dies on its own, surface that
    # rather than hanging forever
    while True:
        for p in procs:
            if p.poll() is not None:
                print(f"\n[run.py] a process exited unexpectedly (code {p.returncode}) — shutting down the other one.")
                _shutdown()
        time.sleep(1)


if __name__ == "__main__":
    main()
