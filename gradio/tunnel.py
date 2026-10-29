"""Public tunnel helpers for remote environments like Kaggle and Google Colab."""
import atexit
import logging
import os
import platform
import re
import shutil
import stat
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

log = logging.getLogger(__name__)

CLOUDFLARED_URLS = {
    ("linux", "x86_64"): "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64",
    ("linux", "amd64"): "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64",
    ("linux", "aarch64"): "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-arm64",
    ("linux", "arm64"): "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-arm64",
    ("windows", "amd64"): "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe",
    ("windows", "x86_64"): "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe",
    ("darwin", "x86_64"): "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-darwin-amd64",
    ("darwin", "arm64"): "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-darwin-arm64",
}


def get_cloudflared_binary() -> Path:
    """Find or download cloudflared binary for the current operating system."""
    found = shutil.which("cloudflared")
    if found:
        return Path(found)

    cache_dir = Path.home() / ".cache" / "cloudflared"
    cache_dir.mkdir(parents=True, exist_ok=True)

    system = platform.system().lower()
    machine = platform.machine().lower()
    exe_name = "cloudflared.exe" if system == "windows" else "cloudflared"
    local_path = cache_dir / exe_name

    if local_path.is_file() and os.access(local_path, os.X_OK):
        return local_path

    url = CLOUDFLARED_URLS.get((system, machine))
    if not url:
        url = CLOUDFLARED_URLS.get((system, "amd64"))

    if not url:
        raise RuntimeError(f"Unsupported platform for cloudflared: {system} {machine}")

    print(f"Downloading cloudflared binary for {system}/{machine}...", flush=True)
    urllib.request.urlretrieve(url, local_path)
    local_path.chmod(local_path.stat().st_mode | stat.S_IEXEC | 0o111)
    print(f"cloudflared ready at: {local_path}", flush=True)
    return local_path


def start_cloudflare_tunnel(port: int, wait_seconds: int = 35) -> str:
    """Start cloudflared tunnel to localhost:port and return the public URL."""
    binary = get_cloudflared_binary()

    log_file = binary.parent / f"tunnel_{port}.log"
    cmd = [
        str(binary),
        "tunnel",
        "--url",
        f"http://127.0.0.1:{port}",
        "--no-autoupdate",
    ]

    with open(log_file, "w", encoding="utf-8") as out:
        proc = subprocess.Popen(
            cmd,
            stdout=out,
            stderr=subprocess.STDOUT,
            text=True,
        )

    def _cleanup():
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                proc.kill()

    atexit.register(_cleanup)

    url_pattern = re.compile(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com")
    start_time = time.time()
    tunnel_url = None

    while time.time() - start_time < wait_seconds:
        if proc.poll() is not None:
            with open(log_file, "r", encoding="utf-8", errors="ignore") as f:
                logs = f.read()
            raise RuntimeError(f"cloudflared exited unexpectedly (code {proc.returncode}):\n{logs}")

        if log_file.exists():
            with open(log_file, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
            match = url_pattern.search(content)
            if match:
                tunnel_url = match.group(0)
                break

        time.sleep(0.5)

    if not tunnel_url:
        with open(log_file, "r", encoding="utf-8", errors="ignore") as f:
            logs = f.read()
        raise TimeoutError(f"Timed out waiting for Cloudflare tunnel URL:\n{logs}")

    print("\n" + "=" * 68, flush=True)
    print(f"[*] PUBLIC CLOUDFLARE SHARE LINK: {tunnel_url}", flush=True)
    print("=" * 68 + "\n", flush=True)
    return tunnel_url


def start_ngrok_tunnel(port: int, token: str = None) -> str:
    """Start ngrok tunnel using pyngrok."""
    try:
        from pyngrok import ngrok
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "pyngrok"], check=True)
        from pyngrok import ngrok

    if token:
        ngrok.set_auth_token(token)

    tunnel = ngrok.connect(port, "http")
    print("\n" + "=" * 68, flush=True)
    print(f"[*] PUBLIC NGROK SHARE LINK: {tunnel.public_url}", flush=True)
    print("=" * 68 + "\n", flush=True)
    return tunnel.public_url
