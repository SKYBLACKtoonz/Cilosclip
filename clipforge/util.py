"""Small platform + process + HTTP helpers shared by the rest of ClipForge."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
import uuid
from pathlib import Path

IS_WINDOWS = os.name == "nt"
IS_MAC = sys.platform == "darwin"

# Windows-only: stop console windows flashing up when the GUI spawns ffmpeg.
CREATE_NO_WINDOW = 0x08000000 if IS_WINDOWS else 0


class ClipForgeError(RuntimeError):
    """A problem we can explain to the user without a traceback."""


# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
def config_dir() -> Path:
    if IS_WINDOWS:
        base = Path(os.environ.get("APPDATA") or Path.home() / "AppData" / "Roaming")
    elif IS_MAC:
        base = Path.home() / "Library" / "Application Support"
    else:
        base = Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config")
    path = base / "ClipForge"
    path.mkdir(parents=True, exist_ok=True)
    return path


def default_output_dir() -> Path:
    return Path.home() / "Videos" / "ClipForge"


def find_binary(name: str) -> str | None:
    """Locate ffmpeg/ffprobe, honouring an explicit override and Windows layouts."""
    override = os.environ.get(f"{name.upper()}_BIN")
    if override and Path(override).exists():
        return override

    found = shutil.which(name)
    if found:
        return found

    exe = f"{name}.exe" if IS_WINDOWS else name
    candidates: list[Path] = []
    if IS_WINDOWS:
        local = Path(os.environ.get("LOCALAPPDATA", ""))
        candidates += [
            Path(r"C:\ffmpeg\bin") / exe,
            Path(r"C:\Program Files\ffmpeg\bin") / exe,
            local / "Microsoft" / "WinGet" / "Links" / exe,
            local / "Programs" / "ffmpeg" / "bin" / exe,
        ]
    else:
        candidates += [Path("/usr/local/bin") / name, Path("/usr/bin") / name]

    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return None


# --------------------------------------------------------------------------- #
# Process
# --------------------------------------------------------------------------- #
def run(
    cmd: list[str],
    cwd: str | Path | None = None,
    log=None,
    check: bool = True,
    env: dict | None = None,
) -> subprocess.CompletedProcess:
    """Run a command, optionally stream its output line-by-line to ``log``."""
    if log:
        log("$ " + " ".join(str(c) for c in cmd))
    proc = subprocess.run(
        [str(c) for c in cmd],
        cwd=str(cwd) if cwd else None,
        capture_output=True,
        text=True,
        errors="replace",
        creationflags=CREATE_NO_WINDOW,
        env=env,
    )
    if log and proc.stdout.strip():
        for line in proc.stdout.strip().splitlines()[-40:]:
            log(line)
    if check and proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-12:]
        detail = "\n".join(tail) or "(no output)"
        raise ClipForgeError(
            f"Command failed ({proc.returncode}): {' '.join(str(c) for c in cmd[:3])} ...\n{detail}"
        )
    return proc


# --------------------------------------------------------------------------- #
# Formatting
# --------------------------------------------------------------------------- #
def fmt_ts(seconds: float) -> str:
    """0:07 / 1:23 / 1:02:03, for humans."""
    seconds = max(0, seconds)
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


def ass_time(seconds: float) -> str:
    """ASS subtitle timestamp: H:MM:SS.cc"""
    seconds = max(0.0, seconds)
    cs = int(round(seconds * 100))
    h, cs = divmod(cs, 360000)
    m, cs = divmod(cs, 6000)
    s, cs = divmod(cs, 100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def show_error_dialog(title: str, message: str) -> None:
    """Surface a startup error even when launched without a console (pythonw)."""
    if IS_WINDOWS:
        try:
            import ctypes

            ctypes.windll.user32.MessageBoxW(None, message, title, 0x10)
            return
        except Exception:  # pragma: no cover - best effort
            pass
    print(f"{title}: {message}", file=sys.stderr)


def slugify(text: str, fallback: str = "clip", limit: int = 48) -> str:
    out = []
    for ch in text.lower():
        if ch.isalnum():
            out.append(ch)
        elif ch in " -_":
            out.append("-")
    slug = "".join(out)
    while "--" in slug:
        slug = slug.replace("--", "-")
    slug = slug.strip("-")[:limit].strip("-")
    return slug or fallback


# --------------------------------------------------------------------------- #
# HTTP (stdlib only, so the core has zero dependencies)
# --------------------------------------------------------------------------- #
def _error_body(err: urllib.error.HTTPError) -> str:
    try:
        return err.read().decode("utf-8", "replace")[:600]
    except Exception:  # pragma: no cover - defensive
        return ""


def http_json(
    url: str,
    payload: dict | None = None,
    headers: dict | None = None,
    timeout: int = 300,
) -> dict:
    body = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json", **(headers or {})},
        method="POST" if body is not None else "GET",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as err:
        raise ClipForgeError(f"HTTP {err.code} from {url}\n{_error_body(err)}") from err
    except urllib.error.URLError as err:
        raise ClipForgeError(
            f"Could not reach {url} ({err.reason}).\n"
            "Check the server is running and the base URL is correct."
        ) from err
    except json.JSONDecodeError as err:
        raise ClipForgeError(f"Server at {url} did not return valid JSON.") from err


def http_multipart(
    url: str,
    fields: dict[str, str],
    files: list[tuple[str, str, bytes, str]],
    headers: dict | None = None,
    timeout: int = 900,
) -> dict:
    """Minimal multipart/form-data POST, used for Whisper audio uploads."""
    boundary = "----ClipForge" + uuid.uuid4().hex
    parts: list[bytes] = []
    for key, value in fields.items():
        parts.append(
            f'--{boundary}\r\nContent-Disposition: form-data; name="{key}"\r\n\r\n{value}\r\n'.encode()
        )
    for field, filename, data, ctype in files:
        parts.append(
            (
                f'--{boundary}\r\nContent-Disposition: form-data; name="{field}"; '
                f'filename="{filename}"\r\nContent-Type: {ctype}\r\n\r\n'
            ).encode()
        )
        parts.append(data)
        parts.append(b"\r\n")
    parts.append(f"--{boundary}--\r\n".encode())
    body = b"".join(parts)

    req = urllib.request.Request(
        url,
        data=body,
        headers={
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "Content-Length": str(len(body)),
            **(headers or {}),
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as err:
        raise ClipForgeError(f"HTTP {err.code} from {url}\n{_error_body(err)}") from err
    except urllib.error.URLError as err:
        raise ClipForgeError(f"Could not reach {url} ({err.reason}).") from err
    except json.JSONDecodeError as err:
        raise ClipForgeError(f"Server at {url} did not return valid JSON.") from err


def extract_json(text: str):
    """Pull a JSON object/array out of an LLM reply that may include prose or fences."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```", 2)[1]
        if text.lstrip().lower().startswith(("json", "javascript")):
            text = text.split("\n", 1)[1] if "\n" in text else text
    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Fall back to the outer-most balanced object/array in the reply.
    for opener, closer in (("{", "}"), ("[", "]")):
        start = text.find(opener)
        if start == -1:
            continue
        depth = 0
        in_str = False
        escape = False
        for idx in range(start, len(text)):
            ch = text[idx]
            if in_str:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == opener:
                depth += 1
            elif ch == closer:
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start : idx + 1])
                    except json.JSONDecodeError:
                        break
    raise ClipForgeError("The model's reply did not contain usable JSON.")
