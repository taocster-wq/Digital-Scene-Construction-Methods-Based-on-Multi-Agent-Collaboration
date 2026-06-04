import os
import re
import logging
from pathlib import Path
from typing import Optional, List

from multi_agent_collaborative_scene_generation_system.utils.file_tools import (
    read_error_log,
)


logger = logging.getLogger(__name__)


def _guess_project_root() -> Path:
    for key in ("MANIM_PROJECT_ROOT", "PROJECT_ROOT", "BASE_DIR"):
        value = os.environ.get(key)

        if value:
            path = Path(value).expanduser()

            if path.exists():
                return path.resolve()

    here = Path(__file__).resolve()
    markers = [
        "media",
        "utils",
        "services",
        "pyproject.toml",
        "requirements.txt",
        ".git",
    ]

    for parent in [here.parent] + list(here.parents):
        if any((parent / marker).exists() for marker in markers):
            return parent.resolve()

    return Path.cwd().resolve()


_TEX_HINTS = (
    "latex error converting to dvi",
    "valueerror: latex error",
    "tex error",
    "dvisvgm",
    "miktex",
    "pdftex",
)


def _looks_like_tex_error(error_message: str) -> bool:
    if not error_message:
        return False

    text = str(error_message).lower()

    if any(hint in text for hint in _TEX_HINTS):
        return True

    if ("log file" in text or "the log" in text) and ".log" in text:
        return True

    if re.search(
        r"media[\\/](?:tex|Tex)[\\/].+?\.log\b",
        text,
        flags=re.IGNORECASE,
    ):
        return True

    return False


def extract_tex_log_path(error_log: str) -> Optional[str]:
    if not error_log:
        return None

    text = str(error_log)

    match = re.search(
        r"(?:the\s+)?log\s*file\s*:\s*([^\r\n]+?\.log)\b",
        text,
        flags=re.IGNORECASE,
    )

    if match:
        return match.group(1).strip().strip('"\'').rstrip(".,;:")

    match = re.search(
        r"(?:the\s+)?log.*?file\s*:\s*([^\r\n]+?\.log)\b",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )

    if match:
        return match.group(1).strip().strip('"\'').rstrip(".,;:")

    candidates = re.findall(
        r"([A-Za-z]:[\\/][^\s\r\n\"']+?\.log\b|(?:media|Media)[\\/][^\s\r\n\"']+?\.log\b|[^\s\r\n\"']+?\.log\b)",
        text,
        flags=re.IGNORECASE,
    )

    if not candidates:
        return None

    for candidate in candidates:
        if re.search(
            r"media[\\/](?:Tex|tex)[\\/].+?\.log\b",
            candidate,
            flags=re.IGNORECASE,
        ):
            return candidate.strip().strip('"\'').rstrip(".,;:")

    return candidates[0].strip().strip('"\'').rstrip(".,;:")


def _read_text_with_fallback(path: Path, enc_first: str = "utf-8") -> str:
    try:
        return path.read_text(encoding=enc_first)
    except UnicodeDecodeError:
        try:
            return path.read_text(encoding="gbk")
        except Exception:
            return path.read_text(encoding="utf-8", errors="replace")


def read_tex_log_from_error_log(
    error_log: str,
    strict: bool = False,
) -> str:
    path_str = extract_tex_log_path(error_log)

    if not path_str:
        if strict:
            raise ValueError("No TeX .log path found in error_message.")
        return ""

    path = Path(path_str)

    if not path.is_absolute():
        root = _guess_project_root()
        path = (root / path).resolve()

    if not path.exists():
        raise FileNotFoundError(f"TeX log file not found: {path}")

    return _read_text_with_fallback(path)


def summarize_tex_log_from_template(
    tex_log: str,
    context_lines: int = 6,
    max_chars: int = 2500,
) -> str:
    if not tex_log:
        return ""

    lines = str(tex_log).splitlines()

    def clip(start: int, end: int) -> str:
        start = max(0, start)
        end = min(len(lines), end)
        return "\n".join(lines[start:end]).rstrip()

    src_tex: Optional[str] = None

    for line in lines[:500]:
        match = re.match(r"^\*\*(.+?\.tex)\s*$", line.strip())

        if match:
            src_tex = match.group(1).strip()
            break

    err_idx: Optional[int] = None
    err_title: Optional[str] = None

    for i, line in enumerate(lines):
        if line.startswith("! "):
            err_idx = i
            err_title = line[2:].strip()
            break

    loc_idx: Optional[int] = None
    loc_line: Optional[str] = None

    if err_idx is not None:
        for j in range(err_idx, min(err_idx + 200, len(lines))):
            if re.match(r"^\s*l\.\d+\s", lines[j]):
                loc_idx = j
                loc_line = lines[j].strip()
                break

    error_block = ""

    if err_idx is not None:
        block: List[str] = []

        for k in range(err_idx, len(lines)):
            line = lines[k]

            if k != err_idx and line.startswith("! "):
                break

            block.append(line)

            if k > err_idx and line.strip() == "" and len(block) >= 6:
                break

        error_block = "\n".join(block).rstrip()

    ctx = ""
    anchor = loc_idx if loc_idx is not None else err_idx

    if anchor is not None:
        ctx = clip(anchor - context_lines, anchor + context_lines + 1)

    tail: List[str] = []

    for line in lines[-160:]:
        stripped = line.strip()

        if stripped.startswith("Here is how much of TeX's memory you used:"):
            tail.append("Here is how much of TeX's memory you used:")

        if stripped.startswith("No pages of output."):
            tail.append("No pages of output.")

    tail_hint_text = "\n".join(dict.fromkeys(tail)).strip()

    parts: List[str] = []

    if src_tex:
        parts.append("[SOURCE_TEX]\n" + src_tex)

    parts.append(
        "[PRIMARY_ERROR]\n"
        + (err_title or "No fatal '! ...' marker found in tex.log.")
    )

    if loc_line:
        parts.append("[ERROR_LOCATION]\n" + loc_line)

    if error_block:
        parts.append("[ERROR_BLOCK]\n" + error_block)

    if ctx:
        parts.append(f"[CONTEXT_{context_lines}]\n" + ctx)

    if tail_hint_text:
        parts.append("[TAIL_HINTS]\n" + tail_hint_text)

    output = "\n\n".join(parts).strip()
    return _cap_text(output, max_chars=max_chars)


def read_tex_log_summary_from_error_log(
    error_log: str,
    strict: bool = False,
    context_lines: int = 6,
    max_chars: int = 2500,
) -> str:
    raw = read_tex_log_from_error_log(error_log, strict=strict)

    if not raw:
        return ""

    return summarize_tex_log_from_template(
        raw,
        context_lines=context_lines,
        max_chars=max_chars,
    )


def _cap_text(text: str, max_chars: int = 2500) -> str:
    if not text:
        return ""

    value = str(text)

    if len(value) <= max_chars:
        return value

    head = value[: int(max_chars * 0.7)].rstrip()
    tail = value[-int(max_chars * 0.25):].lstrip()

    return head + "\n\n...[TRUNCATED]...\n\n" + tail


def error_tool(
    error_message: str,
    raw: bool = False,
    context_lines: int = 6,
    max_chars: int = 2500,
) -> str:
    try:
        logger.info("Calling error_tool (raw=%s)", raw)

        if not _looks_like_tex_error(error_message):
            return ""

        if raw:
            content = read_tex_log_from_error_log(error_message, strict=False)
            return _cap_text(content, max_chars=max(max_chars, 6000))

        summary = read_tex_log_summary_from_error_log(
            error_message,
            strict=False,
            context_lines=context_lines,
            max_chars=max_chars,
        )

        return summary

    except FileNotFoundError as e:
        logger.exception("TeX log file not found: %s", e)
        return f"[ERROR] TeX log file not found: {e}"

    except Exception as e:
        logger.exception("Failed to read TeX log: %s", e)
        return f"[ERROR] Failed to read TeX log: {e}"


if __name__ == "__main__":
    error_log = read_error_log()

    result = error_tool(error_log, raw=False)
    print("=== RAW LOG ===")
    print(result)