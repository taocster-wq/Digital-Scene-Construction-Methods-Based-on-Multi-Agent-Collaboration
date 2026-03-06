import os
import re
import logging
from pathlib import Path
from typing import Optional, List

from utils.file_tools import read_error_log

logger = logging.getLogger(__name__)

# -----------------------------
# 1) Project root auto-detection
# -----------------------------
def _guess_project_root() -> Path:
    """
    Best-effort project root resolver (Windows/Linux):
    Priority:
      1) ENV: MANIM_PROJECT_ROOT / PROJECT_ROOT / BASE_DIR
      2) Walk up from this file: find marker files/folders
      3) CWD
    """
    for k in ("MANIM_PROJECT_ROOT", "PROJECT_ROOT", "BASE_DIR"):
        v = os.environ.get(k)
        if v:
            p = Path(v).expanduser()
            if p.exists():
                return p.resolve()

    here = Path(__file__).resolve()
    markers = ["media", "utils", "services", "pyproject.toml", "requirements.txt", ".git"]
    for parent in [here.parent] + list(here.parents):
        if any((parent / m).exists() for m in markers):
            return parent.resolve()

    return Path.cwd().resolve()


# -----------------------------
# 2) TeX error detection
# -----------------------------
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
    s = str(error_message).lower()
    if any(h in s for h in _TEX_HINTS):
        return True
    if ("log file" in s or "the log" in s) and ".log" in s:
        return True
    if re.search(r"media[\\/](?:tex|Tex)[\\/].+?\.log\b", s, flags=re.IGNORECASE):
        return True
    return False


# -----------------------------
# 3) Extract .log path from error message
# -----------------------------
def extract_tex_log_path(error_log: str) -> Optional[str]:
    if not error_log:
        return None

    s = str(error_log)

    # Primary: "log file:" with whitespace/newlines between tokens
    m = re.search(
        r"(?:the\s+)?log\s*file\s*:\s*([^\r\n]+?\.log)\b",
        s,
        flags=re.IGNORECASE,
    )
    if m:
        return m.group(1).strip().strip('"\'').rstrip(".,;:")

    # Secondary: "log ... file:" with DOTALL to allow newline
    m = re.search(
        r"(?:the\s+)?log.*?file\s*:\s*([^\r\n]+?\.log)\b",
        s,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if m:
        return m.group(1).strip().strip('"\'').rstrip(".,;:")

    # Fallback: any token ending with .log (prefer media/Tex)
    candidates = re.findall(
        r"([A-Za-z]:[\\/][^\s\r\n\"']+?\.log\b|(?:media|Media)[\\/][^\s\r\n\"']+?\.log\b|[^\s\r\n\"']+?\.log\b)",
        s,
        flags=re.IGNORECASE,
    )
    if not candidates:
        return None

    for c in candidates:
        if re.search(r"media[\\/](?:Tex|tex)[\\/].+?\.log\b", c, flags=re.IGNORECASE):
            return c.strip().strip('"\'').rstrip(".,;:")

    return candidates[0].strip().strip('"\'').rstrip(".,;:")


# -----------------------------
# 4) Read file with robust decoding
# -----------------------------
def _read_text_with_fallback(p: Path, enc_first: str = "utf-8") -> str:
    """
    utf-8 -> gbk -> utf-8(replace)
    """
    try:
        return p.read_text(encoding=enc_first)
    except UnicodeDecodeError:
        try:
            return p.read_text(encoding="gbk")
        except Exception:
            return p.read_text(encoding="utf-8", errors="replace")


def read_tex_log_from_error_log(
    error_log: str,
    strict: bool = False,
) -> str:
    """
    Parse error_log to find the TeX .log path and return its file content.

    - strict=False: if no TeX .log path is found, return "" (NOT an error).
    - strict=True : raise ValueError if not found.
    """
    path_str = extract_tex_log_path(error_log)
    if not path_str:
        if strict:
            raise ValueError("No TeX .log path found in error_message.")
        return ""

    p = Path(path_str)
    if not p.is_absolute():
        root = _guess_project_root()
        p = (root / p).resolve()

    if not p.exists():
        raise FileNotFoundError(f"TeX log file not found: {p}")

    return _read_text_with_fallback(p)


# -----------------------------
# 5) Summarize tex.log (template-based)
# -----------------------------
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

    # Source .tex path: "**./media/Tex/xxxx.tex"
    src_tex: Optional[str] = None
    for ln in lines[:500]:
        m = re.match(r"^\*\*(.+?\.tex)\s*$", ln.strip())
        if m:
            src_tex = m.group(1).strip()
            break

    # First fatal error: line starts with "! "
    err_idx: Optional[int] = None
    err_title: Optional[str] = None
    for i, ln in enumerate(lines):
        if ln.startswith("! "):
            err_idx = i
            err_title = ln[2:].strip()
            break

    # Location line: "l.<N> ..."
    loc_idx: Optional[int] = None
    loc_line: Optional[str] = None
    if err_idx is not None:
        for j in range(err_idx, min(err_idx + 200, len(lines))):
            if re.match(r"^\s*l\.\d+\s", lines[j]):
                loc_idx = j
                loc_line = lines[j].strip()
                break

    # Error block: from "! " until blank line (or next "! ")
    error_block = ""
    if err_idx is not None:
        block: List[str] = []
        for k in range(err_idx, len(lines)):
            ln = lines[k]
            if k != err_idx and ln.startswith("! "):
                break
            block.append(ln)
            if k > err_idx and ln.strip() == "" and len(block) >= 6:
                break
        error_block = "\n".join(block).rstrip()

    # Context window around location if available
    ctx = ""
    anchor = loc_idx if loc_idx is not None else err_idx
    if anchor is not None:
        ctx = clip(anchor - context_lines, anchor + context_lines + 1)

    # Tail hints
    tail: List[str] = []
    for ln in lines[-160:]:
        if ln.strip().startswith("Here is how much of TeX's memory you used:"):
            tail.append("Here is how much of TeX's memory you used:")
        if ln.strip().startswith("No pages of output."):
            tail.append("No pages of output.")
    tail_hint_text = "\n".join(dict.fromkeys(tail)).strip()

    parts: List[str] = []
    if src_tex:
        parts.append("[SOURCE_TEX]\n" + src_tex)
    parts.append("[PRIMARY_ERROR]\n" + (err_title or "No fatal '! ...' marker found in tex.log."))
    if loc_line:
        parts.append("[ERROR_LOCATION]\n" + loc_line)
    if error_block:
        parts.append("[ERROR_BLOCK]\n" + error_block)
    if ctx:
        parts.append(f"[CONTEXT_{context_lines}]\n" + ctx)
    if tail_hint_text:
        parts.append("[TAIL_HINTS]\n" + tail_hint_text)

    out = "\n\n".join(parts).strip()
    return _cap_text(out, max_chars=max_chars)


def read_tex_log_summary_from_error_log(
    error_log: str,
    strict: bool = False,
    context_lines: int = 6,
    max_chars: int = 2500,
) -> str:
    """
    Return summarized tex.log content (bounded).
    """
    raw = read_tex_log_from_error_log(error_log, strict=strict)
    if not raw:
        return ""
    return summarize_tex_log_from_template(raw, context_lines=context_lines, max_chars=max_chars)


# -----------------------------
# 6) Safety cap
# -----------------------------
def _cap_text(text: str, max_chars: int = 2500) -> str:
    if not text:
        return ""
    s = str(text)
    if len(s) <= max_chars:
        return s
    head = s[: int(max_chars * 0.7)].rstrip()
    tail = s[-int(max_chars * 0.25) :].lstrip()
    return head + "\n\n...[TRUNCATED]...\n\n" + tail

def error_tool(
    error_message: str,
    raw: bool = False,
    context_lines: int = 6,
    max_chars: int = 2500,
) -> str:
    """
    Parse LaTeX/TeX errors and extract TeX .log content.

    Behavior:
    - If error_message is NOT TeX-related: return "" (NOT an error).
    - If TeX-related:
        - raw=False (default): return a compact summary (high-signal, bounded length).
        - raw=True: return the full log content (still bounded by a safety cap).
    """
    try:
        logger.info("Calling error_tool (raw=%s)", raw)

        if not _looks_like_tex_error(error_message):
            return ""

        if raw:
            content = read_tex_log_from_error_log(error_message, strict=False)
            # raw can be larger; still cap to avoid blowing up agent context
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
    error_log =read_error_log()

    result = error_tool(error_log, raw=False)
    print("=== RAW LOG ===")
    print(result)
