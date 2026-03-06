# utils/command_tools.py
from __future__ import annotations

import os
import sys
import time
import logging
import subprocess
import traceback
from pathlib import Path
from fastapi import HTTPException

logger = logging.getLogger(__name__)


def _pick_quality_flag(quality: str) -> str:
    q = (quality or "").lower()
    if q == "high":
        return "-qh"
    if q == "medium":
        return "-qm"
    return "-ql"


def _find_project_root(scene_file: Path) -> Path:
    """
    从 scene 文件路径向上找项目根：包含 utils/__init__.py 的目录。
    """
    scene_file = scene_file.resolve()
    for p in [scene_file.parent] + list(scene_file.parents):
        if (p / "utils" / "__init__.py").exists():
            return p
    return scene_file.parent


def start_manim_command(quality, scene_code_file_path, class_name, err_message):
    """
    参数签名保持不变。

    - 只要函数被调用到，就一定在 project_root 下生成 error.log（覆盖写，不追加）
    - 任何异常（包括 manim 没启动）都会写入 error.log
    - stdout/stderr 二进制落盘，避免编码问题
    - env 快照写入 error.mcp.env.log
    - 返回 error_message：
        * 成功返回 None
        * 失败返回 str（包含 log 路径/最后若干行）
    """
    def _tail_bytes(path: Path, max_bytes: int = 12000) -> bytes:
        try:
            if not path.exists():
                return b""
            size = path.stat().st_size
            with open(path, "rb") as f:
                if size > max_bytes:
                    f.seek(size - max_bytes)
                return f.read()
        except Exception:
            return b""

    def _make_error_message(prefix: str, log_path: Path, extra: str = "") -> str:
        tail = _tail_bytes(log_path)
        tail_text = tail.decode("utf-8", "replace") if tail else ""
        parts = []
        if prefix:
            parts.append(prefix)
        if extra:
            parts.append(extra)
        parts.append(f"See error.log at: {log_path}")
        if tail_text.strip():
            parts.append("---- error.log tail ----\n" + tail_text.strip())
        return "\n".join(parts)

    scene_path = Path(scene_code_file_path).resolve()
    if not scene_path.exists():
        return f"{err_message}: scene file not found: {scene_path}"

    project_root = _find_project_root(scene_path)
    log_path = project_root / "error.log"
    envlog_path = project_root / "error.mcp.env.log"

    try:
        project_root.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        return f"{err_message}: cannot mkdir project_root={project_root}: {e}"

    qflag = _pick_quality_flag(quality)

    cmd = [
        sys.executable, "-m", "manim",
        qflag,
        "--save_sections",
        str(scene_path),
        str(class_name),
    ]

    env = os.environ.copy()
    old_pp = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(project_root) + (os.pathsep + old_pp if old_pp else "")

    # env 快照（失败不影响主流程）
    try:
        with open(envlog_path, "w", encoding="utf-8", errors="replace") as f:
            f.write("=== Environment Snapshot ===\n")
            f.write(f"time={time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"scene_path={scene_path}\n")
            f.write(f"project_root={project_root}\n")
            f.write(f"cwd_for_subprocess={project_root}\n")
            f.write(f"sys.executable={sys.executable}\n")
            f.write(f"cmd={' '.join(cmd)}\n")
            f.write(f"PYTHONPATH(injected)={env.get('PYTHONPATH','')}\n")
            f.write(f"PATH(head)={(env.get('PATH','')[:300] + '...') if env.get('PATH') else ''}\n")
            f.write(f"platform={sys.platform}\n")
            f.write("============================\n")
    except Exception:
        pass

    logger.info("Running Manim command: %s", " ".join(cmd))
    logger.info("project_root=%s log_path=%s", str(project_root), str(log_path))

    # ✅ 覆盖写入（wb），不追加
    try:
        with open(log_path, "wb") as f:
            # 先写入哨兵信息
            f.write(b"=== error.log created (start_manim_command entered) ===\n")
            f.write(f"time={time.strftime('%Y-%m-%d %H:%M:%S')}\n".encode("utf-8", "replace"))
            f.write(f"scene_path={scene_path}\n".encode("utf-8", "replace"))
            f.write(f"project_root={project_root}\n".encode("utf-8", "replace"))
            f.write(f"class_name={class_name}\n".encode("utf-8", "replace"))
            f.write(b"========================================================\n\n")

            f.write(b"--- running manim ---\n")
            f.write(f"cmd={' '.join(cmd)}\n".encode("utf-8", "replace"))
            f.write(b"\n")
            f.flush()

            proc = subprocess.run(
                cmd,
                cwd=str(project_root),
                env=env,
                stdout=f,
                stderr=subprocess.STDOUT,
                timeout=6000,
                check=False,
                text=False,
            )

            f.write(b"\n--- manim finished ---\n")
            f.write(f"returncode={proc.returncode}\n".encode("utf-8", "replace"))
            f.write(b"----------------------\n")
            f.flush()

        if proc.returncode != 0:
            return _make_error_message(
                prefix=err_message,
                log_path=log_path,
                extra=f"(manim failed, code={proc.returncode})",
            )

        return None

    except Exception as e:
        # 异常也写进 error.log（这里用 ab 追加到“本次新文件”的末尾即可）
        try:
            with open(log_path, "ab") as f:
                f.write(b"\n=== Exception in start_manim_command ===\n")
                f.write(str(e).encode("utf-8", "replace") + b"\n\n")
                f.write(traceback.format_exc().encode("utf-8", "replace"))
                f.write(b"\n======================================\n")
                f.flush()
        except Exception:
            pass

        return _make_error_message(
            prefix=err_message,
            log_path=log_path,
            extra=f"(exception: {e})",
        )


