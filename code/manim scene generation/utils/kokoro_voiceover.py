"""
Copyright (c) 2025 Xposed73
All rights reserved.
This file is part of the Manim Voiceover project.
"""

import hashlib
import json
import re
import wave
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from kokoro_onnx import Kokoro
from manim_voiceover.services.base import SpeechService
from scipy.io.wavfile import write as write_wav

from config import cfg


# -----------------------------
# Duration patch (WAV-friendly)
# -----------------------------
def _duration_seconds_any_audio(path: Path) -> float:
    """
    Robust duration reader:
    - WAV: Python stdlib wave
    - MP3: mutagen (optional)
    """
    p = Path(path)
    ext = p.suffix.lower()

    if ext == ".wav":
        try:
            with wave.open(str(p), "rb") as w:
                frames = w.getnframes()
                rate = w.getframerate()
                return float(frames) / float(rate) if rate else 0.0
        except Exception:
            return 0.0

    try:
        from mutagen.mp3 import MP3  # type: ignore

        return float(MP3(str(p)).info.length)
    except Exception:
        return 0.0


def _patch_voiceover_tracker_duration():
    """
    manim_voiceover.tracker imports get_duration at module import time.
    Patch the symbol in tracker module directly so WAV works.
    """
    try:
        import manim_voiceover.tracker as tracker_mod  # type: ignore

        tracker_mod.get_duration = lambda path: _duration_seconds_any_audio(Path(path))
    except Exception:
        pass


# -----------------------------
# Voice listing helpers
# -----------------------------
def list_kokoro_voices(
    model_path: str = cfg.KOKORO_MODEL_PATH,
    voices_path: str = cfg.KOKORO_VOICES_PATH,
) -> List[str]:
    """
    Return all available Kokoro voice IDs (sorted, unique).
    """
    kokoro = Kokoro(model_path, voices_path)
    voices = list(getattr(kokoro, "voices", []) or [])
    voices = [str(v).strip() for v in voices if str(v).strip()]
    return sorted(set(voices))


def list_kokoro_voices_grouped(
    model_path: str = cfg.KOKORO_MODEL_PATH,
    voices_path: str = cfg.KOKORO_VOICES_PATH,
) -> Dict[str, List[str]]:
    """
    Return voices grouped by prefix (useful to quickly find zh/en families).
    Example keys: "af", "am", "zf", "zm", ...
    """
    voices = list_kokoro_voices(model_path=model_path, voices_path=voices_path)
    groups: Dict[str, List[str]] = {}
    for v in voices:
        key = v.split("_", 1)[0] if "_" in v else (v[:2] if len(v) >= 2 else v)
        groups.setdefault(key, []).append(v)
    return groups


# -----------------------------
# Text normalization & splitting
# -----------------------------
_CJK_PUNCT = "，。！？；：、"
_EN_PUNCT = ".,!?;:"


def _normalize_text_for_kokoro(text: str, prefer_zh: bool) -> str:
    """
    Language-aware normalization:
    - always: strip + collapse whitespace
    - zh: add spacing after punct; add spacing between CJK and alnum boundaries
    - en: add spacing after English punctuation only
    """
    if text is None:
        return ""
    t = str(text).strip()
    t = re.sub(r"\s+", " ", t)

    if prefer_zh:
        t = re.sub(r"([，。！？；：、])(?=\S)", r"\1 ", t)
        t = re.sub(r"([,.!?;:])(?=\S)", r"\1 ", t)
        t = re.sub(r"([\u4e00-\u9fff])([A-Za-z0-9])", r"\1 \2", t)
        t = re.sub(r"([A-Za-z0-9])([\u4e00-\u9fff])", r"\1 \2", t)
    else:
        t = re.sub(r"([,.!?;:])(?=\S)", r"\1 ", t)

    t = re.sub(r"\s{2,}", " ", t).strip()
    return t


def _split_text_for_tts(text: str, max_chars: int, prefer_zh: bool) -> List[str]:
    """
    Split long text into manageable chunks for BOTH zh/en.
    Prefer splitting at punctuation boundaries, then enforce max length.
    """
    t = _normalize_text_for_kokoro(text, prefer_zh=prefer_zh)
    if not t:
        return []

    punct = _CJK_PUNCT + _EN_PUNCT
    parts = re.split(rf"(?<=[{re.escape(punct)}])\s+", t)
    parts = [p.strip() for p in parts if p.strip()]

    out: List[str] = []
    for p in parts:
        if len(p) <= max_chars:
            out.append(p)
            continue

        buf = p
        while len(buf) > max_chars:
            cut = buf.rfind(" ", 0, max_chars)
            if cut < max_chars * 0.6:
                cut = max_chars
            out.append(buf[:cut].strip())
            buf = buf[cut:].strip()
        if buf:
            out.append(buf)

    return out


# -----------------------------
# Language + voice helpers
# -----------------------------
def _is_chinese_context(lang: str, voice: str) -> bool:
    l = (lang or "").strip().lower()
    v = (voice or "").strip().lower()
    if l in {"z", "zh", "zh-cn", "zh_cn", "cmn", "mandarin", "chinese"}:
        return True
    # Kokoro Chinese voices commonly start with zf_/zm_
    if v.startswith("zf_") or v.startswith("zm_") or v.startswith("z"):
        return True
    return False


def _map_lang_for_espeak(lang: str) -> str:
    """
    phonemizer(espeak) doesn't accept "z". Use ISO 639-3 codes where possible.
    Mandarin is commonly "cmn".
    """
    l = (lang or "").strip().lower()
    if l in {"z", "zh", "zh-cn", "zh_cn"}:
        return "cmn"
    return lang


def _pick_voice(kokoro: Kokoro, requested_voice: str, prefer_zh: bool) -> str:
    voices = list(getattr(kokoro, "voices", []) or [])
    if not voices:
        return str(requested_voice or "")

    req = str(requested_voice or "").strip()
    if req in voices:
        return req

    # Common mismatch: "af" -> "af_alloy"
    if req and (req + "_alloy") in voices:
        return req + "_alloy"

    # Prefix match: "af" matches "af_*"
    if req:
        pref = req + "_"
        for v in voices:
            if v.startswith(pref):
                return v

    # Prefer a Chinese voice if context looks Chinese
    if prefer_zh:
        for v in voices:
            if v.startswith("zf_") or v.startswith("zm_"):
                return v

    # Fallback: first voice
    return voices[0]


# -----------------------------
# KokoroService
# -----------------------------
class KokoroService(SpeechService):
    """
    Speech service class for kokoro_self (using text_to_speech via Kokoro ONNX).

    Key fixes:
    - Output WAV (PCM16) to avoid mp3/av issues on Windows
    - Patch manim_voiceover duration reader for WAV
    - Map lang="z"/"zh" -> "cmn" for espeak backend
    - Optional Chinese G2P via misaki (if installed) to avoid espeak language errors & repetition
    - Validate/auto-fix voice name (e.g., "af" -> "af_alloy")
    """

    def __init__(
        self,
        engine=None,
        model_path: str = cfg.KOKORO_MODEL_PATH,
        voices_path: str = cfg.KOKORO_VOICES_PATH,
        voice: str = cfg.KOKORO_DEFAULT_VOICE,
        speed: float = cfg.KOKORO_DEFAULT_SPEED,
        lang: str = cfg.KOKORO_DEFAULT_LANG,
        **kwargs,
    ):
        _patch_voiceover_tracker_duration()

        self.kokoro = Kokoro(model_path, voices_path)

        # Decide context + map lang early
        self.lang = str(lang or "").strip()
        self.speed = float(speed)
        self._prefer_zh = _is_chinese_context(self.lang, str(voice or ""))

        # Fix lang for phonemizer/espeak
        self._espeak_lang = str(_map_lang_for_espeak(self.lang or "en-us")).strip()
        if not self._espeak_lang:
            self._espeak_lang = "en-us"

        # Pick an available voice
        self.voice = _pick_voice(self.kokoro, str(voice or ""), self._prefer_zh)

        # Optional: misaki zh g2p (only used when Chinese context)
        self._zh_g2p = None
        if self._prefer_zh:
            try:
                from misaki import zh  # type: ignore

                try:
                    self._zh_g2p = zh.ZHG2P(version="1.1")
                except Exception:
                    self._zh_g2p = zh.ZHG2P()
            except Exception:
                self._zh_g2p = None

        if engine is None:
            engine = self.text_to_speech
        self.engine = engine

        super().__init__(**kwargs)

    @staticmethod
    def available_voices(
        model_path: str = cfg.KOKORO_MODEL_PATH,
        voices_path: str = cfg.KOKORO_VOICES_PATH,
    ) -> List[str]:
        """
        Convenience wrapper for listing available Kokoro voices.
        """
        return list_kokoro_voices(model_path=model_path, voices_path=voices_path)

    @staticmethod
    def available_voices_grouped(
        model_path: str = cfg.KOKORO_MODEL_PATH,
        voices_path: str = cfg.KOKORO_VOICES_PATH,
    ) -> Dict[str, List[str]]:
        """
        Convenience wrapper for listing voices grouped by prefix.
        """
        return list_kokoro_voices_grouped(model_path=model_path, voices_path=voices_path)

    def get_data_hash(self, input_data: dict) -> str:
        data_str = json.dumps(input_data, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(data_str.encode("utf-8")).hexdigest()

    def _to_kokoro_inputs(self, text: str) -> Tuple[List[str], bool]:
        """
        Return (chunks, is_phonemes).
        - If Chinese and misaki available: generate phonemes (is_phonemes=True)
        - Else: return normalized text chunks (is_phonemes=False)
        """
        max_chars = 80 if self._prefer_zh else 140
        chunks = _split_text_for_tts(text, max_chars=max_chars, prefer_zh=self._prefer_zh)
        chunks = [c for c in chunks if c]

        if not chunks:
            return [], False

        # Chinese phonemes path (recommended)
        if self._prefer_zh and self._zh_g2p is not None:
            phoneme_chunks: List[str] = []
            for c in chunks:
                try:
                    ph, _ = self._zh_g2p(c)
                    ph = str(ph).strip()
                    if ph:
                        phoneme_chunks.append(ph)
                except Exception:
                    # If g2p fails for a chunk, fallback to text
                    return chunks, False
            if phoneme_chunks:
                return phoneme_chunks, True

        return chunks, False

    def text_to_speech(self, text, output_file, voice_name, speed, lang):
        """
        Generates speech from text using Kokoro ONNX and saves WAV (PCM16).
        For Chinese: prefer misaki phonemes when available.
        """
        chunks, is_phonemes = self._to_kokoro_inputs(text)

        # If no chunks, write a short silence wav to avoid downstream failures
        if not chunks:
            sr = 24000
            silence = np.zeros(int(0.2 * sr), dtype=np.int16)
            write_wav(output_file, sr, silence)
            return output_file

        all_samples: List[np.ndarray] = []
        sr_final: Optional[int] = None

        for idx, c in enumerate(chunks):
            if is_phonemes:
                samples, sample_rate = self.kokoro.create(
                    c,
                    voice=str(voice_name),
                    speed=float(speed),
                    lang=str(self._espeak_lang),
                    is_phonemes=True,
                )
            else:
                # IMPORTANT: use mapped espeak lang (avoid "z")
                samples, sample_rate = self.kokoro.create(
                    c,
                    voice=str(voice_name),
                    speed=float(speed),
                    lang=str(self._espeak_lang),
                )

            sr = int(sample_rate)
            if sr_final is None:
                sr_final = sr

            # Normalize to [-1, 1]
            samples = np.asarray(samples, dtype=np.float32)
            max_val = float(np.max(np.abs(samples))) if samples.size else 0.0
            if max_val > 0:
                samples = samples / max_val

            # Convert to PCM16
            pcm = (samples * 32767.0).astype(np.int16)

            # Add a tiny silence gap between chunks (reduces "stuck"/repeat feeling)
            if idx > 0 and sr_final:
                gap = np.zeros(int(0.06 * sr_final), dtype=np.int16)
                all_samples.append(gap)

            all_samples.append(pcm)

        sr_final = sr_final or 24000
        merged = np.concatenate(all_samples) if all_samples else np.zeros(int(0.2 * sr_final), dtype=np.int16)

        write_wav(output_file, sr_final, merged)
        return output_file

    def generate_from_text(self, text: str, cache_dir: str = None, path: str = None) -> dict:
        if cache_dir is None:
            cache_dir = self.cache_dir

        norm_text = _normalize_text_for_kokoro(text, prefer_zh=self._prefer_zh)

        # IMPORTANT: unique service name to avoid old mp3 cache collisions
        input_data = {
            "input_text": norm_text,
            "service": "kokoro_self_wav_v2",
            "voice": self.voice,
            "lang": self.lang,
            "espeak_lang": self._espeak_lang,
            "speed": float(self.speed),
            "prefer_zh": bool(self._prefer_zh),
            "use_misaki": bool(self._prefer_zh and self._zh_g2p is not None),
        }

        cached = self.get_cached_result(input_data, cache_dir)
        if cached is not None:
            if "final_audio" not in cached and "original_audio" in cached:
                cached["final_audio"] = cached["original_audio"]
            return cached

        # Always WAV
        if path is None:
            audio_name = self.get_data_hash(input_data) + ".wav"
        else:
            audio_name = str(path)
            if not audio_name.lower().endswith(".wav"):
                audio_name = audio_name.rsplit(".", 1)[0] + ".wav"

        audio_path_wav = str(Path(cache_dir) / audio_name)

        # Generate WAV
        self.engine(
            text=norm_text,
            output_file=audio_path_wav,
            voice_name=self.voice,
            speed=self.speed,
            lang=self.lang,
        )

        return {
            "input_text": norm_text,
            "input_data": input_data,
            "original_audio": audio_name,
            "final_audio": audio_name,  # manim_voiceover uses this
        }


# ---- Optional quick test (manual) ----
# if __name__ == "__main__":
#     print("Available voices:", KokoroService.available_voices())
#     print("Grouped voices:", KokoroService.available_voices_grouped())
