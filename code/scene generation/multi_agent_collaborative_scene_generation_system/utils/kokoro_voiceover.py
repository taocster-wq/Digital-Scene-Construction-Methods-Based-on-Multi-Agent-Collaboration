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


def _duration_seconds_any_audio(path: Path) -> float:
    path = Path(path)
    ext = path.suffix.lower()

    if ext == ".wav":
        try:
            with wave.open(str(path), "rb") as wav_file:
                frames = wav_file.getnframes()
                rate = wav_file.getframerate()
                return float(frames) / float(rate) if rate else 0.0
        except Exception:
            return 0.0

    try:
        from mutagen.mp3 import MP3

        return float(MP3(str(path)).info.length)
    except Exception:
        return 0.0


def _patch_voiceover_tracker_duration():
    try:
        import manim_voiceover.tracker as tracker_mod

        tracker_mod.get_duration = lambda path: _duration_seconds_any_audio(Path(path))
    except Exception:
        pass


def list_kokoro_voices(
    model_path: str = cfg.KOKORO_MODEL_PATH,
    voices_path: str = cfg.KOKORO_VOICES_PATH,
) -> List[str]:
    kokoro = Kokoro(model_path, voices_path)
    voices = list(getattr(kokoro, "voices", []) or [])
    voices = [str(v).strip() for v in voices if str(v).strip()]
    return sorted(set(voices))


def list_kokoro_voices_grouped(
    model_path: str = cfg.KOKORO_MODEL_PATH,
    voices_path: str = cfg.KOKORO_VOICES_PATH,
) -> Dict[str, List[str]]:
    voices = list_kokoro_voices(
        model_path=model_path,
        voices_path=voices_path,
    )

    groups: Dict[str, List[str]] = {}

    for voice in voices:
        key = (
            voice.split("_", 1)[0]
            if "_" in voice
            else voice[:2]
            if len(voice) >= 2
            else voice
        )

        groups.setdefault(key, []).append(voice)

    return groups


_CJK_PUNCT = "，。！？；：、"
_EN_PUNCT = ".,!?;:"


def _normalize_text_for_kokoro(text: str, prefer_zh: bool) -> str:
    if text is None:
        return ""

    value = str(text).strip()
    value = re.sub(r"\s+", " ", value)

    if prefer_zh:
        value = re.sub(r"([，。！？；：、])(?=\S)", r"\1 ", value)
        value = re.sub(r"([,.!?;:])(?=\S)", r"\1 ", value)
        value = re.sub(r"([\u4e00-\u9fff])([A-Za-z0-9])", r"\1 \2", value)
        value = re.sub(r"([A-Za-z0-9])([\u4e00-\u9fff])", r"\1 \2", value)
    else:
        value = re.sub(r"([,.!?;:])(?=\S)", r"\1 ", value)

    value = re.sub(r"\s{2,}", " ", value).strip()
    return value


def _split_text_for_tts(
    text: str,
    max_chars: int,
    prefer_zh: bool,
) -> List[str]:
    value = _normalize_text_for_kokoro(text, prefer_zh=prefer_zh)

    if not value:
        return []

    punct = _CJK_PUNCT + _EN_PUNCT
    parts = re.split(rf"(?<=[{re.escape(punct)}])\s+", value)
    parts = [part.strip() for part in parts if part.strip()]

    output: List[str] = []

    for part in parts:
        if len(part) <= max_chars:
            output.append(part)
            continue

        buffer = part

        while len(buffer) > max_chars:
            cut = buffer.rfind(" ", 0, max_chars)

            if cut < max_chars * 0.6:
                cut = max_chars

            output.append(buffer[:cut].strip())
            buffer = buffer[cut:].strip()

        if buffer:
            output.append(buffer)

    return output


def _is_chinese_context(lang: str, voice: str) -> bool:
    lang = (lang or "").strip().lower()
    voice = (voice or "").strip().lower()

    if lang in {"z", "zh", "zh-cn", "zh_cn", "cmn", "mandarin", "chinese"}:
        return True

    if voice.startswith("zf_") or voice.startswith("zm_") or voice.startswith("z"):
        return True

    return False


def _map_lang_for_espeak(lang: str) -> str:
    lang = (lang or "").strip().lower()

    if lang in {"z", "zh", "zh-cn", "zh_cn"}:
        return "cmn"

    return lang


def _pick_voice(kokoro: Kokoro, requested_voice: str, prefer_zh: bool) -> str:
    voices = list(getattr(kokoro, "voices", []) or [])

    if not voices:
        return str(requested_voice or "")

    requested = str(requested_voice or "").strip()

    if requested in voices:
        return requested

    if requested and (requested + "_alloy") in voices:
        return requested + "_alloy"

    if requested:
        prefix = requested + "_"

        for voice in voices:
            if voice.startswith(prefix):
                return voice

    if prefer_zh:
        for voice in voices:
            if voice.startswith("zf_") or voice.startswith("zm_"):
                return voice

    return voices[0]


class KokoroService(SpeechService):
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

        self.lang = str(lang or "").strip()
        self.speed = float(speed)
        self._prefer_zh = _is_chinese_context(self.lang, str(voice or ""))

        self._espeak_lang = str(_map_lang_for_espeak(self.lang or "en-us")).strip()

        if not self._espeak_lang:
            self._espeak_lang = "en-us"

        self.voice = _pick_voice(
            self.kokoro,
            str(voice or ""),
            self._prefer_zh,
        )

        self._zh_g2p = None

        if self._prefer_zh:
            try:
                from misaki import zh

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
        return list_kokoro_voices(
            model_path=model_path,
            voices_path=voices_path,
        )

    @staticmethod
    def available_voices_grouped(
        model_path: str = cfg.KOKORO_MODEL_PATH,
        voices_path: str = cfg.KOKORO_VOICES_PATH,
    ) -> Dict[str, List[str]]:
        return list_kokoro_voices_grouped(
            model_path=model_path,
            voices_path=voices_path,
        )

    def get_data_hash(self, input_data: dict) -> str:
        data_str = json.dumps(
            input_data,
            sort_keys=True,
            ensure_ascii=False,
        )
        return hashlib.sha256(data_str.encode("utf-8")).hexdigest()

    def _to_kokoro_inputs(self, text: str) -> Tuple[List[str], bool]:
        max_chars = 80 if self._prefer_zh else 140

        chunks = _split_text_for_tts(
            text,
            max_chars=max_chars,
            prefer_zh=self._prefer_zh,
        )
        chunks = [chunk for chunk in chunks if chunk]

        if not chunks:
            return [], False

        if self._prefer_zh and self._zh_g2p is not None:
            phoneme_chunks: List[str] = []

            for chunk in chunks:
                try:
                    phoneme, _ = self._zh_g2p(chunk)
                    phoneme = str(phoneme).strip()

                    if phoneme:
                        phoneme_chunks.append(phoneme)
                except Exception:
                    return chunks, False

            if phoneme_chunks:
                return phoneme_chunks, True

        return chunks, False

    def text_to_speech(self, text, output_file, voice_name, speed, lang):
        chunks, is_phonemes = self._to_kokoro_inputs(text)

        if not chunks:
            sample_rate = 24000
            silence = np.zeros(int(0.2 * sample_rate), dtype=np.int16)
            write_wav(output_file, sample_rate, silence)
            return output_file

        all_samples: List[np.ndarray] = []
        sample_rate_final: Optional[int] = None

        for idx, chunk in enumerate(chunks):
            if is_phonemes:
                samples, sample_rate = self.kokoro.create(
                    chunk,
                    voice=str(voice_name),
                    speed=float(speed),
                    lang=str(self._espeak_lang),
                    is_phonemes=True,
                )
            else:
                samples, sample_rate = self.kokoro.create(
                    chunk,
                    voice=str(voice_name),
                    speed=float(speed),
                    lang=str(self._espeak_lang),
                )

            sample_rate = int(sample_rate)

            if sample_rate_final is None:
                sample_rate_final = sample_rate

            samples = np.asarray(samples, dtype=np.float32)
            max_val = float(np.max(np.abs(samples))) if samples.size else 0.0

            if max_val > 0:
                samples = samples / max_val

            pcm = (samples * 32767.0).astype(np.int16)

            if idx > 0 and sample_rate_final:
                gap = np.zeros(int(0.06 * sample_rate_final), dtype=np.int16)
                all_samples.append(gap)

            all_samples.append(pcm)

        sample_rate_final = sample_rate_final or 24000

        merged = (
            np.concatenate(all_samples)
            if all_samples
            else np.zeros(int(0.2 * sample_rate_final), dtype=np.int16)
        )

        write_wav(output_file, sample_rate_final, merged)
        return output_file

    def generate_from_text(
        self,
        text: str,
        cache_dir: str = None,
        path: str = None,
    ) -> dict:
        if cache_dir is None:
            cache_dir = self.cache_dir

        norm_text = _normalize_text_for_kokoro(
            text,
            prefer_zh=self._prefer_zh,
        )

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

        if path is None:
            audio_name = self.get_data_hash(input_data) + ".wav"
        else:
            audio_name = str(path)

            if not audio_name.lower().endswith(".wav"):
                audio_name = audio_name.rsplit(".", 1)[0] + ".wav"

        audio_path_wav = str(Path(cache_dir) / audio_name)

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
            "final_audio": audio_name,
        }