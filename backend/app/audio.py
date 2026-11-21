"""音声の変換・読み込み・前処理（ffmpeg + soundfile）。"""
from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np

from .config import ffmpeg_path

ANALYSIS_SR = 22050


class AudioError(Exception):
    pass


def _run_ffmpeg(args: list[str]) -> None:
    ff = ffmpeg_path()
    if not ff:
        raise AudioError("ffmpeg が見つかりません。")
    cmd = [ff, "-hide_banner", "-loglevel", "error", "-y", *args]
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if proc.returncode != 0:
        raise AudioError(f"ffmpeg でエラー: {proc.stderr.strip()[:400]}")


def decode_to_wav(
    src: Path,
    dst: Path,
    sr: int = 44100,
    start: float | None = None,
    end: float | None = None,
) -> None:
    """どんな動画・音声でも 44.1kHz ステレオ 16bit WAV にする。区間指定も可。"""
    args: list[str] = []
    if start and start > 0:
        args += ["-ss", f"{start:.3f}"]
    args += ["-i", str(src)]
    if end and end > (start or 0):
        args += ["-t", f"{end - (start or 0):.3f}"]
    args += ["-vn", "-ac", "2", "-ar", str(sr), "-c:a", "pcm_s16le", str(dst)]
    _run_ffmpeg(args)
    if not dst.exists() or dst.stat().st_size < 1000:
        raise AudioError("音声トラックが見つかりませんでした。")


def encode_mp3(src: Path, dst: Path, bitrate: str = "192k") -> None:
    _run_ffmpeg(["-i", str(src), "-vn", "-c:a", "libmp3lame", "-b:a", bitrate, str(dst)])


def read(path: Path) -> tuple[np.ndarray, int]:
    """(channels, samples) の float32 と sr を返す。"""
    import soundfile as sf  # ブラウザ版（Pyodide）には無いので使う時に読む

    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    return data.T.copy(), sr


def write(path: Path, data: np.ndarray, sr: int) -> None:
    import soundfile as sf

    arr = data.T if data.ndim == 2 else data
    sf.write(str(path), np.clip(arr, -1.0, 1.0), sr, subtype="PCM_16")


def to_mono(data: np.ndarray) -> np.ndarray:
    return data.mean(axis=0) if data.ndim == 2 else data


def resample(y: np.ndarray, sr: int, target: int) -> np.ndarray:
    if sr == target:
        return y.astype(np.float32)
    import librosa

    return librosa.resample(y.astype(np.float32), orig_sr=sr, target_sr=target, res_type="soxr_hq")


def load_mono(path: Path, target_sr: int = ANALYSIS_SR) -> np.ndarray:
    data, sr = read(path)
    return resample(to_mono(data), sr, target_sr)


def trim_silence(data: np.ndarray, sr: int, threshold_db: float = -50.0, pad: float = 0.15) -> tuple[np.ndarray, float]:
    """前後の無音を切る。切った先頭の秒数も返す（元動画との時刻合わせ用）。"""
    mono = np.abs(to_mono(data))
    if mono.size == 0:
        return data, 0.0
    peak = float(mono.max()) or 1.0
    thr = peak * (10 ** (threshold_db / 20))
    win = max(1, int(sr * 0.02))
    # 20ms 窓の最大値で判定（クリック雑音に引っぱられにくい）
    n = mono.size // win
    env = mono[: n * win].reshape(n, win).max(axis=1) if n else mono
    idx = np.where(env > thr)[0]
    if idx.size == 0:
        return data, 0.0
    a = max(0, idx[0] * win - int(pad * sr))
    b = min(mono.size, (idx[-1] + 1) * win + int(pad * sr))
    return data[..., a:b], a / sr


def normalize(data: np.ndarray, peak_db: float = -1.0) -> np.ndarray:
    peak = float(np.abs(data).max()) if data.size else 0.0
    if peak < 1e-6:
        return data
    return data * (10 ** (peak_db / 20) / peak)


def rms_db(y: np.ndarray) -> float:
    r = float(np.sqrt(np.mean(np.square(y)))) if y.size else 0.0
    return 20 * np.log10(max(r, 1e-9))
