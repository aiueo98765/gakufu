"""音源分離（Stem 分離）と、曲の種類（ピアノ演奏か歌ものか）の判定。

- Demucs htdemucs_6s: drums / bass / other / vocals / guitar / piano の6本に分ける。
  品質プリセットで shifts（時間ずらしアンサンブル）と overlap を変える。
- Demucs が無い環境では librosa の HPSS（調波/打楽器分離）で打楽器だけ落とす。
"""
from __future__ import annotations

import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Callable

import numpy as np

from . import audio as A
from .config import torch_device

QUALITY = {
    # shifts: ランダムに時間をずらして推論した結果の平均回数。増やすほど分離が澄むが比例して遅い。
    "fast": {"shifts": 0, "overlap": 0.25, "label": "高速"},
    "high": {"shifts": 2, "overlap": 0.25, "label": "高品質"},
    "ultra": {"shifts": 5, "overlap": 0.5, "label": "最高（Virtuoso 用）"},
}
MODEL_NAME = "htdemucs_6s"

_lock = threading.Lock()
_model = None


def demucs_available() -> bool:
    try:
        import demucs.apply  # noqa: F401
        import torch  # noqa: F401

        return True
    except Exception:
        return False


def _get_model():
    global _model
    if _model is None:
        from demucs.pretrained import get_model

        _model = get_model(MODEL_NAME)
        _model.eval()
    return _model


def separate_demucs(
    mix_wav: Path,
    out_dir: Path,
    quality: str = "high",
    progress: Callable[[float, str], None] | None = None,
) -> dict[str, Path]:
    import torch
    import demucs.apply as dapply

    q = QUALITY.get(quality, QUALITY["high"])
    out_dir.mkdir(parents=True, exist_ok=True)
    data, sr = A.read(mix_wav)

    with _lock:
        model = _get_model()
        if sr != model.samplerate:
            data = np.stack([A.resample(c, sr, model.samplerate) for c in data])
            sr = model.samplerate
        if data.shape[0] == 1:
            data = np.repeat(data, 2, axis=0)
        wav = torch.from_numpy(data[:2].copy())
        ref = wav.mean(0)
        mean, std = ref.mean(), ref.std() + 1e-8
        wav = (wav - mean) / std

        # demucs の進捗表示(tqdm)を横取りして自分の進捗に流す
        passes = max(1, q["shifts"])
        state = {"pass": 0}

        def fake_tqdm(iterable, **_kw):
            items = list(iterable)
            n = len(items) or 1
            for i, it in enumerate(items):
                if progress:
                    frac = (state["pass"] + i / n) / passes
                    progress(min(frac, 0.99), f"音源分離 {MODEL_NAME}（{q['label']}）")
                yield it
            state["pass"] += 1

        orig_tqdm = dapply.tqdm
        dapply.tqdm = SimpleNamespace(tqdm=fake_tqdm)
        try:
            with torch.inference_mode():
                sources = dapply.apply_model(
                    model,
                    wav[None],
                    device=torch_device(),
                    shifts=q["shifts"],
                    split=True,
                    overlap=q["overlap"],
                    progress=True,
                    num_workers=0,
                )[0]
        finally:
            dapply.tqdm = orig_tqdm
        sources = (sources * std + mean).cpu().numpy()
        if torch_device() == "cuda":
            torch.cuda.empty_cache()

    paths: dict[str, Path] = {}
    for name, src in zip(model.sources, sources):
        p = out_dir / f"{name}.wav"
        A.write(p, src, sr)
        paths[name] = p
    if progress:
        progress(1.0, "音源分離 完了")
    return paths


def separate_hpss(mix_wav: Path, out_dir: Path, progress=None) -> dict[str, Path]:
    """Demucs が使えない時の代替。打楽器成分だけを取り除く。"""
    import librosa

    out_dir.mkdir(parents=True, exist_ok=True)
    data, sr = A.read(mix_wav)
    harm = []
    perc = []
    for i, ch in enumerate(data):
        if progress:
            progress(i / len(data), "調波/打楽器分離 (HPSS)")
        D = librosa.stft(ch, n_fft=4096, hop_length=1024)
        H, P = librosa.decompose.hpss(D, margin=(1.0, 2.0))
        harm.append(librosa.istft(H, hop_length=1024, length=ch.size))
        perc.append(librosa.istft(P, hop_length=1024, length=ch.size))
    paths = {"harmonic": out_dir / "harmonic.wav", "drums": out_dir / "drums.wav"}
    A.write(paths["harmonic"], np.stack(harm), sr)
    A.write(paths["drums"], np.stack(perc), sr)
    return paths


def stem_energies(stems: dict[str, Path]) -> dict[str, float]:
    """各 Stem の平均パワー（相対比較用）。"""
    out = {}
    for name, p in stems.items():
        d, _ = A.read(p)
        out[name] = float(np.mean(np.square(d)))
    return out


def classify_source(energies: dict[str, float]) -> dict:
    """ピアノ独奏 / ピアノ入りバンド / 歌もの / 器楽 を判定する。"""
    tonal_keys = [k for k in ("piano", "other", "guitar", "vocals", "bass") if k in energies]
    total = sum(energies.get(k, 0.0) for k in tonal_keys) + 1e-12
    ratio = {k: energies.get(k, 0.0) / total for k in tonal_keys}
    drums = energies.get("drums", 0.0) / (total + energies.get("drums", 0.0))
    piano = ratio.get("piano", 0.0)
    vocals = ratio.get("vocals", 0.0)
    if "piano" not in ratio:  # HPSS のみ
        kind = "piano"
    elif piano >= 0.55 and vocals < 0.12 and drums < 0.08:
        kind = "piano_solo"
    elif vocals >= 0.15:
        kind = "song"
    elif piano >= 0.35:
        kind = "piano_band"
    else:
        kind = "instrumental"
    return {"kind": kind, "ratios": {k: round(v, 3) for k, v in ratio.items()}, "drums": round(drums, 3)}


def mix_stems(stems: dict[str, Path], names: list[str], dst: Path) -> Path:
    acc = None
    sr = 44100
    for n in names:
        if n not in stems:
            continue
        d, sr = A.read(stems[n])
        acc = d if acc is None else acc[:, : d.shape[1]] + d[:, : acc.shape[1]]
    if acc is None:
        raise ValueError("mix_stems: stem がありません")
    A.write(dst, acc, sr)
    return dst
