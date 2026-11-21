"""ByteDance 高解像度ピアノ採譜モデル（Kong et al., 2021）のアダプタ。

MAESTRO で学習した CRNN。音符の立ち上がり・離鍵・強さ・ダンパーペダルを
10ms 単位で推定する。公開チェックポイントは約165MB（初回のみ自動取得）。

元パッケージは wget で重みを落とそうとするため Windows では動かない。
ここで自前に取得してからパスを渡す。
"""
from __future__ import annotations

import threading
import urllib.request
from pathlib import Path
from typing import Callable

import numpy as np

from ..config import MODELS_DIR, torch_device

CKPT_URL = (
    "https://zenodo.org/record/4034264/files/"
    "CRNN_note_F1%3D0.9677_pedal_F1%3D0.9186.pth?download=1"
)
CKPT_PATH = MODELS_DIR / "piano_transcription_note_pedal.pth"
CKPT_MIN_BYTES = 160_000_000
SAMPLE_RATE = 16000

_lock = threading.Lock()
_model = None


def available() -> bool:
    try:
        import piano_transcription_inference  # noqa: F401
        import torch  # noqa: F401

        return True
    except Exception:
        return False


def checkpoint_ready() -> bool:
    return CKPT_PATH.exists() and CKPT_PATH.stat().st_size >= CKPT_MIN_BYTES


def ensure_checkpoint(progress: Callable[[float, str], None] | None = None) -> Path:
    """重みが無ければダウンロードする（途中で落ちた残骸は作り直す）。"""
    if checkpoint_ready():
        return CKPT_PATH
    tmp = CKPT_PATH.with_suffix(".part")
    req = urllib.request.Request(CKPT_URL, headers={"User-Agent": "gakufu/1.0"})
    with urllib.request.urlopen(req, timeout=60) as resp, open(tmp, "wb") as f:
        total = int(resp.headers.get("Content-Length") or 0) or 172_000_000
        done = 0
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            if progress:
                progress(min(done / total, 1.0), f"採譜モデルを取得中 {done / 1e6:.0f}/{total / 1e6:.0f} MB")
    if tmp.stat().st_size < CKPT_MIN_BYTES:
        tmp.unlink(missing_ok=True)
        raise RuntimeError("採譜モデルのダウンロードが途中で切れました。再実行してください。")
    tmp.replace(CKPT_PATH)
    return CKPT_PATH


def _load_model():
    global _model
    if _model is not None:
        return _model
    import torch
    from piano_transcription_inference import PianoTranscription

    ensure_checkpoint()
    device = torch.device(torch_device())
    _orig_load = torch.load

    def _load(*a, **kw):  # 古い形式の重みなので weights_only を外す（配布元は固定URL）
        kw.setdefault("weights_only", False)
        return _orig_load(*a, **kw)

    torch.load = _load
    try:
        _model = PianoTranscription(device=device, checkpoint_path=str(CKPT_PATH))
    finally:
        torch.load = _orig_load
    return _model


def transcribe(
    audio: np.ndarray,
    progress: Callable[[float, str], None] | None = None,
    onset_threshold: float = 0.3,
    frame_threshold: float = 0.1,
) -> tuple[list[dict], list[dict]]:
    """16kHz モノラル音声を採譜する。

    Returns:
        notes:  [{"pitch", "on", "off", "vel"}]  （秒、vel は 0-127）
        pedals: [{"on", "off"}]
    """
    import piano_transcription_inference.inference as inf
    import piano_transcription_inference.pytorch_utils as pu

    with _lock:
        model = _load_model()
        model.onset_threshold = onset_threshold
        model.frame_threshold = frame_threshold

        # 区間ごとの推論ループに進捗フックを差し込む
        orig_forward = pu.forward

        def forward_with_progress(m, x, batch_size):
            import torch

            out: dict[str, list] = {}
            dev = next(m.parameters()).device
            n = len(x)
            for i in range(0, n, 4):
                batch = torch.as_tensor(x[i : i + 4], dtype=torch.float32, device=dev)
                with torch.no_grad():
                    m.eval()
                    o = m(batch)
                for k, v in o.items():
                    out.setdefault(k, []).append(v.data.cpu().numpy())
                if progress:
                    progress(min((i + 4) / n, 1.0), f"ピアノ採譜AI 推論中 {min(i + 4, n)}/{n} 区間")
            return {k: np.concatenate(v, axis=0) for k, v in out.items()}

        inf.forward = forward_with_progress
        try:
            result = model.transcribe(audio.astype(np.float32), None)
        finally:
            inf.forward = orig_forward

    return events_to_notes(result["est_note_events"], result["est_pedal_events"])


def events_to_notes(note_events: list[dict], pedal_events: list[dict] | None) -> tuple[list[dict], list[dict]]:
    """モデルの音符・ペダルイベント → アプリの音符形式（ブラウザ版と共通）。"""
    notes = [
        {
            "pitch": int(e["midi_note"]),
            "on": float(e["onset_time"]),
            "off": float(e["offset_time"]),
            "vel": int(e["velocity"]),
        }
        for e in note_events
        if 21 <= int(e["midi_note"]) <= 108 and e["offset_time"] > e["onset_time"]
    ]
    pedals = [
        {"on": float(e["onset_time"]), "off": float(e["offset_time"])}
        for e in (pedal_events or [])
        if e["offset_time"] > e["onset_time"]
    ]
    notes = _drop_overtone_ghosts(notes)
    notes.sort(key=lambda n: (n["on"], n["pitch"]))
    return notes, pedals


def _drop_overtone_ghosts(notes: list[dict]) -> list[dict]:
    """最高音域の弱い音で、同時に鳴る強い音のちょうど倍音位置（12度・2オクターブ・17度…）に
    あるものは倍音の聴き違いとして落とす。実音のオクターブ重ねは残すため 12 半音は対象外。"""
    order = sorted(notes, key=lambda n: n["on"])
    ons = np.array([n["on"] for n in order])
    keep = []
    for i, n in enumerate(order):
        if n["pitch"] >= 88 and n["vel"] < 72:
            lo = np.searchsorted(ons, n["on"] - 0.06)
            hi = np.searchsorted(ons, n["on"] + 0.06)
            ghost = any(
                (n["pitch"] - order[j]["pitch"]) in (19, 24, 28, 31, 34, 36) and order[j]["vel"] >= n["vel"] - 4
                for j in range(lo, hi)
                if j != i
            )
            if ghost:
                continue
        keep.append(n)
    return keep
