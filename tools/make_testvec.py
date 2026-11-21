"""ブラウザ（onnxruntime-web）での推論結果を確かめるための入力と正解を作る。

    python tools/make_testvec.py

frontend/public/engine/testvec/ に、各モデルの入力（float32 の生バイト列）と、
ONNX Runtime（CPU）での出力の要約（合計・絶対値の合計・間引いた値）を書く。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from export_onnx import bt_logmel, np_demucs_spec, ort_session, test_audio  # noqa: E402

MODELS = ROOT / "frontend" / "public" / "engine" / "models"
OUT = ROOT / "frontend" / "public" / "engine" / "testvec"
OUT.mkdir(parents=True, exist_ok=True)


def summary(a: np.ndarray) -> dict:
    a = np.asarray(a, dtype=np.float32)
    flat = a.reshape(-1)
    step = max(1, flat.size // 997)
    return {"shape": list(a.shape), "sum": float(flat.astype(np.float64).sum()), "abs": float(np.abs(flat.astype(np.float64)).sum()), "step": step, "sample": flat[::step][:1000].tolist()}


def save(name: str, inputs: dict, outputs: dict) -> None:
    meta = {"inputs": {}, "outputs": {k: summary(v) for k, v in outputs.items()}}
    for k, v in inputs.items():
        v = np.ascontiguousarray(v, dtype=np.float32)
        (OUT / f"{name}.{k}.bin").write_bytes(v.tobytes())
        meta["inputs"][k] = list(v.shape)
    (OUT / f"{name}.json").write_text(json.dumps(meta), encoding="utf-8")
    print(name, {k: v["shape"] for k, v in meta["outputs"].items()})


def main() -> None:
    # ピアノ採譜
    x = test_audio(16000, 10.0)[None, :]
    s = ort_session(MODELS / "piano_transcription.onnx")
    names = [o.name for o in s.get_outputs()]
    save("piano_transcription", {"audio": x}, dict(zip(names, s.run(None, {"audio": x}))))
    # ピアノ採譜（分割版の前半）。後半（GRU）の正解は上の piano_transcription の出力
    s = ort_session(MODELS / "piano_frontend.onnx")
    names = [o.name for o in s.get_outputs()]
    save("piano_frontend", {"audio": x}, dict(zip(names, s.run(None, {"audio": x}))))
    # 音源分離
    s = ort_session(MODELS / "htdemucs_6s.onnx")
    wav = test_audio(44100, 343980 / 44100, stereo=True)
    z = np_demucs_spec(wav.astype(np.float64))
    mag = np.stack([z.real, z.imag], 1).reshape(1, 4, z.shape[-2], z.shape[-1]).astype(np.float32)
    o = s.run(None, {"mix": wav[None], "spec": mag})
    save("htdemucs_6s", {"mix": wav[None], "spec": mag}, {"spec_out": o[0], "wave_out": o[1]})
    # Stem への戻し（JS の demucsStem / demucsMonoPair の正解）
    from export_onnx import np_demucs_ispec

    L = wav.shape[-1]
    so = o[0][0].reshape(6, 2, 2, o[0].shape[-2], o[0].shape[-1])
    zc = so[:, :, 0].astype(np.float64) + 1j * so[:, :, 1].astype(np.float64)
    stems = o[1][0].astype(np.float64) + np_demucs_ispec(zc, L)
    save("demucs_stems", {}, {"stems": stems.astype(np.float32), "mono": stems.mean(1).astype(np.float32)})
    # 拍
    s = ort_session(MODELS / "beat_this.onnx")
    bt_audio = test_audio(22050, 30.0)
    spec = bt_logmel(bt_audio)[None, :1500]
    o = s.run(None, {"spect": spec})
    save("beat_this", {"spect": spec, "audio": bt_audio}, {"beat": o[0], "downbeat": o[1]})
    # Basic Pitch
    s = ort_session(MODELS / "basic_pitch.onnx")
    inp = s.get_inputs()[0].name
    a = test_audio(22050, 43844 / 22050)[None, :, None]
    o = s.run(None, {inp: a})
    save("basic_pitch", {"input": a}, {out.name: v for out, v in zip(s.get_outputs(), o)})
    print("input name for basic pitch:", inp, [o.name for o in s.get_outputs()])


if __name__ == "__main__":
    main()
