"""ブラウザ版用に書き直した「AI出力 → 音符」変換が元の実装と同じ結果か確かめる。

    python tools/check_postprocess.py

  ・ByteDance: piano_transcription_inference の RegressionPostProcessor と bd_post.to_events
  ・Basic Pitch: basic_pitch.note_creation と bp_notes
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))
JOBS = ROOT / "backend" / "data" / "jobs"


def check_bytedance(fails: list) -> None:
    import onnxruntime as ort
    from piano_transcription_inference.utilities import RegressionPostProcessor

    from app import audio as A
    from app.transcription import bd_post

    onnx = ROOT / "tools" / "onnx" / "piano_transcription.onnx"
    sess = ort.InferenceSession(str(onnx), providers=["CPUExecutionProvider"])
    names = [o.name for o in sess.get_outputs()]
    for jid in ("77f5f8cdea0b", "9d59eac821dd", "356e74ca0aee"):
        p = JOBS / jid / "target.wav"
        if not p.exists():
            continue
        y = A.load_mono(p, 16000)
        seg = bd_post.SEGMENT_SAMPLES
        n = int(np.ceil(len(y) / seg)) * seg
        a = np.zeros(n, np.float32)
        a[: len(y)] = y
        outs = {k: [] for k in names}
        for s in range(0, n - seg + 1, seg // 2):
            for k, v in zip(names, sess.run(None, {"audio": a[None, s : s + seg]})):
                outs[k].append(v if v.ndim == 2 else v[:, None])
        od = {k: bd_post.deframe(np.stack(v)) for k, v in outs.items()}
        t0 = time.time()
        pp = RegressionPostProcessor(100, classes_num=88, onset_threshold=0.3, offset_threshold=0.3, frame_threshold=0.1, pedal_offset_threshold=0.2)
        ref_n, ref_p = pp.output_dict_to_midi_events({k: v.copy() for k, v in od.items()})
        t1 = time.time()
        got_n, got_p = bd_post.to_events({k: v.copy() for k, v in od.items()})
        t2 = time.time()
        key = lambda e: (round(float(e["onset_time"]), 5), e["midi_note"], round(float(e["offset_time"]), 5), e["velocity"])
        same_n = sorted(map(key, ref_n)) == sorted(map(key, got_n))
        keyp = lambda e: (round(float(e["onset_time"]), 5), round(float(e["offset_time"]), 5))
        same_p = sorted(map(keyp, ref_p or [])) == sorted(map(keyp, got_p))
        print(f"  ByteDance {jid}: notes {len(ref_n)}/{len(got_n)} {'一致' if same_n else '不一致'}  pedals {len(ref_p or [])}/{len(got_p)} {'一致' if same_p else '不一致'}  ({t1 - t0:.1f}s → {t2 - t1:.1f}s)")
        if not (same_n and same_p):
            fails.append(jid)


def check_basicpitch(fails: list) -> None:
    import logging

    logging.getLogger().setLevel(logging.ERROR)
    from basic_pitch import ICASSP_2022_MODEL_PATH
    from basic_pitch import note_creation as nc
    from basic_pitch.inference import Model, run_inference

    from app.transcription import bp_notes
    from app.transcription.basicpitch import PROFILES

    model = Model(ICASSP_2022_MODEL_PATH)
    for jid in ("45988a35088d", "f64edd0b1805", "b3f6b83b4b99", "9d59eac821dd"):
        p = JOBS / jid / "target.wav"
        if not p.exists():
            continue
        out = run_inference(str(p), model)
        for prof_name in ("guitar", "voices", "ensemble"):
            prof = PROFILES[prof_name]
            min_len = int(np.round(prof["min_ms"] / 1000 * (22050 / 256)))
            t0 = time.time()
            ref = nc.output_to_notes_polyphonic(
                out["note"].copy(), out["onset"].copy(), onset_thresh=prof["onset"], frame_thresh=prof["frame"],
                infer_onsets=True, min_note_len=min_len, min_freq=prof["fmin"], max_freq=prof["fmax"], melodia_trick=True,
            )
            t1 = time.time()
            got = bp_notes.output_to_notes_polyphonic(
                out["note"], out["onset"], onset_thresh=prof["onset"], frame_thresh=prof["frame"],
                infer_onsets=True, min_note_len=min_len, min_freq=prof["fmin"], max_freq=prof["fmax"], melodia_trick=True,
            )
            t2 = time.time()
            same = [(a, b, c, round(float(d), 6)) for a, b, c, d in ref] == [(a, b, c, round(float(d), 6)) for a, b, c, d in got]
            tr = nc.model_frames_to_time(out["contour"].shape[0])
            tg = bp_notes.model_frames_to_time(out["contour"].shape[0])
            same_t = np.allclose(tr, tg, atol=1e-12)
            print(f"  BasicPitch {jid} {prof_name:8s}: {len(ref)}/{len(got)} {'一致' if same and same_t else '不一致'}  ({t1 - t0:.2f}s → {t2 - t1:.2f}s)")
            if not (same and same_t):
                fails.append(f"{jid}:{prof_name}")


def main() -> None:
    fails: list = []
    check_bytedance(fails)
    check_basicpitch(fails)
    print("OK" if not fails else f"NG: {fails}")


if __name__ == "__main__":
    main()
