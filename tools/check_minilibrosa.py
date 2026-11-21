"""minilibrosa（ブラウザ版で librosa の代わりに使う numpy 実装）が本物と同じ結果を出すか確かめる。

    python tools/check_minilibrosa.py

手元のジョブ（backend/data/jobs）の音源で、拍・オンセット・pYIN・RMS を比べ、
さらにアプリの処理（rhythm.track_beats / beatfit.best_beats / pitchtrack.track）を
両方で通して結果が一致するかを見る。
"""
from __future__ import annotations

import contextlib
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))

import librosa as real  # noqa: E402

from app import audio as A  # noqa: E402
from app import beatfit, minilibrosa as lite, rhythm  # noqa: E402
from app.transcription import pitchtrack  # noqa: E402

JOBS = ROOT / "backend" / "data" / "jobs"


@contextlib.contextmanager
def use_lite():
    """ブラウザ版と同じ条件：librosa は minilibrosa、scipy は import できない。"""
    saved = {k: v for k, v in sys.modules.items() if k == "librosa" or k == "scipy" or k.startswith("scipy.")}
    sys.modules["librosa"] = lite
    for k in saved:
        if k != "librosa":
            sys.modules[k] = None  # import すると ImportError
    sys.modules["scipy"] = None
    try:
        yield
    finally:
        for k in list(sys.modules):
            if k == "scipy" or k.startswith("scipy."):
                del sys.modules[k]
        sys.modules.update(saved)


def same_events(a, b, tol):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.size == 0 or b.size == 0:
        return int(a.size == b.size), max(a.size, b.size)
    hit = sum(np.min(np.abs(b - x)) <= tol for x in a)
    return hit, max(a.size, b.size)


def main() -> None:
    fails = 0
    jobs = [p for p in sorted(JOBS.glob("*")) if (p / "mix.wav").exists() and (p / "raw.json").exists()]
    seen = set()
    for jd in jobs:
        meta = json.loads((jd / "job.json").read_text(encoding="utf-8")).get("meta", {})
        title = meta.get("title", "")
        if title in seen:
            continue
        seen.add(title)
        y = A.load_mono(jd / "mix.wav", 22050)
        print(f"== {jd.name} {title[:30]} ({y.size / 22050:.0f}s)")
        # onset strength
        e_r = real.onset.onset_strength(y=y, sr=22050, hop_length=512, aggregate=np.median)
        e_l = lite.onset.onset_strength(y=y, sr=22050, hop_length=512, aggregate=np.median)
        rel = float(np.max(np.abs(e_r - e_l)) / (np.max(np.abs(e_r)) + 1e-9))
        print(f"  onset_strength  len {e_r.size}/{e_l.size}  max rel diff {rel:.2e}")
        # beat_track on the same envelope（アルゴリズムだけを比べる）
        t_r, b_r = real.beat.beat_track(onset_envelope=e_r, sr=22050, hop_length=512, start_bpm=100, tightness=100, units="time", trim=False)
        t_l, b_l = lite.beat.beat_track(onset_envelope=e_r, sr=22050, hop_length=512, start_bpm=100, tightness=100, units="time", trim=False)
        h, n = same_events(b_r, b_l, 1e-6)
        print(f"  beat_track      tempo {float(np.atleast_1d(t_r)[0]):.2f}/{float(np.atleast_1d(t_l)[0]):.2f}  beats {h}/{n}")
        fails += h != n
        # onset_detect
        o_r = real.onset.onset_detect(y=y, sr=22050, hop_length=512, units="time")
        o_l = lite.onset.onset_detect(y=y, sr=22050, hop_length=512, units="time")
        h, n = same_events(o_r, o_l, 0.024)
        print(f"  onset_detect    {h}/{n}")
        # アプリの処理を両方で
        r1 = rhythm.track_beats(y, 22050)
        with use_lite():
            r2 = rhythm.track_beats(y, 22050)
        h, n = same_events(r1["beats"], r2["beats"], 0.024)
        print(f"  rhythm.track_beats  bpm {r1['bpm']}/{r2['bpm']}  beats {h}/{n}")
        fails += h < n * 0.98
        raw = json.loads((jd / "raw.json").read_text(encoding="utf-8"))
        notes = raw["notes"]
        t0 = time.time()
        f1 = beatfit.best_beats(r1["beats"], notes, r1["duration"])
        t1 = time.time()
        with use_lite():
            f2 = beatfit.best_beats(r1["beats"], notes, r1["duration"])
        t2 = time.time()
        h, n = same_events(f1["beats"], f2["beats"], 0.005)
        print(f"  beatfit.best_beats  {f1['method']}/{f2['method']}  beats {h}/{n}  ({t1 - t0:.1f}s / {t2 - t1:.1f}s)")
        fails += f1["method"] != f2["method"] or h < n
        # pYIN（歌ものの stem があれば）
        for stem, kind in (("vocals", "vocals"), ("bass", "bass")):
            sp = jd / "stems" / f"{stem}.wav"
            if not sp.exists():
                continue
            ys = A.load_mono(sp, pitchtrack.SR)[: pitchtrack.SR * 40]
            prof = pitchtrack.PROFILES[kind]
            t0 = time.time()
            f0r, vr, pr = real.pyin(ys, fmin=prof["fmin"], fmax=prof["fmax"], sr=pitchtrack.SR, frame_length=prof["frame"], hop_length=pitchtrack.HOP)
            t1 = time.time()
            f0l, vl, pl = lite.pyin(ys, fmin=prof["fmin"], fmax=prof["fmax"], sr=pitchtrack.SR, frame_length=prof["frame"], hop_length=pitchtrack.HOP)
            t2 = time.time()
            both = vr & vl
            agree = float(np.mean(vr == vl))
            cents = np.abs(1200 * np.log2(f0r[both] / f0l[both])) if both.any() else np.zeros(1)
            print(f"  pyin {stem:6s} voiced agree {agree * 100:.2f}%  f0 max diff {cents.max():.1f} cents  vprob max diff {np.max(np.abs(pr - pl)):.1e}  ({t1 - t0:.1f}s / {t2 - t1:.1f}s)")
            fails += agree < 0.995
            n1 = pitchtrack.track(ys, pitchtrack.SR, kind)
            with use_lite():
                n2 = pitchtrack.track(ys, pitchtrack.SR, kind)
            same = sum(1 for a in n1 if any(b["pitch"] == a["pitch"] and abs(b["on"] - a["on"]) < 0.021 for b in n2))
            print(f"  pitchtrack {stem:6s} notes {same}/{max(len(n1), len(n2))}")
            fails += same < max(len(n1), len(n2)) * 0.97
            r_r = real.feature.rms(y=ys, frame_length=prof["frame"], hop_length=pitchtrack.HOP)
            r_l = lite.feature.rms(y=ys, frame_length=prof["frame"], hop_length=pitchtrack.HOP)
            print(f"  rms {stem:6s} max diff {np.max(np.abs(r_r - r_l)):.1e}")
    print("OK" if not fails else f"NG: {fails} 件のずれ")


if __name__ == "__main__":
    main()
