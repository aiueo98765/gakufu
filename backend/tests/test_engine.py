"""音声ファイル1本で採譜〜楽譜化を通す（API を使わない検証用）。

python tests/test_engine.py <audio.wav> [bytedance|builtin] [出力フォルダ]
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import audio as A  # noqa: E402
from app import rhythm, theory  # noqa: E402
from app.arrange import arrange  # noqa: E402
from app.midi_export import score_to_midi  # noqa: E402
from app.notation import score_to_musicxml  # noqa: E402
from app.transcription import builtin, bytedance  # noqa: E402


def main() -> None:
    path = Path(sys.argv[1])
    engine = sys.argv[2] if len(sys.argv) > 2 else "bytedance"
    out_dir = Path(sys.argv[3]) if len(sys.argv) > 3 else path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    if engine == "bytedance":
        y16 = A.load_mono(path, 16000)
        notes, pedals = bytedance.transcribe(y16)
    else:
        y = A.load_mono(path, builtin.SR)
        notes = builtin.transcribe(y, builtin.SR, 0.5)
        pedals = []
    for n in notes:
        n["role"] = "piano"
    print(f"[{engine}] {len(notes)} notes, {len(pedals)} pedals in {time.time() - t0:.1f}s")
    y = A.load_mono(path, A.ANALYSIS_SR)
    beats = rhythm.track_beats(y, A.ANALYSIS_SR)
    key = theory.estimate_key(notes)
    print("bpm", beats["bpm"], "key", key["ja"], key["en"], "conf", key["confidence"])
    raw = {"notes": notes, "pedals": pedals, "engine": engine, "sourceKind": "piano_solo"}
    analysis = {"beats": beats["beats"], "bpm": beats["bpm"], "duration": beats["duration"],
                "firstOnset": rhythm.first_strong_onset(y, A.ANALYSIS_SR), "key": key}
    for mode in ("easy", "standard", "virtuoso"):
        t1 = time.time()
        s = arrange(raw, analysis, mode, {}, {"title": path.stem})
        xml = score_to_musicxml(s, {"fingering": "all"})
        mid = score_to_midi(s)
        (out_dir / f"{path.stem}_{mode}.json").write_text(json.dumps(s, ensure_ascii=False), encoding="utf-8")
        (out_dir / f"{path.stem}_{mode}.musicxml").write_text(xml, encoding="utf-8")
        (out_dir / f"{path.stem}_{mode}.mid").write_bytes(mid)
        d = s["difficulty"]
        print(
            f"  {mode:9s} notes={len(s['notes']):4d} tempo={s['tempo']} ts={s['timeSignature']} "
            f"measures={s['totalBeats'] // s['timeSignature'][0]} offset={s['audio']['offsetBeats']} "
            f"chords={' '.join(c['symbol'] for c in s['chords'][:10])} diff={d['overall']}({d['label']}) "
            f"hard={d['difficultMeasures'][:8]} {time.time() - t1:.2f}s"
        )


if __name__ == "__main__":
    main()
