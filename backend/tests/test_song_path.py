"""歌もの経路の検証：歌声の音高追跡（pYIN）→ 歌メロ＋伴奏型のアレンジ → MusicXML。

python tests/test_song_path.py
合成した「声」だけのトラックを分離済みボーカル Stem とみなして追跡し、正解の旋律と比べる。
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import make_song as S  # noqa: E402
from app import rhythm, theory  # noqa: E402
from app.arrange import arrange  # noqa: E402
from app.notation import score_to_musicxml  # noqa: E402
from app.transcription import pitchtrack  # noqa: E402


def main():
    total = 32 * S.SPB + 1.5
    vox = np.zeros(int(total * S.SR))
    for b, L, p in S.MEL:
        sig = S.voice(p, L * S.SPB * 0.95)
        s = int((0.5 + b * S.SPB) * S.SR)
        vox[s : s + sig.size] += sig[: vox.size - s]
    vox /= np.abs(vox).max()
    mel = pitchtrack.track(vox.astype(np.float32), S.SR, "vocals")
    hit = 0
    for b, L, p in S.MEL:
        t = 0.5 + b * S.SPB
        if any(m["pitch"] == p and abs(m["on"] - t) < 0.1 for m in mel):
            hit += 1
    print(f"pYIN: detected {len(mel)} notes, matched {hit}/{len(S.MEL)} melody notes (pitch+onset)")

    notes = [dict(m, role="melody") for m in mel]
    for i, (root, q) in enumerate(S.PROG):
        bar = 0.5 + i * 4 * S.SPB
        third = 3 if q == "min" else 4
        for beat in (0, 2):
            for iv in (12, 12 + third, 19):
                notes.append({"pitch": root + iv, "on": bar + beat * S.SPB, "off": bar + (beat + 1.8) * S.SPB, "vel": 60, "role": "harmony"})
        for beat in range(4):
            notes.append({"pitch": root - 12 + (7 if beat == 2 else 0), "on": bar + beat * S.SPB, "off": bar + (beat + 0.9) * S.SPB, "vel": 80, "role": "bass"})
    beats = list(np.arange(0.5 - S.SPB, total, S.SPB))
    analysis = {"beats": beats, "bpm": S.BPM, "duration": total, "firstOnset": 0.5,
                "key": theory.estimate_key([n for n in notes if n["role"] != "melody"])}
    raw = {"notes": notes, "pedals": [], "engine": "test", "sourceKind": "song"}
    for mode in ("easy", "standard", "virtuoso"):
        for lh in (("auto", "broken", "alberti", "triad") if mode == "standard" else ("auto",)):
            s = arrange(raw, analysis, mode, {"leftHand": lh}, {"title": "song test"})
            xml = score_to_musicxml(s, {"fingering": "all"})
            r = [n for n in s["notes"] if n["hand"] == "R"]
            l = [n for n in s["notes"] if n["hand"] == "L"]
            print(
                f"{mode:9s} lh={lh:8s} ts={s['timeSignature']} R={len(r)} L={len(l)} "
                f"melody={sum(1 for n in r if n.get('role') == 'melody')} pedals={len(s['pedals'])} "
                f"chords={' '.join(c['symbol'] for c in s['chords'][:8])} xml={len(xml) // 1024}KB diff={s['difficulty']['overall']}"
            )


if __name__ == "__main__":
    main()
