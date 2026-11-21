"""合成曲の正解と採譜結果（秒単位の生ノート）を比べる。

python tests/eval_synth.py [bytedance|builtin] <ode.wav>
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import make_synth as M  # noqa: E402
from app import audio as A  # noqa: E402
from app.transcription import builtin, bytedance  # noqa: E402


def truth():
    out = []
    for b, L, p in M.MELODY:
        out.append((b * M.SPB + 0.5, p))
    for b, ps in M.CHORDS:
        for p in ps:
            out.append((b * M.SPB + 0.5, p))
    return sorted(out)


def main():
    engine = sys.argv[1]
    path = Path(sys.argv[2])
    if engine == "bytedance":
        notes, _ = bytedance.transcribe(A.load_mono(path, 16000))
    else:
        notes = builtin.transcribe(A.load_mono(path, builtin.SR), builtin.SR, 0.5)
    gt = truth()
    used = set()
    tp = 0
    fps = []
    for n in notes:
        hit = None
        for i, (t, p) in enumerate(gt):
            if i not in used and p == n["pitch"] and abs(t - n["on"]) < 0.06:
                hit = i
                break
        if hit is None:
            fps.append(n)
        else:
            used.add(hit)
            tp += 1
    fn = [gt[i] for i in range(len(gt)) if i not in used]
    print(f"{engine}: det={len(notes)} truth={len(gt)} TP={tp} P={tp / max(1, len(notes)):.2f} R={tp / len(gt):.2f}")
    print("false positives (on, pitch, vel):", [(round(n["on"], 2), n["pitch"], n["vel"]) for n in fps[:30]])
    print("missed (t, pitch):", [(round(t, 2), p) for t, p in fn[:30]])


if __name__ == "__main__":
    main()
