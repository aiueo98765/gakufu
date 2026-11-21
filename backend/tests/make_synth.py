"""検証用：答えのわかっている曲を簡易ピアノ音で合成する。

python tests/make_synth.py out.wav
右手=「よろこびの歌」の旋律＋7小節目に16分の音階、左手=ベース+和音。100 BPM・4/4・ハ長調。
"""
import sys

import numpy as np
import soundfile as sf

SR = 44100
BPM = 100.0
SPB = 60.0 / BPM

E, F, G, C, D = 64, 65, 67, 60, 62
MELODY = [
    # (拍, 長さ, 音)
    *[(i, 1, p) for i, p in enumerate([E, E, F, G, G, F, E, D, C, C, D, E])],
    (12, 1.5, E), (13.5, 0.5, D), (14, 2, D),
    *[(16 + i, 1, p) for i, p in enumerate([E, E, F, G, G, F, E, D, C, C, D, E])],
    (28, 1.5, D), (29.5, 0.5, C), (30, 2, C),
]
# 16分の上行・下行音階（右手の速いパッセージ）
SCALE = [60, 62, 64, 65, 67, 69, 71, 72, 74, 72, 71, 69, 67, 65, 64, 62]
for i, p in enumerate(SCALE):
    MELODY.append((32 + i * 0.25, 0.25, p))
MELODY.append((36, 4, 72))
CHORDS = [(0, [48, 52, 55]), (4, [43, 50, 55]), (8, [45, 52, 57]), (12, [43, 50, 55]),
          (16, [48, 52, 55]), (20, [43, 50, 55]), (24, [41, 48, 53]), (28, [43, 50, 55]),
          (32, [48, 52, 55]), (36, [48, 55, 60])]


def tone(midi: int, dur: float, vel: float) -> np.ndarray:
    f0 = 440.0 * 2 ** ((midi - 69) / 12)
    n = int((dur + 0.6) * SR)
    t = np.arange(n) / SR
    y = np.zeros(n)
    B = 0.0004
    for h in range(1, 10):
        fh = h * f0 * np.sqrt(1 + B * h * h)
        if fh > SR / 2.2:
            break
        y += (1 / h ** 1.15) * np.sin(2 * np.pi * fh * t) * np.exp(-t * (1.2 + 0.45 * h))
    env = np.minimum(1, t / 0.004)
    rel = np.clip(1 - (t - dur) / 0.12, 0, 1)
    return y * env * rel * vel


def main(path: str) -> None:
    total = (40 + 2) * SPB
    out = np.zeros(int(total * SR) + SR)
    rng = np.random.default_rng(0)
    for b, L, p in MELODY:
        s = int((b * SPB + rng.normal(0, 0.006)) * SR + 0.5 * SR)
        y = tone(p, L * SPB * 0.95, 0.5)
        out[s : s + y.size] += y
    for b, ps in CHORDS:
        s = int((b * SPB) * SR + 0.5 * SR)
        for p in ps:
            y = tone(p, 4 * SPB * 0.97, 0.28)
            out[s : s + y.size] += y
    out /= np.abs(out).max() * 1.1
    sf.write(path, np.stack([out, out], axis=1), SR)
    print("wrote", path, f"{out.size / SR:.1f}s")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "synth.wav")
