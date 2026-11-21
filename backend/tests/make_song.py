"""検証用：歌もの風の合成曲（歌メロ・ベース・和音・ドラム）。

python tests/make_song.py out.wav
96 BPM・4/4・ハ長調、C-G-Am-F を2回。歌メロは声に近い（ビブラート＋母音のフォルマント）。
"""
import sys

import numpy as np
import soundfile as sf
from scipy.signal import lfilter

SR = 44100
BPM = 96.0
SPB = 60.0 / BPM
PROG = [(48, "maj"), (43, "maj"), (45, "min"), (41, "maj")] * 2
MEL = [
    (0, 1, 64), (1, 1, 67), (2, 2, 72), (4, 1, 71), (5, 1, 67), (6, 2, 62),
    (8, 1, 60), (9, 1, 64), (10, 2, 69), (12, 1, 65), (13, 1, 69), (14, 2, 72),
    (16, 1, 72), (17, 1, 71), (18, 2, 67), (20, 1, 67), (21, 1, 69), (22, 2, 71),
    (24, 1, 69), (25, 1, 67), (26, 2, 64), (28, 2, 65), (30, 2, 64),
]


def hz(m):
    return 440 * 2 ** ((m - 69) / 12)


def voice(m, dur):
    n = int(dur * SR)
    t = np.arange(n) / SR
    vib = 1 + 0.012 * np.sin(2 * np.pi * 5.5 * t) * np.clip(t / 0.25, 0, 1)
    ph = 2 * np.pi * np.cumsum(hz(m) * vib) / SR
    y = sum(np.sin(k * ph) / k for k in range(1, 25))
    # 母音「あ」のフォルマント（簡易な共振器）
    out = np.zeros_like(y)
    for f, bw, g in ((800, 80, 1.0), (1150, 90, 0.5), (2900, 120, 0.25)):
        r = np.exp(-np.pi * bw / SR)
        th = 2 * np.pi * f / SR
        out += g * lfilter([1 - r], [1, -2 * r * np.cos(th), r * r], y)
    env = np.minimum(1, t / 0.04) * np.clip((dur - t) / 0.06, 0, 1)
    return out * env


def tone(m, dur, decay=1.5, harm=8):
    n = int((dur + 0.3) * SR)
    t = np.arange(n) / SR
    y = sum(np.sin(2 * np.pi * hz(m) * k * t) * np.exp(-t * (decay + 0.4 * k)) / k for k in range(1, harm))
    return y * np.minimum(1, t / 0.005) * np.clip(1 - (t - dur) / 0.1, 0, 1)


def main(path):
    total = 32 * SPB + 1.5
    mix = np.zeros(int(total * SR))
    rng = np.random.default_rng(1)

    def add(sig, at, gain):
        s = int(at * SR)
        e = min(mix.size, s + sig.size)
        mix[s:e] += gain * sig[: e - s]

    for b, L, p in MEL:
        add(voice(p, L * SPB * 0.95), 0.5 + b * SPB, 0.20)
    for i, (root, q) in enumerate(PROG):
        third = 3 if q == "min" else 4
        bar = 0.5 + i * 4 * SPB
        for beat in range(4):
            add(tone(root - 12 + (7 if beat == 2 else 0), SPB * 0.9, 2.0, 5), bar + beat * SPB, 0.45)
        for beat in (0, 2):
            for iv in (12, 12 + third, 19):
                add(tone(root + iv, SPB * 1.8, 1.2), bar + beat * SPB, 0.10)
        for beat in range(4):
            kick = np.sin(2 * np.pi * 55 * np.arange(int(0.15 * SR)) / SR * np.exp(-np.arange(int(0.15 * SR)) / SR * 8)) * np.exp(-np.arange(int(0.15 * SR)) / SR * 25)
            add(kick, bar + beat * SPB, 0.5 if beat % 2 == 0 else 0)
            if beat % 2 == 1:
                add(rng.normal(0, 1, int(0.12 * SR)) * np.exp(-np.arange(int(0.12 * SR)) / SR * 30), bar + beat * SPB, 0.25)
            add(rng.normal(0, 1, int(0.03 * SR)) * np.exp(-np.arange(int(0.03 * SR)) / SR * 120), bar + (beat + 0.5) * SPB, 0.08)
    mix /= np.abs(mix).max() * 1.1
    sf.write(path, np.stack([mix, mix], 1), SR)
    print("wrote", path, f"{mix.size / SR:.1f}s")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "song.wav")
