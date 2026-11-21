"""拍の検出・テンポ地図・拍子と小節頭の推定。

録音はテンポが揺れる（ルバート）ので、一定テンポのグリッドではなく
検出した拍の時刻列（テンポ地図）で「秒 → 拍」を変換してから量子化する。
"""
from __future__ import annotations

import numpy as np

HOP = 512


def _onset_env(y: np.ndarray, sr: int) -> np.ndarray:
    import librosa

    return librosa.onset.onset_strength(y=y, sr=sr, hop_length=HOP, aggregate=np.median)


def track_beats(y: np.ndarray, sr: int) -> dict:
    """拍の時刻列（秒）と代表テンポ。曲の頭から終わりまで拍を補う。"""
    import librosa

    duration = y.size / sr
    oenv = _onset_env(y, sr)
    tempo, beats = librosa.beat.beat_track(
        onset_envelope=oenv, sr=sr, hop_length=HOP, start_bpm=100, tightness=100, units="time", trim=False
    )
    bpm = float(np.atleast_1d(tempo)[0]) if np.size(tempo) else 0.0
    beats = np.asarray(beats, dtype=float)
    if beats.size < 4 or bpm <= 0:
        bpm = bpm if 40 < bpm < 240 else 100.0
        beats = np.arange(0.0, duration + 60 / bpm, 60 / bpm)
    beats = _fix_octave(beats, oenv, sr)
    beats = _extend(beats, duration)
    ibi = np.median(np.diff(beats))
    return {"beats": beats.tolist(), "bpm": round(60.0 / ibi, 2), "duration": duration}


def _strength_at(times: np.ndarray, oenv: np.ndarray, sr: int) -> np.ndarray:
    idx = np.clip(np.round(times * sr / HOP).astype(int), 0, len(oenv) - 1)
    return oenv[idx]


def _fix_octave(beats: np.ndarray, oenv: np.ndarray, sr: int) -> np.ndarray:
    """テンポの倍・半分の取り違えを、楽譜として自然な範囲（約 60〜170）に寄せる。"""
    for _ in range(3):
        ibi = float(np.median(np.diff(beats)))
        bpm = 60 / ibi
        if bpm > 170:
            a, b = beats[0::2], beats[1::2]
            beats = a if _strength_at(a, oenv, sr).sum() >= _strength_at(b, oenv, sr).sum() else b
        elif bpm < 58:
            mids = (beats[:-1] + beats[1:]) / 2
            beats = np.sort(np.concatenate([beats, mids]))
        else:
            break
    return beats


def _extend(beats: np.ndarray, duration: float) -> np.ndarray:
    ibi_head = float(np.median(np.diff(beats[:8]))) if beats.size > 2 else 0.5
    ibi_tail = float(np.median(np.diff(beats[-8:]))) if beats.size > 2 else 0.5
    head = []
    t = beats[0] - ibi_head
    while t > -ibi_head * 0.5:
        head.append(t)
        t -= ibi_head
    tail = []
    t = beats[-1] + ibi_tail
    while t < duration + ibi_tail:
        tail.append(t)
        t += ibi_tail
    out = np.concatenate([np.array(head[::-1]), beats, np.array(tail)])
    return out


def rescale_beats(beats: list[float], scale: float) -> list[float]:
    """テンポ×2（拍を間に挿入）/ ÷2（1つおきに間引き）。"""
    b = np.asarray(beats, dtype=float)
    if scale == 2:
        mids = (b[:-1] + b[1:]) / 2
        return np.sort(np.concatenate([b, mids])).tolist()
    if scale == 0.5:
        return b[0::2].tolist()
    return b.tolist()


def uniform_beats(bpm: float, duration: float, anchor: float = 0.0) -> list[float]:
    """指定テンポの等間隔グリッド。anchor（最初の強いオンセット）に位相を合わせる。"""
    ibi = 60.0 / bpm
    start = anchor - np.floor(anchor / ibi) * ibi
    return np.arange(start - ibi, duration + ibi, ibi).tolist()


class TempoMap:
    """秒 ⇔ 楽譜上の拍。score_beat = audio_beat_index + offset。"""

    def __init__(self, beats: list[float], offset: int = 0):
        self.beats = np.asarray(beats, dtype=float)
        self.idx = np.arange(self.beats.size, dtype=float)
        self.offset = offset
        d = np.diff(self.beats)
        self.ibi_head = float(d[0]) if d.size else 0.5
        self.ibi_tail = float(d[-1]) if d.size else 0.5

    def sec_to_beat(self, t: float | np.ndarray):
        t = np.asarray(t, dtype=float)
        b = np.interp(t, self.beats, self.idx)
        b = np.where(t < self.beats[0], (t - self.beats[0]) / self.ibi_head, b)
        b = np.where(t > self.beats[-1], self.idx[-1] + (t - self.beats[-1]) / self.ibi_tail, b)
        return b + self.offset

    def beat_to_sec(self, b: float | np.ndarray):
        b = np.asarray(b, dtype=float) - self.offset
        t = np.interp(b, self.idx, self.beats)
        t = np.where(b < 0, self.beats[0] + b * self.ibi_head, t)
        t = np.where(b > self.idx[-1], self.beats[-1] + (b - self.idx[-1]) * self.ibi_tail, t)
        return t

    def score_beat_times(self, total_beats: int) -> list[float]:
        return [round(float(x), 4) for x in self.beat_to_sec(np.arange(total_beats + 1))]

    def quarter_times(self, total_quarters: int, beat_len: float) -> list[float]:
        """楽譜の4分音符位置 0,1,2,… が元音声の何秒か（複合拍子でも4分音符刻みで返す）。"""
        q = np.arange(total_quarters + 1, dtype=float)
        return [round(float(x), 4) for x in self.beat_to_sec(q / beat_len)]

    @property
    def bpm(self) -> float:
        return 60.0 / float(np.median(np.diff(self.beats)))


def estimate_meter(beats: list[float], notes: list[dict], forced: int | None = None) -> dict:
    """拍子（3 か 4）と、何拍目が小節の頭か（位相）を推定する。

    小節の頭には「低音の打鍵」「長い音の始まり」「和音の変わり目」が集まりやすい。
    """
    b = np.asarray(beats, dtype=float)
    n = b.size
    if n < 8:
        return {"beatsPerMeasure": forced or 4, "phase": 0}
    ibi = float(np.median(np.diff(b)))
    tol = min(0.08, ibi * 0.2)
    bass_on = np.zeros(n)
    all_on = np.zeros(n)
    long_on = np.zeros(n)
    chroma = np.zeros((n, 12))
    ons = np.array([x["on"] for x in notes]) if notes else np.zeros(0)
    for x, on in zip(notes, ons):
        i = int(np.searchsorted(b, on))
        for j in (i - 1, i):
            if 0 <= j < n and abs(b[j] - on) < tol:
                v = x.get("vel", 80) / 127
                all_on[j] += v
                if x["pitch"] < 55:
                    bass_on[j] += v * 1.5
                if x["off"] - x["on"] > ibi * 0.9:
                    long_on[j] += v
                break
        k = int(np.clip(np.searchsorted(b, on, side="right") - 1, 0, n - 1))
        chroma[k, x["pitch"] % 12] += (x["off"] - x["on"]) * x.get("vel", 80) / 127
    change = np.zeros(n)
    for i in range(1, n):
        a, c = chroma[i - 1], chroma[i]
        na, nc = np.linalg.norm(a), np.linalg.norm(c)
        if na > 0 and nc > 0:
            change[i] = 1 - float(a @ c) / (na * nc)

    def z(v):
        s = v.std()
        return (v - v.mean()) / s if s > 1e-9 else v * 0

    accent = 1.0 * z(bass_on) + 0.6 * z(long_on) + 0.7 * z(change) + 0.3 * z(all_on)
    best = None
    for m in ([forced] if forced else [4, 3]):
        for p in range(m):
            sel = accent[p::m]
            if sel.size < 2:
                continue
            contrast = float(sel.mean() - accent.mean())
            score = contrast + (0.08 if m == 4 else 0.0)  # 迷ったら 4拍子
            if best is None or score > best[0]:
                best = (score, m, p)
    if best is None:
        return {"beatsPerMeasure": forced or 4, "phase": 0}
    return {"beatsPerMeasure": best[1], "phase": best[2]}


SIMPLE_METERS = {"4/4": 4, "3/4": 3, "2/4": 2}
COMPOUND_METERS = {"6/8": 2, "12/8": 4}


def _accent(beats: np.ndarray, notes: list[dict]) -> np.ndarray:
    """拍ごとの「小節頭らしさ」。低音の打鍵・長い音の始まり・和音の変わり目を合成する。"""
    n = beats.size
    ibi = float(np.median(np.diff(beats)))
    tol = min(0.08, ibi * 0.2)
    bass_on = np.zeros(n)
    all_on = np.zeros(n)
    long_on = np.zeros(n)
    chroma = np.zeros((n, 12))
    for x in notes:
        on = x["on"]
        i = int(np.searchsorted(beats, on))
        for j in (i - 1, i):
            if 0 <= j < n and abs(beats[j] - on) < tol:
                v = x.get("vel", 80) / 127
                all_on[j] += v
                if x["pitch"] < 55:
                    bass_on[j] += v * 1.5
                if x["off"] - x["on"] > ibi * 0.9:
                    long_on[j] += v
                break
        k = int(np.clip(np.searchsorted(beats, on, side="right") - 1, 0, n - 1))
        chroma[k, x["pitch"] % 12] += (x["off"] - x["on"]) * x.get("vel", 80) / 127
    change = np.zeros(n)
    for i in range(1, n):
        a, c = chroma[i - 1], chroma[i]
        na, nc = np.linalg.norm(a), np.linalg.norm(c)
        if na > 0 and nc > 0:
            change[i] = 1 - float(a @ c) / (na * nc)

    def z(v):
        s = v.std()
        return (v - v.mean()) / s if s > 1e-9 else v * 0

    return 1.0 * z(bass_on) + 0.6 * z(long_on) + 0.7 * z(change) + 0.3 * z(all_on)


def _best_phase(accent: np.ndarray, period: int) -> tuple[float, int]:
    best = (-1e9, 0)
    for p in range(period):
        sel = accent[p::period]
        if sel.size < 2:
            continue
        contrast = float(sel.mean() - accent.mean())
        if contrast > best[0]:
            best = (contrast, p)
    return best


def _triplet_ratio(beats: np.ndarray, notes: list[dict]) -> tuple[float, int]:
    """拍の中で 1/3・2/3 の位置に来る音が、1/2・1/4 の位置より多いか（3分割の拍か）。"""
    thirds = halves = 0
    for x in notes:
        i = int(np.searchsorted(beats, x["on"], side="right")) - 1
        if i < 0 or i + 1 >= beats.size:
            continue
        f = (x["on"] - beats[i]) / (beats[i + 1] - beats[i])
        if min(abs(f - 1 / 3), abs(f - 2 / 3)) < 0.06:
            thirds += 1
        elif min(abs(f - 0.5), abs(f - 0.25), abs(f - 0.75)) < 0.05:
            halves += 1
    return thirds / max(1, thirds + halves), thirds


PERIODS = (2, 3, 4, 6, 8, 12, 16)


def _zscores(accent: np.ndarray, contrasts: dict[int, float], n_perm: int = 300) -> dict[int, float]:
    """周期ごとの「最良位相のコントラスト」を、アクセントを並べ替えた帰無分布で標準化する。
    周期が長いほど位相の選び方が増えて値が大きく出る偏りを打ち消す。"""
    rng = np.random.default_rng(12345)
    null = {p: [] for p in contrasts}
    for _ in range(n_perm):
        sh = rng.permutation(accent)
        for p in contrasts:
            null[p].append(_best_phase(sh, p)[0])
    out = {}
    for p, v in contrasts.items():
        arr = np.asarray(null[p])
        out[p] = float((v - arr.mean()) / (arr.std() + 1e-9))
    return out


def structure_z(beats: list[float] | np.ndarray, notes: list[dict]) -> float:
    """拍の列が小節構造（2〜12拍周期のアクセント）をどれだけはっきり持つか（最大の z 値）。"""
    b = np.asarray(beats, dtype=float)
    if b.size < 12:
        return 0.0
    acc = _accent(b, notes)
    C = {p: _best_phase(acc, p)[0] for p in (2, 3, 4, 6, 12)}
    return max(_zscores(acc, C, 120).values())


def detect_meter(beats: list[float], notes: list[dict], forced: str | None = None) -> dict:
    """拍子を決め、楽譜の拍に使う時刻列を返す。

    戻り値:
      ts        "4/4" | "3/4" | "2/4" | "6/8" | "12/8"
      beats     楽譜の1拍（単純拍子=4分、複合拍子=付点4分）の時刻列
      beatLen   1拍が4分音符いくつ分か（1 または 1.5）
      perBar    1小節の拍数（beats の単位）
      phase     beats の何番目が小節頭か
    周期ごとの「最良位相のアクセント差」を、アクセントを並べ替えた帰無分布で z 値にして比べる。
    複合拍子は2通りの取られ方がある：
      ・拍追跡が8分音符を拾った（速い拍で 6・12 拍周期）→ 3つずつ束ねて付点4分に
      ・拍追跡が付点4分を拾った（遅い拍の中が 3 分割）→ そのまま付点4分
    """
    b = np.asarray(beats, dtype=float)
    if b.size < 8:
        ts = forced or "4/4"
        if ts in COMPOUND_METERS:
            return {"ts": ts, "beats": b.tolist(), "beatLen": 1.5, "perBar": COMPOUND_METERS[ts], "phase": 0}
        return {"ts": ts, "beats": b.tolist(), "beatLen": 1.0, "perBar": SIMPLE_METERS.get(ts, 4), "phase": 0}
    accent = _accent(b, notes)
    tracked_bpm = 60.0 / float(np.median(np.diff(b)))
    tri, n_tri = _triplet_ratio(b, notes)
    C = {p: _best_phase(accent, p) for p in PERIODS}
    c = _zscores(accent, {p: v[0] for p, v in C.items()})

    def eighth_level(per_bar_eighths: int) -> dict:
        p = C[per_bar_eighths][1]
        grouped = b[p % 3 :: 3]
        per = per_bar_eighths // 3
        return {"ts": "12/8" if per == 4 else "6/8", "beats": grouped.tolist(), "beatLen": 1.5, "perBar": per, "phase": (p // 3) % per}

    def dotted_level(per: int | None) -> dict:
        if per is None:
            per = 4 if c[4] > c[2] + 0.5 or (tracked_bpm < 52 and c[4] >= c[2] - 0.5) else 2
        return {"ts": "12/8" if per == 4 else "6/8", "beats": b.tolist(), "beatLen": 1.5, "perBar": per, "phase": C[per][1]}

    fast = tracked_bpm >= 84
    if forced in COMPOUND_METERS:
        per = COMPOUND_METERS[forced]
        if fast and not (tri >= 0.5 and n_tri >= 10 and tracked_bpm < 100):
            return eighth_level(per * 3)
        return dotted_level(per)
    if forced is None:
        weak3 = c[3] < 2.0
        if fast and weak3 and c[12] >= 1.5 and c[12] > max(c[4], c[8], c[16]) + 0.8:
            return eighth_level(12)
        if fast and weak3 and c[6] >= 1.5 and c[6] > max(c[4], c[8]) + 0.8:
            return eighth_level(6)
        if tracked_bpm < 80 and tri >= 0.5 and n_tri >= 10:
            return dotted_level(None)
    if forced in SIMPLE_METERS:
        per = SIMPLE_METERS[forced]
        return {"ts": forced, "beats": b.tolist(), "beatLen": 1.0, "perBar": per, "phase": C[per][1]}
    s4 = max(c[4], c[8])
    s3 = max(c[3], c[6])
    # 根拠が弱いとき（短い曲・アクセントの薄い曲）は、いちばん多い 4/4 にしておく
    if s3 > s4 + 0.5 and s3 >= 1.5:
        return {"ts": "3/4", "beats": b.tolist(), "beatLen": 1.0, "perBar": 3, "phase": C[3][1]}
    return {"ts": "4/4", "beats": b.tolist(), "beatLen": 1.0, "perBar": 4, "phase": C[4][1]}

def first_strong_onset(y: np.ndarray, sr: int) -> float:
    import librosa

    on = librosa.onset.onset_detect(y=y, sr=sr, hop_length=HOP, units="time")
    return float(on[0]) if len(on) else 0.0
