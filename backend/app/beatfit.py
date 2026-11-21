"""採譜した音符の立ち上がりに拍の格子を合わせ込む（ビートの精密化）。

音声のエネルギー変化から取った拍（librosa）は、ピアノのように打楽器の無い曲や
ルバートのある演奏では 30〜80ms ずれたり揺れたりする。拍がずれていると
同じ「16分音符の位置」の音が別々の位置に量子化され、楽譜が汚くなる。

ここでは:
 1. 音符の立ち上がり（10ms 精度）から打点エンベロープを作り直し、拍を追跡し直す
 2. 各拍の前後の音符を「最寄りの 16分 / 3連の格子点」に当てたときの時間ずれを測り、
    拍をそのずれの重み付き中央値だけ動かす
 3. 拍の並びを2階差分の罰（テンポが急に変わらない）で滑らかにする
 を数回くり返す（グリッドを音に吸着させる EM 的な反復）。
複数の候補（元の拍・作り直した拍・それぞれ滑らかにしたもの・一定テンポ）を作り、
「音符が格子にどれだけ乗るか」で一番よいものを選ぶ。
"""
from __future__ import annotations

import numpy as np

FPS = 200  # 打点エンベロープの分解能（5ms）


def onset_envelope(notes: list[dict], duration: float) -> np.ndarray:
    n = int(duration * FPS) + FPS
    env = np.zeros(n)
    for x in notes:
        i = int(round(x["on"] * FPS))
        if 0 <= i < n:
            w = (x.get("vel", 80) / 127.0) * (1.5 if x["pitch"] < 55 else 1.0)
            env[i] += w
    # σ=10ms のガウスでぼかす
    k = np.exp(-0.5 * (np.arange(-6, 7) / 2.0) ** 2)
    return np.convolve(env, k / k.sum(), mode="same")


def track_on_notes(notes: list[dict], duration: float, start_bpm: float) -> np.ndarray:
    import librosa

    env = onset_envelope(notes, duration)
    # テンポは音声側の推定に固定し、拍の「位置」だけを音符に合わせて取り直す
    # （テンポまで自由にすると拍を余分に挟んで格子への乗りだけ良く見えてしまう）
    _tempo, beats = librosa.beat.beat_track(
        onset_envelope=env, sr=FPS, hop_length=1, bpm=start_bpm, tightness=300, units="time", trim=False
    )
    return np.asarray(beats, dtype=float)


def _residuals(beats: np.ndarray, ons: np.ndarray, w: np.ndarray):
    """各拍について、近くの音符を最寄りの格子点（16分 or 3連）に当てたときの時間ずれ。"""
    n = beats.size
    corr = np.zeros(n)
    conf = np.zeros(n)
    if n < 3:
        return corr, conf
    ibi = np.diff(beats)
    ibi = np.concatenate([ibi, ibi[-1:]])
    j0 = 0
    for i in range(n):
        lo = beats[i] - 0.5 * ibi[max(0, i - 1)]
        hi = beats[i] + 0.5 * ibi[i]
        while j0 < ons.size and ons[j0] < lo:
            j0 += 1
        j = j0
        rs, ws = [], []
        while j < ons.size and ons[j] < hi:
            pos = (ons[j] - beats[i]) / ibi[i]
            best = None
            for grid, gw in ((4, 1.0), (3, 0.8)):
                k = round(pos * grid)
                r = (pos - k / grid) * ibi[i]
                weight = gw * (1.0 if k == 0 else 0.6 if (grid == 4 and k % 2 == 0) else 0.4)
                if best is None or abs(r) < abs(best[0]):
                    best = (r, weight)
            rs.append(best[0])
            ws.append(best[1] * w[j])
            j += 1
        if rs:
            rs = np.asarray(rs)
            ws = np.asarray(ws)
            order = np.argsort(rs)
            cw = np.cumsum(ws[order])
            med = rs[order][np.searchsorted(cw, cw[-1] / 2)]
            corr[i] = med
            conf[i] = float(ws.sum())
    return corr, conf


def _smooth(y: np.ndarray, w: np.ndarray, lam: float) -> np.ndarray:
    """min Σ w_i (x_i - y_i)^2 + lam Σ (x_{i+1} - 2x_i + x_{i-1})^2 を解く（五重対角の連立方程式）。"""
    n = y.size
    if n < 4:
        return y.copy()
    try:
        from scipy.sparse import diags
        from scipy.sparse.linalg import spsolve
    except ImportError:  # ブラウザ版（scipy なし）
        from .npcompat import smooth_second_diff

        return smooth_second_diff(y, np.maximum(w, 1e-3), lam)
    D = diags([1.0, -2.0, 1.0], [0, 1, 2], shape=(n - 2, n))
    W = diags(np.maximum(w, 1e-3))
    A = (W + lam * (D.T @ D)).tocsc()
    return spsolve(A, W @ y)


def refine(beats: np.ndarray, notes: list[dict], iters: int = 8, lam: float = 4.0) -> np.ndarray:
    ons = np.array(sorted(x["on"] for x in notes))
    order = np.argsort([x["on"] for x in notes])
    w = np.array([notes[i].get("vel", 80) / 127.0 for i in order])
    b = np.asarray(beats, dtype=float).copy()
    for _ in range(iters):
        corr, conf = _residuals(b, ons, w)
        target = b + corr
        weight = np.minimum(conf, 3.0)
        b = _smooth(target, weight, lam)
        b = np.maximum.accumulate(b + np.arange(b.size) * 1e-6)
    return b


def constant_fit(beats: np.ndarray) -> np.ndarray:
    """一定テンポの直線あてはめ（揺れの小さい演奏・クリック録音向け）。"""
    i = np.arange(beats.size)
    a, c = np.polyfit(i, beats, 1)
    return a * i + c


def alignment(beats: np.ndarray, notes: list[dict]) -> float:
    """音符の立ち上がりが 16分/3連 の格子からどれだけずれるか（拍単位の平均、小さいほどよい）。"""
    b = np.asarray(beats, dtype=float)
    if b.size < 3:
        return 1.0
    on = np.array([x["on"] for x in notes])
    w = np.array([x.get("vel", 80) for x in notes], dtype=float)
    idx = np.interp(on, b, np.arange(b.size))
    m = (on >= b[0]) & (on <= b[-1])
    if not m.any():
        return 1.0
    f = idx[m] - np.floor(idx[m])
    d4 = np.abs(f * 4 - np.round(f * 4)) / 4
    d3 = np.abs(f * 3 - np.round(f * 3)) / 3
    d = np.minimum(d4, d3 * 1.15)
    return float(np.average(d, weights=w[m]))


def best_beats(audio_beats: list[float], notes: list[dict], duration: float, dnn_beats: list[float] | None = None) -> dict:
    """候補の拍列を作り、格子への乗り（と拍数のもっともらしさ）で一番よいものを返す。"""
    from .rhythm import _extend

    piano_notes = [x for x in notes if x.get("role") != "melody"] or notes
    a = np.asarray(audio_beats, dtype=float)
    ibi = float(np.median(np.diff(a))) if a.size > 2 else 0.5
    cands: dict[str, np.ndarray] = {"audio": a}
    try:
        nb = track_on_notes(piano_notes, duration, 60.0 / ibi)
        if nb.size > 8:
            # 音声側と同じ拍の細かさに合わせる（倍・半分の取り違えをそろえる）
            r = float(np.median(np.diff(nb))) / ibi
            if r < 0.7:
                nb = nb[::2]
            elif r > 1.4:
                nb = np.sort(np.concatenate([nb, (nb[:-1] + nb[1:]) / 2]))
            cands["notes"] = _extend(nb, duration)
    except Exception:
        pass
    if dnn_beats is not None and len(dnn_beats) > 8:
        # 深層学習（Beat This!）の拍。拍の細かさを音声側にそろえてから候補に入れる
        db = np.asarray(dnn_beats, dtype=float)
        r = float(np.median(np.diff(db))) / ibi
        if 0.4 < r < 0.7:
            db = db[::2]
        elif 1.4 < r < 2.6:
            db = np.sort(np.concatenate([db, (db[:-1] + db[1:]) / 2]))
        cands["dnn"] = _extend(db, duration)
    # 拍の数（テンポ）が音声側と 6% 以上違う候補は、格子への乗りを比べる土俵に上げない
    for k in list(cands):
        v = cands[k]
        if v.size > 2 and abs(float(np.median(np.diff(v))) / ibi - 1) > 0.06:
            del cands[k]
    for k in list(cands):
        cands[k + "+fit"] = refine(cands[k], piano_notes)
    for k in [k for k in cands if k.endswith("+fit")]:
        c = constant_fit(cands[k])
        # 一定テンポは、揺れの小さい演奏でだけ候補にする
        if float(np.abs(c - cands[k]).max()) < 0.06:
            cands[k + "+const"] = c
    # 格子への乗り（alignment）は、拍が16分ひとつ分ずれても同じ値になる。
    # 乗りが十分よい候補の中から、小節構造（拍周期のアクセント）が一番はっきりした列を選ぶ。
    from .rhythm import structure_z

    scores = {k: alignment(v, piano_notes) for k, v in cands.items()}
    best_align = min(scores.values())
    ok = [k for k in cands if scores[k] <= best_align * 1.35 + 0.004]
    struct = {k: structure_z(cands[k], notes) for k in ok}
    best = max(ok, key=lambda k: (round(struct[k], 1), -scores[k]))
    return {
        "beats": cands[best].tolist(),
        "method": best,
        "scores": {k: round(v, 4) for k, v in scores.items()},
        "structure": {k: round(v, 2) for k, v in struct.items()},
    }