"""ByteDance ピアノ採譜モデルの出力（フレームごとの確率）→ 音符・ペダルの変換。

piano_transcription_inference（Qiuqiang Kong, MIT License）の
RegressionPostProcessor と piano_vad を移植したもの。元はフレームごとの
Python ループで、ブラウザ版（Pyodide）では数十秒かかるので、
同じ判定を numpy と「次の出来事の位置を探す」形に書き換えた。結果は元と同じ
（tools/check_postprocess.py で確認）。
"""
from __future__ import annotations

import numpy as np

FRAMES_PER_SECOND = 100
BEGIN_NOTE = 21
VELOCITY_SCALE = 128
SEGMENT_SAMPLES = 16000 * 10


def deframe(seg_outputs: np.ndarray) -> np.ndarray:
    """(区間数, 1001, …) → (フレーム, …)。PianoTranscription.deframe と同じ（区間は半分ずつ重なる）。"""
    if seg_outputs.shape[0] == 1:
        return seg_outputs[0]
    x = seg_outputs[:, :-1]
    n, seg = x.shape[:2]
    parts = [x[0, : int(seg * 0.75)]]
    for i in range(1, n - 1):
        parts.append(x[i, int(seg * 0.25) : int(seg * 0.75)])
    parts.append(x[-1, int(seg * 0.25) :])
    return np.concatenate(parts, axis=0)


def binarize(reg_output: np.ndarray, threshold: float, neighbour: int):
    """get_binarized_output_from_regression（2次元 (frames, classes) / 1次元どちらも可）"""
    x = np.asarray(reg_output)
    squeeze = x.ndim == 1
    if squeeze:
        x = x[:, None]
    T = x.shape[0]
    binary = np.zeros_like(x)
    shift = np.zeros_like(x)
    if T > 2 * neighbour:
        n = np.arange(neighbour, T - neighbour)
        ok = x[n] > threshold
        for i in range(neighbour):
            ok &= ~(x[n - i] < x[n - i - 1])
            ok &= ~(x[n + i] < x[n + i + 1])
        rows, cols = np.nonzero(ok)
        nn = n[rows]
        xm, x0, xp = x[nn - 1, cols], x[nn, cols], x[nn + 1, cols]
        with np.errstate(divide="ignore", invalid="ignore"):
            s = np.where(xm > xp, (xp - xm) / (x0 - xp) / 2, (xp - xm) / (x0 - xm) / 2)
        binary[nn, cols] = 1
        shift[nn, cols] = s.astype(x.dtype)
    if squeeze:
        return binary[:, 0], shift[:, 0]
    return binary, shift


def _first_after(idx: np.ndarray, b: int) -> int | None:
    k = int(np.searchsorted(idx, b, side="right"))
    return int(idx[k]) if k < idx.size else None


def detect_notes_one(frame, onset, onset_shift, offset, offset_shift, velocity, frame_threshold) -> list:
    """note_detection_with_onset_offset_regress と同じ結果を、出来事の位置探索で求める。"""
    N = onset.shape[0]
    on_idx = np.flatnonzero(onset == 1)
    if on_idx.size == 0:
        return []
    fd_idx = np.flatnonzero(frame <= frame_threshold)
    off_idx = np.flatnonzero(offset == 1)
    out = []
    k = 0
    while k < on_idx.size:
        b = int(on_idx[k])
        nxt_on = int(on_idx[k + 1]) if k + 1 < on_idx.size else None
        if b == 0:  # 元の実装は bgn=0 を「なし」と扱うので、この音は作られない
            k += 1
            continue
        fd = _first_after(fd_idx, b)
        lim = min(b + 600, N - 1) if b < N - 1 else None
        # その反復で何が起きるか：同じ i では「次の打鍵」→「消音」→「上限」の順に判定される
        cands = []
        if nxt_on is not None:
            cands.append((nxt_on, 0))
        if fd is not None:
            cands.append((fd, 1))
        if lim is not None:
            cands.append((lim, 2))
        if not cands:
            k += 1
            continue
        i, kind = min(cands)
        if kind == 0:
            out.append([b, max(i - 1, 0), onset_shift[b], 0, velocity[b]])
            k += 1  # 次の打鍵からそのまま新しい音
            continue
        oo = _first_after(off_idx, b)
        if oo is not None and oo > i:
            oo = None
        if kind == 1:
            fin = oo if (oo and oo - b > i - oo) else i
        else:
            fin = i
        out.append([b, fin, onset_shift[b], offset_shift[fin], velocity[b]])
        # 消音・上限で終わったとき、次の打鍵は必ず i より後（同じか前なら打鍵で終わっている）。
        # 次の打鍵まで何もしないので、そのまま次の打鍵へ進めばよい
        k += 1
    out.sort(key=lambda p: p[0])
    return out


def _detect_notes_reference(frame, onset, onset_shift, offset, offset_shift, velocity, frame_threshold) -> list:
    """元の実装そのまま（検証用・遅い）"""
    output_tuples = []
    bgn = None
    frame_disappear = None
    offset_occur = None
    for i in range(onset.shape[0]):
        if onset[i] == 1:
            if bgn:
                fin = max(i - 1, 0)
                output_tuples.append([bgn, fin, onset_shift[bgn], 0, velocity[bgn]])
                frame_disappear, offset_occur = None, None
            bgn = i
        if bgn and i > bgn:
            if frame[i] <= frame_threshold and not frame_disappear:
                frame_disappear = i
            if offset[i] == 1 and not offset_occur:
                offset_occur = i
            if frame_disappear:
                if offset_occur and offset_occur - bgn > frame_disappear - offset_occur:
                    fin = offset_occur
                else:
                    fin = frame_disappear
                output_tuples.append([bgn, fin, onset_shift[bgn], offset_shift[fin], velocity[bgn]])
                bgn, frame_disappear, offset_occur = None, None, None
            if bgn and (i - bgn >= 600 or i == onset.shape[0] - 1):
                fin = i
                output_tuples.append([bgn, fin, onset_shift[bgn], offset_shift[fin], velocity[bgn]])
                bgn, frame_disappear, offset_occur = None, None, None
    output_tuples.sort(key=lambda pair: pair[0])
    return output_tuples


def detect_pedals(frame_output, offset_output, offset_shift_output, frame_threshold=0.5) -> list:
    """pedal_detection_with_onset_offset_regress（元のまま。1本だけなので速い）"""
    output_tuples = []
    bgn = None
    frame_disappear = None
    offset_occur = None
    for i in range(1, frame_output.shape[0]):
        if frame_output[i] >= frame_threshold and frame_output[i] > frame_output[i - 1]:
            if not bgn:
                bgn = i
        if bgn and i > bgn:
            if frame_output[i] <= frame_threshold and not frame_disappear:
                frame_disappear = i
            if offset_output[i] == 1 and not offset_occur:
                offset_occur = i
            if offset_occur:
                fin = offset_occur
                output_tuples.append([bgn, fin, 0.0, offset_shift_output[fin]])
                bgn, frame_disappear, offset_occur = None, None, None
            if frame_disappear and i - frame_disappear >= 10:
                fin = frame_disappear
                output_tuples.append([bgn, fin, 0.0, offset_shift_output[fin]])
                bgn, frame_disappear, offset_occur = None, None, None
    output_tuples.sort(key=lambda pair: pair[0])
    return output_tuples


def to_events(out: dict, onset_threshold=0.3, offset_threshold=0.3, frame_threshold=0.1, pedal_offset_threshold=0.2, reference: bool = False):
    """RegressionPostProcessor.output_dict_to_midi_events と同じ。

    out: reg_onset_output / reg_offset_output / frame_output / velocity_output: (T, 88)
         reg_pedal_onset_output / reg_pedal_offset_output / pedal_frame_output: (T,) または (T, 1)
    """
    onset_b, onset_s = binarize(out["reg_onset_output"], onset_threshold, 2)
    offset_b, offset_s = binarize(out["reg_offset_output"], offset_threshold, 4)
    det = _detect_notes_reference if reference else detect_notes_one
    tuples, midi = [], []
    frame = out["frame_output"]
    vel = out["velocity_output"]
    for k in range(frame.shape[-1]):
        t = det(frame[:, k], onset_b[:, k], onset_s[:, k], offset_b[:, k], offset_s[:, k], vel[:, k], frame_threshold)
        tuples += t
        midi += [k + BEGIN_NOTE] * len(t)
    notes = []
    if tuples:
        a = np.array(tuples)
        on = (a[:, 0] + a[:, 2]) / FRAMES_PER_SECOND
        off = (a[:, 1] + a[:, 3]) / FRAMES_PER_SECOND
        arr = np.stack((on, off, np.array(midi), a[:, 4]), axis=-1).astype(np.float32)
        notes = [
            {"onset_time": arr[i][0], "offset_time": arr[i][1], "midi_note": int(arr[i][2]), "velocity": int(arr[i][3] * VELOCITY_SCALE)}
            for i in range(arr.shape[0])
        ]
    pedals = []
    if "pedal_frame_output" in out:
        pf = np.asarray(out["pedal_frame_output"]).reshape(-1)
        po_b, po_s = binarize(np.asarray(out["reg_pedal_offset_output"]).reshape(-1), pedal_offset_threshold, 4)
        pt = detect_pedals(pf, po_b, po_s, 0.5)
        if pt:
            a = np.array(pt)
            on = (a[:, 0] + a[:, 2]) / FRAMES_PER_SECOND
            off = (a[:, 1] + a[:, 3]) / FRAMES_PER_SECOND
            arr = np.stack((on, off), axis=-1).astype(np.float32)
            pedals = [{"onset_time": arr[i, 0], "offset_time": arr[i, 1]} for i in range(len(arr))]
    return notes, pedals
