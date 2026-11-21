"""Basic Pitch のモデル出力（note / onset / contour）→ 音符イベント。

basic_pitch.note_creation（Copyright 2022 Spotify AB, Apache License 2.0）の
output_to_notes_polyphonic などを、librosa・scipy なしで動くように移植したもの。
ブラウザ版（Pyodide）用。判定は元と同じで、「メロディアの技」の最大値探しだけ
毎回全体を探さずに、値の大きい順の並びを先頭から使う形にして速くした
（値は 0 に消されるだけなので、結果は同じ）。
"""
from __future__ import annotations

import numpy as np

AUDIO_SAMPLE_RATE = 22050
FFT_HOP = 256
ANNOTATIONS_FPS = AUDIO_SAMPLE_RATE // FFT_HOP
AUDIO_WINDOW_LENGTH = 2
ANNOT_N_FRAMES = ANNOTATIONS_FPS * AUDIO_WINDOW_LENGTH
AUDIO_N_SAMPLES = AUDIO_SAMPLE_RATE * AUDIO_WINDOW_LENGTH - FFT_HOP
N_OVERLAPPING_FRAMES = 30
OVERLAP_LEN = N_OVERLAPPING_FRAMES * FFT_HOP
HOP_SIZE = AUDIO_N_SAMPLES - OVERLAP_LEN
MIDI_OFFSET = 21
MAX_FREQ_IDX = 87


def _hz_to_midi(f: float) -> float:
    return 12 * (np.log2(f) - np.log2(440.0)) + 69


def unwrap_output(output: np.ndarray, audio_original_length: int) -> np.ndarray:
    """(窓数, 172, n) → (フレーム, n)。basic_pitch.inference.unwrap_output と同じ。"""
    n_olap = int(0.5 * N_OVERLAPPING_FRAMES)
    if n_olap > 0:
        output = output[:, n_olap:-n_olap, :]
    n_frames = int(np.floor(audio_original_length * (ANNOTATIONS_FPS / AUDIO_SAMPLE_RATE)))
    return output.reshape(output.shape[0] * output.shape[1], output.shape[2])[:n_frames, :]


def model_frames_to_time(n_frames: int) -> np.ndarray:
    original_times = np.arange(n_frames) * FFT_HOP / float(AUDIO_SAMPLE_RATE)
    window_numbers = np.floor(np.arange(n_frames) / ANNOT_N_FRAMES)
    window_offset = (FFT_HOP / AUDIO_SAMPLE_RATE) * (ANNOT_N_FRAMES - (AUDIO_N_SAMPLES / FFT_HOP)) + 0.0018
    return original_times - (window_offset * window_numbers)


def _infered_onsets(onsets: np.ndarray, frames: np.ndarray, n_diff: int = 2) -> np.ndarray:
    diffs = []
    for n in range(1, n_diff + 1):
        fa = np.concatenate([np.zeros((n, frames.shape[1])), frames])
        diffs.append(fa[n:, :] - fa[:-n, :])
    frame_diff = np.min(diffs, axis=0)
    frame_diff[frame_diff < 0] = 0
    frame_diff[:n_diff, :] = 0
    frame_diff = np.max(onsets) * frame_diff / np.max(frame_diff)
    return np.max([onsets, frame_diff], axis=0)


def _argrelmax0(x: np.ndarray):
    """scipy.signal.argrelmax(x, axis=0)（order=1, mode='clip'：端は自分自身と比べるので極大にならない）"""
    res = np.zeros(x.shape, dtype=bool)
    if x.shape[0] >= 3:
        res[1:-1] = (x[1:-1] > x[2:]) & (x[1:-1] > x[:-2])
    return np.nonzero(res)


def output_to_notes_polyphonic(frames, onsets, onset_thresh, frame_thresh, min_note_len, infer_onsets, max_freq, min_freq, melodia_trick=True, energy_tol=11):
    n_frames = frames.shape[0]
    onsets = np.array(onsets, copy=True)
    frames = np.array(frames, copy=True)
    if max_freq is not None:
        i = int(np.round(_hz_to_midi(max_freq) - MIDI_OFFSET))
        onsets[:, i:] = 0
        frames[:, i:] = 0
    if min_freq is not None:
        i = int(np.round(_hz_to_midi(min_freq) - MIDI_OFFSET))
        onsets[:, :i] = 0
        frames[:, :i] = 0
    if infer_onsets:
        onsets = _infered_onsets(onsets, frames)

    peak_thresh_mat = np.zeros(onsets.shape)
    peaks = _argrelmax0(onsets)
    peak_thresh_mat[peaks] = onsets[peaks]
    onset_idx = np.where(peak_thresh_mat >= onset_thresh)
    onset_time_idx = onset_idx[0][::-1]
    onset_freq_idx = onset_idx[1][::-1]

    remaining_energy = np.zeros(frames.shape)
    remaining_energy[:, :] = frames[:, :]
    note_events = []
    for note_start_idx, freq_idx in zip(onset_time_idx, onset_freq_idx):
        if note_start_idx >= n_frames - 1:
            continue
        # 閾値を下回るフレームが energy_tol 回続く所を探す（元の while ループと同じ）
        col = remaining_energy[note_start_idx + 1 : n_frames - 1, freq_idx] < frame_thresh
        i = note_start_idx + 1
        k = 0
        for below in col:
            if k >= energy_tol:
                break
            k = k + 1 if below else 0
            i += 1
        i -= k
        if i - note_start_idx <= min_note_len:
            continue
        remaining_energy[note_start_idx:i, freq_idx] = 0
        if freq_idx < MAX_FREQ_IDX:
            remaining_energy[note_start_idx:i, freq_idx + 1] = 0
        if freq_idx > 0:
            remaining_energy[note_start_idx:i, freq_idx - 1] = 0
        amplitude = np.mean(frames[note_start_idx:i, freq_idx])
        note_events.append((note_start_idx, i, freq_idx + MIDI_OFFSET, amplitude))

    if melodia_trick:
        shape = remaining_energy.shape
        flat = remaining_energy.reshape(-1)
        # np.argmax は同じ値なら先に出る位置を返す → 安定ソートで同じ順に並べる
        cand = np.flatnonzero(flat > frame_thresh)
        order = cand[np.argsort(-flat[cand], kind="stable")]
        orig = flat[order].copy()
        p = 0
        while True:
            while p < order.size and flat[order[p]] != orig[p]:
                p += 1  # もう消された（0 になった）所は飛ばす
            if p >= order.size or not (orig[p] > frame_thresh):
                break
            i_mid, freq_idx = np.unravel_index(order[p], shape)
            remaining_energy[i_mid, freq_idx] = 0
            # 前向き
            i = i_mid + 1
            k = 0
            while i < n_frames - 1 and k < energy_tol:
                if remaining_energy[i, freq_idx] < frame_thresh:
                    k += 1
                else:
                    k = 0
                remaining_energy[i, freq_idx] = 0
                if freq_idx < MAX_FREQ_IDX:
                    remaining_energy[i, freq_idx + 1] = 0
                if freq_idx > 0:
                    remaining_energy[i, freq_idx - 1] = 0
                i += 1
            i_end = i - 1 - k
            # 後ろ向き
            i = i_mid - 1
            k = 0
            while i > 0 and k < energy_tol:
                if remaining_energy[i, freq_idx] < frame_thresh:
                    k += 1
                else:
                    k = 0
                remaining_energy[i, freq_idx] = 0
                if freq_idx < MAX_FREQ_IDX:
                    remaining_energy[i, freq_idx + 1] = 0
                if freq_idx > 0:
                    remaining_energy[i, freq_idx - 1] = 0
                i -= 1
            i_start = i + 1 + k
            if i_end - i_start <= min_note_len:
                continue
            amplitude = np.mean(frames[i_start:i_end, freq_idx])
            note_events.append((i_start, i_end, freq_idx + MIDI_OFFSET, amplitude))
    return note_events


def model_output_to_events(note, onset, contour_frames: int, onset_thresh, frame_thresh, min_note_len_ms, min_freq, max_freq, melodia_trick=True):
    """basic_pitch.inference.predict の後半と同じ：(開始秒, 終了秒, MIDI, 強さ0-1) の列。"""
    min_note_len = int(np.round(min_note_len_ms / 1000 * (AUDIO_SAMPLE_RATE / FFT_HOP)))
    est = output_to_notes_polyphonic(
        note, onset, onset_thresh=onset_thresh, frame_thresh=frame_thresh, infer_onsets=True,
        min_note_len=min_note_len, min_freq=min_freq, max_freq=max_freq, melodia_trick=melodia_trick,
    )
    times = model_frames_to_time(contour_frames)
    return [(times[a], times[b], p, amp) for a, b, p, amp in est]
