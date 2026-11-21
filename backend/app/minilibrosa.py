"""librosa のうち、このアプリが使う関数だけを numpy だけで書き直したもの。

ブラウザ版エンジン（Pyodide）では librosa が動かない（numba が無い）ので、
`sys.modules["librosa"] = minilibrosa` として差し替える。PC 版は本物の librosa を使う。
計算手順は librosa 1.0.1 と同じにしてあり、tools/check_minilibrosa.py で突き合わせる。

使っているもの:
  librosa.onset.onset_strength / onset_detect
  librosa.beat.beat_track
  librosa.feature.rms
  librosa.pyin
  librosa.hz_to_midi / midi_to_hz / resample
"""
from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np

__version__ = "1.0.1-lite"


def _tiny(x) -> float:
    x = np.asarray(x)
    dt = x.dtype if np.issubdtype(x.dtype, np.floating) or np.issubdtype(x.dtype, np.complexfloating) else np.float32
    return float(np.finfo(dt).tiny)


# --------------------------------------------------------------------------- 単位変換
def hz_to_midi(frequencies):
    f = np.asanyarray(frequencies)
    return 12 * (np.log2(f) - np.log2(440.0)) + 69


def midi_to_hz(notes):
    return 440.0 * (2.0 ** ((np.asanyarray(notes) - 69.0) / 12.0))


def frames_to_time(frames, *, sr=22050, hop_length=512):
    return (np.asanyarray(frames) * hop_length) / float(sr)


def _hz_to_mel(f):
    f = np.asanyarray(f, dtype=np.float64)
    f_sp = 200.0 / 3
    mels = f / f_sp
    min_log_hz = 1000.0
    min_log_mel = min_log_hz / f_sp
    logstep = np.log(6.4) / 27.0
    return np.where(f >= min_log_hz, min_log_mel + np.log(np.maximum(f, 1e-300) / min_log_hz) / logstep, mels)


def _mel_to_hz(m):
    m = np.asanyarray(m, dtype=np.float64)
    f_sp = 200.0 / 3
    freqs = f_sp * m
    min_log_hz = 1000.0
    min_log_mel = min_log_hz / f_sp
    logstep = np.log(6.4) / 27.0
    return np.where(m >= min_log_mel, min_log_hz * np.exp(logstep * (m - min_log_mel)), freqs)


def _mel_filter(sr, n_fft, n_mels=128, fmin=0.0, fmax=None):
    """librosa.filters.mel(htk=False, norm='slaney', dtype=float32)"""
    if fmax is None:
        fmax = sr / 2.0
    n_freq = 1 + n_fft // 2
    fftfreqs = np.linspace(0, float(sr) / 2, n_freq)
    mel_f = _mel_to_hz(np.linspace(_hz_to_mel(fmin), _hz_to_mel(fmax), n_mels + 2))
    fdiff = np.diff(mel_f)
    ramps = np.subtract.outer(mel_f, fftfreqs)
    weights = np.zeros((n_mels, n_freq))
    for i in range(n_mels):
        lower = -ramps[i] / fdiff[i]
        upper = ramps[i + 2] / fdiff[i + 1]
        weights[i] = np.maximum(0, np.minimum(lower, upper))
    enorm = 2.0 / (mel_f[2 : n_mels + 2] - mel_f[:n_mels])
    weights *= enorm[:, None]
    return weights.astype(np.float32)


# --------------------------------------------------------------------------- 信号処理の部品
def _frame(x: np.ndarray, frame_length: int, hop_length: int) -> np.ndarray:
    """librosa.util.frame(axis=-1): (..., frame_length, n_frames)"""
    n_frames = 1 + (x.shape[-1] - frame_length) // hop_length
    idx = np.arange(frame_length)[:, None] + hop_length * np.arange(n_frames)[None, :]
    return x[..., idx]


def _hann(n: int) -> np.ndarray:
    """周期的なハン窓（get_window('hann', n, fftbins=True)）"""
    return 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n) / n)


def _stft_power(y: np.ndarray, n_fft: int, hop_length: int) -> np.ndarray:
    """librosa.stft(center=True, pad_mode='constant', window='hann') の |X|^2。(1+n_fft/2, T)"""
    pad = n_fft // 2
    yp = np.pad(y.astype(np.float64), (pad, pad), mode="constant")
    frames = _frame(yp, n_fft, hop_length) * _hann(n_fft)[:, None]
    X = np.fft.rfft(frames, axis=0)
    return (X.real**2 + X.imag**2).astype(np.float32)


def _power_to_db(S, ref=1.0, amin=1e-10, top_db=80.0):
    S = np.asarray(S)
    log_spec = 10.0 * np.log10(np.maximum(amin, S))
    log_spec -= 10.0 * np.log10(np.maximum(amin, ref))
    if top_db is not None:
        log_spec = np.maximum(log_spec, log_spec.max() - top_db)
    return log_spec


def _autocorrelate(y: np.ndarray, max_size: int | None = None, axis: int = -1) -> np.ndarray:
    """FFT による自己相関（librosa.autocorrelate と同じ）"""
    n = y.shape[axis]
    if max_size is None:
        max_size = n
    n_pad = 1 << int(math.ceil(math.log2(max(2 * n - 1, 1))))
    f = np.fft.rfft(y, n=n_pad, axis=axis)
    ac = np.fft.irfft(f.real**2 + f.imag**2, n=n_pad, axis=axis)
    sl = [slice(None)] * y.ndim
    sl[axis] = slice(0, max_size)
    return ac[tuple(sl)]


def _localmax(x: np.ndarray) -> np.ndarray:
    """librosa.util.localmax: 内側は x[i] > x[i-1] かつ x[i] >= x[i+1]。
    先頭は numba stencil の境界なので常に False、末尾は x[-1] > x[-2]。"""
    out = np.zeros(x.shape, dtype=bool)
    if x.size >= 3:
        out[1:-1] = (x[1:-1] > x[:-2]) & (x[1:-1] >= x[2:])
    if x.size >= 2:
        out[-1] = x[-1] > x[-2]
    return out


def _localmin(x: np.ndarray) -> np.ndarray:
    out = np.zeros(x.shape, dtype=bool)
    if x.size >= 3:
        out[1:-1] = (x[1:-1] < x[:-2]) & (x[1:-1] <= x[2:])
    if x.size >= 2:
        out[-1] = x[-1] < x[-2]
    return out


# --------------------------------------------------------------------------- onset
def _onset_strength(y, sr=22050, hop_length=512, n_fft=2048, aggregate=None, lag=1, center=True, top_db=80.0):
    if aggregate is None:
        aggregate = np.mean
    S = _stft_power(np.asarray(y), n_fft, hop_length)
    mel = _mel_filter(sr, n_fft, n_mels=128, fmin=0.0, fmax=0.5 * sr)
    M = (mel.astype(np.float64) @ S.astype(np.float64)).astype(np.float32)
    S = _power_to_db(M, top_db=top_db)
    env = np.maximum(0.0, S[:, lag:] - S[:, :-lag])
    env = aggregate(env, axis=0)
    pad_width = lag + (n_fft // (2 * hop_length) if center else 0)
    env = np.pad(env, (int(pad_width), 0), mode="constant")
    if center:
        env = env[: S.shape[-1]]
    return env.astype(np.float32)


def _peak_pick(x, pre_max, post_max, pre_avg, post_avg, delta, wait):
    pre_max, post_max, pre_avg, post_avg, wait = (int(np.ceil(v)) for v in (pre_max, post_max, pre_avg, post_avg, wait))
    n_total = x.shape[0]
    peaks = np.zeros(n_total, dtype=bool)
    peaks[0] = x[0] >= np.max(x[: min(post_max, n_total)])
    peaks[0] &= x[0] >= np.mean(x[: min(post_avg, n_total)]) + delta
    n = wait + 1 if peaks[0] else 1
    while n < n_total:
        maxn = np.max(x[max(0, n - pre_max) : min(n + post_max, n_total)])
        peaks[n] = x[n] == maxn
        if not peaks[n]:
            n += 1
            continue
        avgn = np.mean(x[max(0, n - pre_avg) : min(n + post_avg, n_total)])
        peaks[n] &= x[n] >= avgn + delta
        if not peaks[n]:
            n += 1
            continue
        n += wait + 1
    return np.flatnonzero(peaks)


def _match_events_left(events_from: np.ndarray, events_to: np.ndarray) -> np.ndarray:
    """librosa.util.match_events(left=True, right=False)"""
    from_idx = np.argsort(events_from, kind="stable")
    sorted_from = events_from[from_idx]
    to_idx = np.argsort(events_to, kind="stable")
    sorted_to = events_to[to_idx]
    matching = np.searchsorted(sorted_to, sorted_from)
    out = np.empty(len(events_from), dtype=np.int64)
    for ind, middle_ind in enumerate(matching):
        x = sorted_from[ind]
        if middle_ind == len(sorted_to):
            middle_ind -= 1
        left_flag = middle_ind > 0
        left_ind = middle_ind - 1
        mid_diff = abs(sorted_to[middle_ind] - x)
        left_diff = abs(sorted_to[left_ind] - x) if left_flag else 0
        if left_flag and ((sorted_to[middle_ind] > x) or (left_diff < mid_diff)):
            out[ind] = to_idx[left_ind]
        else:
            out[ind] = to_idx[middle_ind]
    sol = np.empty_like(out)
    sol[from_idx] = out
    return sol


def _onset_backtrack(events, energy):
    minima = np.flatnonzero((energy[1:-1] <= energy[:-2]) & (energy[1:-1] < energy[2:]))
    minima = np.unique(np.concatenate([[0], np.clip(1 + minima, 0, None)])).astype(int)
    return minima[_match_events_left(np.asarray(events), minima)]


def _onset_detect(*, y=None, sr=22050, onset_envelope=None, hop_length=512, backtrack=False, energy=None, units="frames", normalize=True, **kw):
    if onset_envelope is None:
        onset_envelope = _onset_strength(y, sr=sr, hop_length=hop_length)
    env = np.asarray(onset_envelope, dtype=np.float32)
    if normalize:
        env = env - env.min()
        env = env / (env.max() + _tiny(env))
    if not env.any() or not np.all(np.isfinite(env)):
        onsets = np.array([], dtype=int)
    else:
        kw.setdefault("pre_max", 0.03 * sr // hop_length)
        kw.setdefault("post_max", 0.00 * sr // hop_length + 1)
        kw.setdefault("pre_avg", 0.10 * sr // hop_length)
        kw.setdefault("post_avg", 0.10 * sr // hop_length + 1)
        kw.setdefault("wait", 0.03 * sr // hop_length)
        kw.setdefault("delta", 0.07)
        onsets = _peak_pick(env, **kw)
        if backtrack and onsets.size:
            onsets = _onset_backtrack(onsets, env if energy is None else energy)
    if units == "time":
        return frames_to_time(onsets, sr=sr, hop_length=hop_length)
    if units == "samples":
        return onsets * hop_length
    return onsets


def onset_strength(*, y=None, sr=22050, hop_length=512, aggregate=None, n_fft=2048, **_kw):
    return _onset_strength(y, sr=sr, hop_length=hop_length, n_fft=n_fft, aggregate=aggregate)


onset = SimpleNamespace(onset_strength=onset_strength, onset_detect=_onset_detect, onset_backtrack=_onset_backtrack)


# --------------------------------------------------------------------------- tempo / beat
def _tempo(onset_envelope, sr, hop_length, start_bpm=120.0, std_bpm=1.0, ac_size=8.0, max_tempo=320.0):
    win_length = int(int(ac_size * sr) // hop_length)
    env = np.asarray(onset_envelope, dtype=np.float64)
    n = env.shape[-1]
    half = win_length // 2
    padded = np.pad(env, (half, half), mode="linear_ramp", end_values=(0, 0))
    frames = _frame(padded, win_length, 1)[:, :n]
    frames = frames * _hann(win_length)[:, None]
    ac = _autocorrelate(frames, axis=0)
    norm = np.max(np.abs(ac), axis=0, keepdims=True)
    norm[norm < _tiny(ac)] = 1.0
    tg = ac / norm
    tg = np.mean(tg, axis=-1, keepdims=True)
    bpms = np.zeros(win_length)
    bpms[0] = np.inf
    bpms[1:] = 60.0 * sr / (hop_length * np.arange(1.0, win_length))
    with np.errstate(divide="ignore", invalid="ignore"):
        logprior = -0.5 * ((np.log2(bpms) - np.log2(start_bpm)) / std_bpm) ** 2
    if max_tempo is not None:
        max_idx = int(np.argmax(bpms < max_tempo))
        logprior[:max_idx] = -np.inf
    best = np.argmax(np.log1p(1e6 * tg) + logprior[:, None], axis=0)
    return np.take(bpms, best)


def _beat_local_score(env: np.ndarray, fpb: float) -> np.ndarray:
    """__beat_local_score（静的テンポ）。

    DP は同点（左右対称な罰で同じ値になる候補）がよく出るので、丸め誤差まで numba 版と
    そろえる：k の昇順に 1 項ずつ、env の型（float32 なら float32）へ丸めながら足す。
    numba 版と同じく i+K//2 < K のときは onset[0] の項を含めない。
    """
    N = env.size
    window = np.exp(-0.5 * (np.arange(-fpb, fpb + 1) * 32.0 / fpb) ** 2)
    K = window.size
    i = np.arange(N)
    lo = np.maximum(0, i + K // 2 - N + 1)
    hi = np.minimum(i + K // 2, K)
    x64 = env.astype(np.float64)
    acc = np.zeros(N, dtype=env.dtype)
    for k in range(K):
        valid = (k >= lo) & (k < hi)
        if not valid.any():
            continue
        idx = np.clip(i + K // 2 - k, 0, N - 1)
        nxt = (acc.astype(np.float64) + window[k] * x64[idx]).astype(env.dtype)
        acc = np.where(valid, nxt, acc)
    return acc


def _beat_track_dp(localscore: np.ndarray, fpb: float, tightness: float):
    # numba の guvectorize は frames_per_beat(float64) に合わせて float64 版の署名を選ぶので、
    # localscore が float32 でも DP は float64 で回る
    localscore = localscore.astype(np.float64)
    n = localscore.size
    backlink = np.full(n, -1, dtype=np.int64)
    cumscore = np.zeros(n, dtype=np.float64)
    score_thresh = 0.01 * localscore.max()
    first_beat = True
    cumscore[0] = localscore[0]
    near = int(round(fpb / 2))
    span = np.arange(near, int(2 * fpb + 1))  # i - loc の値（loc は i-near から i-2fpb まで）
    tight = float(np.float32(tightness))
    d = np.log(span.astype(np.float64)) - np.log(float(fpb))
    penalty = tight * (d * d)
    cs64 = np.zeros(n)
    for i in range(n):
        score_i = localscore[i]
        lo = i - near
        if lo >= 0:
            m = min(span.size, lo + 1)
            locs_vals = cs64[lo - np.arange(m)] if m else np.zeros(0)
            scores = locs_vals - penalty[:m]
            k = int(np.argmax(scores))
            best_score = scores[k]
            beat_location = lo - k
            cumscore[i] = score_i + best_score
        else:
            beat_location = -1
            cumscore[i] = score_i
        cs64[i] = float(cumscore[i])
        if first_beat and score_i < score_thresh:
            backlink[i] = -1
        else:
            backlink[i] = beat_location
            first_beat = False
    return backlink, cumscore


def _last_beat(cumscore: np.ndarray) -> int:
    mask = ~_localmax(cumscore)
    vals = cumscore[~mask]
    threshold = 0.5 * float(np.median(vals)) if vals.size else 0.0
    n = cumscore.size - 1
    out = n
    while n >= 0:
        if not mask[n] and cumscore[n] >= threshold:
            out = n
            break
        n -= 1
    return out


def beat_track(*, y=None, sr=22050, onset_envelope=None, hop_length=512, start_bpm=120.0, tightness=100, trim=True, bpm=None, units="frames", **_kw):
    if onset_envelope is None:
        onset_envelope = _onset_strength(y, sr=sr, hop_length=hop_length, aggregate=np.median)
    env = np.asarray(onset_envelope)
    if not env.any():
        return 0.0, np.array([], dtype=int)
    if bpm is None:
        bpm = _tempo(env, sr, hop_length, start_bpm=start_bpm)
    bpm_arr = np.atleast_1d(bpm)
    fpb = float(np.round(float(sr) / hop_length * 60.0 / float(bpm_arr[0])))
    norm = env.std(ddof=1)
    local = _beat_local_score((env / (norm + _tiny(env))).astype(env.dtype), fpb)
    backlink, cumscore = _beat_track_dp(local, fpb, tightness)
    tail = _last_beat(cumscore)
    beats = np.zeros(env.size, dtype=bool)
    n = tail
    while n >= 0:
        beats[n] = True
        n = backlink[n]
    # __trim_beats（trim=False でも localscore<=0 の頭と尻は落とす）
    if trim:
        w = np.hanning(5)
        smooth = np.convolve(local[beats], w)[len(w) // 2 : len(local) + len(w) // 2]
        threshold = 0.5 * ((smooth**2).mean() ** 0.5)
    else:
        threshold = 0.0
    n = 0
    while n < local.size and local[n] <= threshold:
        beats[n] = False
        n += 1
    n = local.size - 1
    while n >= 0 and local[n] <= threshold:
        beats[n] = False
        n -= 1
    frames = np.flatnonzero(beats)
    if units == "time":
        return bpm, frames_to_time(frames, sr=sr, hop_length=hop_length)
    if units == "samples":
        return bpm, frames * hop_length
    return bpm, frames


beat = SimpleNamespace(beat_track=beat_track)


# --------------------------------------------------------------------------- feature
def rms(*, y=None, frame_length=2048, hop_length=512, center=True, pad_mode="constant", **_kw):
    y = np.asarray(y)
    if center:
        y = np.pad(y, (frame_length // 2, frame_length // 2), mode=pad_mode)
    x = _frame(y, frame_length, hop_length)
    power = np.mean(np.abs(x) ** 2, axis=-2, keepdims=True)
    return np.sqrt(power)


feature = SimpleNamespace(rms=rms)


# --------------------------------------------------------------------------- pYIN
def _beta_cdf_int(x, a: int, b: int):
    """整数パラメータの正則化不完全ベータ関数 I_x(a, b)"""
    x = np.asarray(x, dtype=np.float64)
    n = a + b - 1
    out = np.zeros_like(x)
    for j in range(a, n + 1):
        out += math.comb(n, j) * x**j * (1 - x) ** (n - j)
    return out


def _boltzmann_pmf(k, lam, N):
    k = np.asarray(k)
    N = np.asarray(N)
    with np.errstate(divide="ignore", invalid="ignore"):
        p = (1 - np.exp(-lam)) * np.exp(-lam * k) / (1 - np.exp(-lam * N))
    return np.where((k >= 0) & (k < N), p, 0.0)


def _triang(M: int) -> np.ndarray:
    """scipy.signal.windows.triang(M, sym=True)"""
    n = np.arange(1, (M + 1) // 2 + 1)
    if M % 2 == 0:
        w = (2 * n - 1.0) / M
        return np.concatenate([w, w[::-1]])
    w = 2 * n / (M + 1.0)
    return np.concatenate([w, w[-2::-1]])


def _transition_local(n_states: int, width: int) -> np.ndarray:
    T = np.zeros((n_states, n_states))
    win = _triang(width)
    lpad = (n_states - width) // 2
    base = np.zeros(n_states)
    base[lpad : lpad + width] = win
    for i in range(n_states):
        row = np.roll(base, n_states // 2 + i + 1)
        row[min(n_states, i + width // 2 + 1) :] = 0
        row[: max(0, i - width // 2)] = 0
        T[i] = row
    return T / T.sum(axis=1, keepdims=True)


def _viterbi_sparse(log_prob: np.ndarray, log_trans: np.ndarray, log_p_init: np.ndarray, log_thr: float) -> np.ndarray:
    """librosa.sequence._viterbi（transition_min_prob あり）。log_prob: (T, S)"""
    n_steps, n_states = log_prob.shape
    preds = [np.flatnonzero(log_trans[:, j] >= log_thr) for j in range(n_states)]
    width = max(len(p) for p in preds)
    P = np.zeros((n_states, width), dtype=np.int64)
    valid = np.zeros((n_states, width), dtype=bool)
    for j, p in enumerate(preds):
        P[j, : len(p)] = p
        valid[j, : len(p)] = True
    LT = np.where(valid, log_trans[P, np.arange(n_states)[:, None]], -np.inf)
    value = log_prob[0] + log_p_init
    ptr = np.zeros((n_steps, n_states), dtype=np.int64)
    rows = np.arange(n_states)
    for t in range(1, n_steps):
        cost = value[P] + LT
        k = np.argmax(cost, axis=1)
        ptr[t] = P[rows, k]
        value = log_prob[t] + cost[rows, k]
    state = np.zeros(n_steps, dtype=np.int64)
    state[-1] = int(np.argmax(value))
    for t in range(n_steps - 2, -1, -1):
        state[t] = ptr[t + 1, state[t + 1]]
    return state


def _cmnd(y_frames: np.ndarray, min_period: int, max_period: int) -> np.ndarray:
    acf = _autocorrelate(y_frames, max_size=max_period + 1, axis=-2)
    yin = np.cumsum(np.square(y_frames), axis=-2)
    yin[0, :] = 0
    k = slice(1, max_period + 1)
    yin[k, :] = 2 * (acf[0:1, :] - acf[k, :]) - yin[: k.stop - 1, :]
    num = yin[min_period : max_period + 1, :]
    k_range = np.arange(1, max_period + 1)[:, None]
    cmean = np.cumsum(yin[k, :], axis=-2) / k_range
    den = cmean[min_period - 1 : max_period, :]
    return num / (den + _tiny(den))


def pyin(y, *, fmin, fmax, sr=22050, frame_length=2048, hop_length=None, n_thresholds=100, beta_parameters=(2, 18), boltzmann_parameter=2, resolution=0.1, max_transition_rate=35.92, switch_prob=0.01, no_trough_prob=0.01, fill_na=np.nan, center=True, pad_mode="constant", transition_min_prob=1e-4, **_kw):
    if hop_length is None:
        hop_length = frame_length // 4
    y = np.asarray(y)
    if center:
        y = np.pad(y, (frame_length // 2, frame_length // 2), mode=pad_mode)
    y_frames = _frame(y, frame_length, hop_length).astype(np.float64)
    min_period = int(np.floor(sr / fmax))
    max_period = min(int(np.ceil(sr / fmin)), frame_length - 1)
    yin_frames = _cmnd(y_frames, min_period, max_period)
    thresholds = np.linspace(0, 1, n_thresholds + 1)
    beta_probs = np.diff(_beta_cdf_int(thresholds, int(beta_parameters[0]), int(beta_parameters[1])))
    n_bins_per_semitone = int(np.ceil(1.0 / resolution))
    n_pitch_bins = int(np.floor(12 * n_bins_per_semitone * np.log2(fmax / fmin))) + 1

    # __pyin_helper
    yin_probs = np.zeros_like(yin_frames)
    for i in range(yin_frames.shape[1]):
        yf = yin_frames[:, i]
        is_trough = _localmin(yf)
        is_trough[0] = yf[0] < yf[1]
        (trough_index,) = np.nonzero(is_trough)
        if len(trough_index) == 0:
            continue
        heights = yf[trough_index]
        tt = np.less.outer(heights, thresholds[1:])
        pos = np.cumsum(tt, axis=0) - 1
        n_troughs = np.count_nonzero(tt, axis=0)
        prior = _boltzmann_pmf(pos, boltzmann_parameter, n_troughs)
        prior[~tt] = 0
        probs = prior.dot(beta_probs)
        gmin = int(np.argmin(heights))
        n_below = np.count_nonzero(~tt[gmin, :])
        probs[gmin] += no_trough_prob * np.sum(beta_probs[:n_below])
        yin_probs[trough_index, i] = probs
    yin_period, frame_index = np.nonzero(yin_probs)
    n_periods = yin_frames.shape[0]
    interior = (yin_period > 0) & (yin_period < n_periods - 1)
    p_safe = np.clip(yin_period, 1, n_periods - 2)
    vc = yin_frames[p_safe, frame_index]
    vp = yin_frames[p_safe + 1, frame_index]
    vm = yin_frames[p_safe - 1, frame_index]
    a = vp + vm - 2.0 * vc
    b = 0.5 * (vp - vm)
    ok = interior & (np.abs(b) < np.abs(a))
    shift = np.zeros_like(b)
    np.divide(-b, a, out=shift, where=ok)
    period = min_period + yin_period + shift
    f0c = sr / period
    bin_index = 12 * n_bins_per_semitone * np.log2(f0c / fmin)
    bin_index = np.clip(np.round(bin_index), 0, n_pitch_bins).astype(int)
    obs = np.zeros((2 * n_pitch_bins, yin_frames.shape[1]))
    obs[bin_index, frame_index] = yin_probs[yin_period, frame_index]
    voiced_prob = np.clip(np.sum(obs[:n_pitch_bins, :], axis=0, keepdims=True), 0, 1)
    obs[n_pitch_bins:, :] = (1 - voiced_prob) / n_pitch_bins

    max_semitones = round(max_transition_rate * 12 * hop_length / sr)
    width = max_semitones * n_bins_per_semitone + 1
    trans = _transition_local(n_pitch_bins, width)
    t_switch = np.array([[1 - switch_prob, switch_prob], [switch_prob, 1 - switch_prob]])
    trans = np.kron(t_switch, trans)
    p_init = np.ones(2 * n_pitch_bins) / (2 * n_pitch_bins)
    eps = _tiny(obs)
    log_trans = np.log(trans + eps)
    log_prob = np.log(obs + eps)
    log_p_init = np.log(p_init + eps)
    if transition_min_prob:
        log_thr = np.log(transition_min_prob + eps)
    else:
        log_thr = -np.inf
    states = _viterbi_sparse(log_prob.T, log_trans, log_p_init, log_thr)
    freqs = fmin * 2 ** (np.arange(n_pitch_bins) / (12 * n_bins_per_semitone))
    f0 = freqs[states % n_pitch_bins]
    voiced = states < n_pitch_bins
    if fill_na is not None:
        f0 = f0.astype(np.float64)
        f0[~voiced] = fill_na
    return f0, voiced, voiced_prob[0]


# --------------------------------------------------------------------------- resample
def resample(y, *, orig_sr, target_sr, **_kw):
    """帯域制限つきの FFT リサンプル（ブラウザ版では呼ばれない想定の保険）"""
    y = np.asarray(y)
    if orig_sr == target_sr:
        return y
    n_out = int(np.ceil(y.shape[-1] * float(target_sr) / orig_sr))
    Y = np.fft.rfft(y, axis=-1)
    n_keep = n_out // 2 + 1
    Z = np.zeros(y.shape[:-1] + (n_keep,), dtype=complex)
    m = min(n_keep, Y.shape[-1])
    Z[..., :m] = Y[..., :m]
    return (np.fft.irfft(Z, n=n_out, axis=-1) * (n_out / y.shape[-1])).astype(y.dtype)
