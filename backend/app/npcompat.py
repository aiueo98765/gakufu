"""scipy が無い環境（ブラウザ版エンジン）向けの、numpy だけで書いた代用品。

PC 版では scipy を使い、import できないときだけこちらに落ちる。
"""
from __future__ import annotations

import math

import numpy as np


def smooth_second_diff(y: np.ndarray, w: np.ndarray, lam: float) -> np.ndarray:
    """min Σ w_i (x_i - y_i)^2 + lam Σ (x_{i+1} - 2x_i + x_{i-1})^2 の解。

    (W + lam DᵀD) x = W y は対称正定値の五重対角なので、帯コレスキー分解で O(n) で解く。
    """
    n = y.size
    # 帯（主対角・1つ上・2つ上）を作る
    d0 = np.asarray(w, dtype=np.float64).copy()
    d1 = np.zeros(max(n - 1, 0))
    d2 = np.zeros(max(n - 2, 0))
    c = (1.0, -2.0, 1.0)
    rows = n - 2
    for a in range(3):
        for b in range(3):
            v = lam * c[a] * c[b]
            off = b - a
            if off == 0:
                d0[a : a + rows] += v
            elif off == 1:
                d1[a : a + rows] += v
            elif off == 2:
                d2[a : a + rows] += v
    # 帯コレスキー L Lᵀ（L の対角 l0、1つ下 l1、2つ下 l2）
    l0 = np.zeros(n)
    l1 = np.zeros(max(n - 1, 0))
    l2 = np.zeros(max(n - 2, 0))
    for j in range(n):
        s = d0[j]
        if j >= 1:
            s -= l1[j - 1] ** 2
        if j >= 2:
            s -= l2[j - 2] ** 2
        l0[j] = math.sqrt(s)
        if j + 1 < n:
            t = d1[j]
            if j >= 1:
                t -= l2[j - 1] * l1[j - 1]
            l1[j] = t / l0[j]
        if j + 2 < n:
            l2[j] = d2[j] / l0[j]
    rhs = np.asarray(w, dtype=np.float64) * np.asarray(y, dtype=np.float64)
    z = np.zeros(n)
    for i in range(n):
        s = rhs[i]
        if i >= 1:
            s -= l1[i - 1] * z[i - 1]
        if i >= 2:
            s -= l2[i - 2] * z[i - 2]
        z[i] = s / l0[i]
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        s = z[i]
        if i + 1 < n:
            s -= l1[i] * x[i + 1]
        if i + 2 < n:
            s -= l2[i] * x[i + 2]
        x[i] = s / l0[i]
    return x


def median_filter1d(x: np.ndarray, size: int) -> np.ndarray:
    """scipy.ndimage.median_filter(x, size)（1次元・mode='reflect'）"""
    x = np.asarray(x)
    if x.size == 0:
        return x.copy()
    h = size // 2
    left = h
    right = size - 1 - h
    # scipy の reflect は端の値も含めて折り返す（d c b a | a b c d | d c b a）
    padded = np.concatenate([x[:left][::-1], x, x[x.size - right :][::-1]]) if x.size >= max(left, right) else np.pad(x, (left, right), mode="symmetric")
    idx = np.arange(size)[None, :] + np.arange(x.size)[:, None]
    return np.median(padded[idx], axis=1).astype(x.dtype)
