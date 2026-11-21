// ブラウザ版エンジンの信号処理（FFT・STFT・リサンプル・メルスペクトログラム・ピーク検出）。
// 計算は tools/export_onnx.py の numpy 版と同じ（tools/make_testvec.py の値で突き合わせる）。

// ---------------------------------------------------------------------------- FFT（基数2・複素）
const FFT_CACHE = new Map();

export class FFT {
  static get(n) {
    let f = FFT_CACHE.get(n);
    if (!f) FFT_CACHE.set(n, (f = new FFT(n)));
    return f;
  }

  constructor(n) {
    this.n = n;
    const bits = Math.log2(n);
    if (!Number.isInteger(bits)) throw new Error(`FFT size ${n} is not a power of two`);
    this.rev = new Uint32Array(n);
    for (let i = 0; i < n; i++) {
      let r = 0;
      for (let b = 0; b < bits; b++) r |= ((i >> b) & 1) << (bits - 1 - b);
      this.rev[i] = r;
    }
    this.cos = new Float64Array(n / 2);
    this.sin = new Float64Array(n / 2);
    for (let k = 0; k < n / 2; k++) {
      this.cos[k] = Math.cos((2 * Math.PI * k) / n);
      this.sin[k] = Math.sin((2 * Math.PI * k) / n);
    }
  }

  // X[k] = Σ x[j] e^{-2πi jk/n}（その場で書き換え）
  forward(re, im) {
    const n = this.n;
    const rev = this.rev;
    for (let i = 0; i < n; i++) {
      const j = rev[i];
      if (j > i) {
        let t = re[i];
        re[i] = re[j];
        re[j] = t;
        t = im[i];
        im[i] = im[j];
        im[j] = t;
      }
    }
    const C = this.cos;
    const S = this.sin;
    for (let size = 2; size <= n; size <<= 1) {
      const half = size >> 1;
      const step = n / size;
      for (let i = 0; i < n; i += size) {
        for (let j = 0, k = 0; j < half; j++, k += step) {
          const a = i + j;
          const b = a + half;
          const c = C[k];
          const s = S[k];
          const tre = re[b] * c + im[b] * s;
          const tim = im[b] * c - re[b] * s;
          re[b] = re[a] - tre;
          im[b] = im[a] - tim;
          re[a] += tre;
          im[a] += tim;
        }
      }
    }
  }

  // x[j] = (1/n) Σ X[k] e^{2πi jk/n}
  inverse(re, im) {
    const n = this.n;
    for (let i = 0; i < n; i++) im[i] = -im[i];
    this.forward(re, im);
    const s = 1 / n;
    for (let i = 0; i < n; i++) {
      re[i] *= s;
      im[i] = -im[i] * s;
    }
  }
}

export function hann(n) {
  const w = new Float64Array(n);
  for (let i = 0; i < n; i++) w[i] = 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / n);
  return w;
}

// numpy の np.pad(mode="reflect")（端の値を含めずに折り返す）
export function reflectPad(x, left, right) {
  const n = x.length;
  const out = new Float64Array(n + left + right);
  for (let i = 0; i < left; i++) out[i] = x[left - i];
  for (let i = 0; i < n; i++) out[left + i] = x[i];
  for (let i = 0; i < right; i++) out[left + n + i] = x[n - 2 - i];
  return out;
}

// ---------------------------------------------------------------------------- Demucs の STFT / iSTFT
const D_NFFT = 4096;
const D_HOP = 1024;

/**
 * HTDemucs._spec と同じ（torch.stft normalized、center、reflect、最後の周波数を落とし、前後 2 フレームを除く）。
 * 戻り値は ONNX の入力 "spec"（1, 4, 2048, le）: [左 実部, 左 虚部, 右 実部, 右 虚部]
 */
export function demucsSpec(left, right, len) {
  const le = Math.ceil(len / D_HOP);
  const pad = (D_HOP / 2) * 3;
  const F = D_NFFT / 2;
  const out = new Float32Array(4 * F * le);
  const win = hann(D_NFFT);
  const fft = FFT.get(D_NFFT);
  const re = new Float64Array(D_NFFT);
  const im = new Float64Array(D_NFFT);
  const norm = 1 / Math.sqrt(D_NFFT);
  const xs = [left, right].map((x) => {
    const xp = reflectPad(x.subarray(0, len), pad, pad + le * D_HOP - len);
    return reflectPad(xp, D_NFFT / 2, D_NFFT / 2);
  });
  for (let t = 0; t < le; t++) {
    const start = (t + 2) * D_HOP;
    // 2チャンネルを 1 回の複素 FFT で（左を実部、右を虚部に入れる）
    for (let i = 0; i < D_NFFT; i++) {
      re[i] = xs[0][start + i] * win[i];
      im[i] = xs[1][start + i] * win[i];
    }
    fft.forward(re, im);
    for (let k = 0; k < F; k++) {
      const k2 = (D_NFFT - k) % D_NFFT;
      const lr = 0.5 * (re[k] + re[k2]);
      const li = 0.5 * (im[k] - im[k2]);
      const rr = 0.5 * (im[k] + im[k2]);
      const ri = -0.5 * (re[k] - re[k2]);
      out[(0 * F + k) * le + t] = lr * norm;
      out[(1 * F + k) * le + t] = li * norm;
      out[(2 * F + k) * le + t] = rr * norm;
      out[(3 * F + k) * le + t] = ri * norm;
    }
  }
  return out;
}

let _dWsum = null;
function demucsWsum(frames) {
  if (_dWsum && _dWsum.frames === frames) return _dWsum.w;
  const win = hann(D_NFFT);
  const total = D_NFFT + D_HOP * (frames - 1);
  const w = new Float64Array(total);
  for (let t = 0; t < frames; t++) for (let i = 0; i < D_NFFT; i++) w[t * D_HOP + i] += win[i] * win[i];
  _dWsum = { frames, w };
  return w;
}

/**
 * HTDemucs._ispec と同じ。specOut（1, S, 4, 2048, le）の stem s の左右を波形に戻し、
 * 時間側の出力 waveOut（1, S, 2, len）を足して返す（[左, 右]）。
 */
export function demucsStem(specOut, waveOut, s, le, len) {
  const F = D_NFFT / 2;
  const frames = le + 4;
  const total = D_NFFT + D_HOP * (frames - 1);
  const win = hann(D_NFFT);
  const wsum = demucsWsum(frames);
  const fft = FFT.get(D_NFFT);
  const re = new Float64Array(D_NFFT);
  const im = new Float64Array(D_NFFT);
  const acc = [new Float64Array(total), new Float64Array(total)];
  const scale = Math.sqrt(D_NFFT);
  const base = s * 4 * F * le;
  for (let t = 0; t < le; t++) {
    // 2チャンネルの実数 iFFT を 1 回の複素 iFFT で（Z = X左 + i X右）
    for (let k = 0; k <= F; k++) {
      let lr = 0, li = 0, rr = 0, ri = 0;
      if (k < F) {
        lr = specOut[base + (0 * F + k) * le + t] * scale;
        li = specOut[base + (1 * F + k) * le + t] * scale;
        rr = specOut[base + (2 * F + k) * le + t] * scale;
        ri = specOut[base + (3 * F + k) * le + t] * scale;
      }
      if (k === 0) {
        li = 0;
        ri = 0;
      }
      // Z[k] = L[k] + i R[k]
      re[k] = lr - ri;
      im[k] = li + rr;
      if (k > 0 && k < F) {
        // Z[n-k] = conj(L[k]) + i conj(R[k])
        re[D_NFFT - k] = lr + ri;
        im[D_NFFT - k] = -li + rr;
      }
    }
    fft.inverse(re, im);
    const off = (t + 2) * D_HOP;
    const aL = acc[0];
    const aR = acc[1];
    for (let i = 0; i < D_NFFT; i++) {
      aL[off + i] += re[i] * win[i];
      aR[off + i] += im[i] * win[i];
    }
  }
  const start = D_NFFT / 2 + (D_HOP / 2) * 3;
  const outs = [new Float32Array(len), new Float32Array(len)];
  const wb = (s * 2) * len;
  for (let c = 0; c < 2; c++) {
    const a = acc[c];
    const o = outs[c];
    const wv = waveOut;
    const woff = wb + c * len;
    for (let i = 0; i < len; i++) {
      const w = wsum[start + i];
      o[i] = a[start + i] / (w > 1e-11 ? w : 1) + wv[woff + i];
    }
  }
  return outs;
}

/**
 * demucsStem のモノラル版（(左+右)/2 だけ欲しいとき）。2 つの Stem を 1 回の複素 iFFT でまとめて戻す。
 * 戻り値は [stem s1 のモノラル, stem s2 のモノラル]（s2 が null なら 2 つめは null）
 */
export function demucsMonoPair(specOut, waveOut, s1, s2, le, len) {
  const F = D_NFFT / 2;
  const frames = le + 4;
  const total = D_NFFT + D_HOP * (frames - 1);
  const win = hann(D_NFFT);
  const wsum = demucsWsum(frames);
  const fft = FFT.get(D_NFFT);
  const re = new Float64Array(D_NFFT);
  const im = new Float64Array(D_NFFT);
  const a1 = new Float64Array(total);
  const a2 = s2 == null ? null : new Float64Array(total);
  const scale = Math.sqrt(D_NFFT) * 0.5;
  const b1 = s1 * 4 * F * le;
  const b2 = s2 == null ? 0 : s2 * 4 * F * le;
  for (let t = 0; t < le; t++) {
    for (let k = 0; k <= F; k++) {
      // M = (L + R) / 2（複素）
      let mr = 0, mi = 0, nr = 0, ni = 0;
      if (k < F) {
        mr = (specOut[b1 + (0 * F + k) * le + t] + specOut[b1 + (2 * F + k) * le + t]) * scale;
        mi = (specOut[b1 + (1 * F + k) * le + t] + specOut[b1 + (3 * F + k) * le + t]) * scale;
        if (a2) {
          nr = (specOut[b2 + (0 * F + k) * le + t] + specOut[b2 + (2 * F + k) * le + t]) * scale;
          ni = (specOut[b2 + (1 * F + k) * le + t] + specOut[b2 + (3 * F + k) * le + t]) * scale;
        }
      }
      if (k === 0) {
        mi = 0;
        ni = 0;
      }
      re[k] = mr - ni;
      im[k] = mi + nr;
      if (k > 0 && k < F) {
        re[D_NFFT - k] = mr + ni;
        im[D_NFFT - k] = -mi + nr;
      }
    }
    fft.inverse(re, im);
    const off = (t + 2) * D_HOP;
    for (let i = 0; i < D_NFFT; i++) {
      a1[off + i] += re[i] * win[i];
      if (a2) a2[off + i] += im[i] * win[i];
    }
  }
  const start = D_NFFT / 2 + (D_HOP / 2) * 3;
  const out = (acc, s) => {
    if (!acc) return null;
    const o = new Float32Array(len);
    const w0 = s * 2 * len;
    for (let i = 0; i < len; i++) {
      const w = wsum[start + i];
      o[i] = acc[start + i] / (w > 1e-11 ? w : 1) + 0.5 * (waveOut[w0 + i] + waveOut[w0 + len + i]);
    }
    return o;
  };
  return [out(a1, s1), out(a2, s2)];
}

// ---------------------------------------------------------------------------- リサンプル
function gcd(a, b) {
  while (b) [a, b] = [b, a % b];
  return a;
}

function bessel0(x) {
  let s = 1, t = 1;
  for (let k = 1; k < 50; k++) {
    t *= (x / (2 * k)) * (x / (2 * k));
    s += t;
    if (t < 1e-12 * s) break;
  }
  return s;
}

function kaiser(x, beta) {
  // x ∈ [-1, 1]
  if (Math.abs(x) > 1) return 0;
  return bessel0(beta * Math.sqrt(1 - x * x)) / bessel0(beta);
}

function sinc(x) {
  if (Math.abs(x) < 1e-12) return 1;
  return Math.sin(Math.PI * x) / (Math.PI * x);
}

const RS_CACHE = new Map();

/** 帯域制限つきの分数比リサンプル（カイザー窓 sinc、多相フィルタ） */
export function resample(x, srIn, srOut) {
  if (srIn === srOut) return Float32Array.from(x);
  const g = gcd(srIn, srOut);
  const up = srOut / g;
  const down = srIn / g;
  const ratio = up / down;
  const fc = 0.5 * Math.min(1, ratio) * 0.94; // 入力サンプル単位の遮断周波数
  const halfTaps = Math.ceil(24 / Math.min(1, ratio));
  const key = `${up}/${down}`;
  let table = RS_CACHE.get(key);
  if (!table) {
    // 位相 p（=n·down mod up）ごとのタップ。入力位置 base - halfTaps + 1 + j
    table = new Float32Array(up * 2 * halfTaps);
    for (let p = 0; p < up; p++) {
      const frac = p / up;
      let sum = 0;
      for (let j = 0; j < 2 * halfTaps; j++) {
        const d = frac - (j - halfTaps + 1); // 出力位置 - 入力位置
        const v = 2 * fc * sinc(2 * fc * d) * kaiser(d / halfTaps, 8.6);
        table[p * 2 * halfTaps + j] = v;
        sum += v;
      }
      for (let j = 0; j < 2 * halfTaps; j++) table[p * 2 * halfTaps + j] /= sum;
    }
    RS_CACHE.set(key, table);
  }
  const nOut = Math.ceil((x.length * up) / down);
  const out = new Float32Array(nOut);
  const L = 2 * halfTaps;
  const n = x.length;
  for (let o = 0; o < nOut; o++) {
    const num = o * down;
    const base = Math.floor(num / up);
    const p = num - base * up;
    const tb = p * L;
    const s0 = base - halfTaps + 1;
    let acc = 0;
    if (s0 >= 0 && s0 + L <= n) {
      for (let j = 0; j < L; j++) acc += table[tb + j] * x[s0 + j];
    } else {
      for (let j = 0; j < L; j++) {
        const i = s0 + j;
        if (i >= 0 && i < n) acc += table[tb + j] * x[i];
      }
    }
    out[o] = acc;
  }
  return out;
}

// 2 分の 1 への間引き（44.1kHz → 22.05kHz）用の FIR
let _half = null;
function halfbandTaps() {
  if (_half) return _half;
  const half = 32;
  const taps = new Float64Array(2 * half + 1);
  let sum = 0;
  for (let i = -half; i <= half; i++) {
    const v = 2 * 0.235 * sinc(2 * 0.235 * i) * kaiser(i / half, 8.6);
    taps[i + half] = v;
    sum += v;
  }
  for (let i = 0; i < taps.length; i++) taps[i] /= sum;
  _half = { half, taps };
  return _half;
}

/**
 * src（入力の位置 offset から始まる 44.1kHz の断片、範囲外は 0）を 22.05kHz に間引いて dst に足す。
 * 出力 n は入力位置 2n に対応する。scale を掛けてから足す。
 */
export function addDecimated(dst, src, offset, scale = 1) {
  const { half, taps } = halfbandTaps();
  const len = src.length;
  const nLo = Math.max(0, Math.ceil((offset - half) / 2));
  const nHi = Math.min(dst.length - 1, Math.floor((offset + len - 1 + half) / 2));
  // 全部のタップが src の内側に入る範囲（左右対称なので掛け算を半分にできる）
  const iLo = Math.max(nLo, Math.ceil((offset + half) / 2));
  const iHi = Math.min(nHi, Math.floor((offset + len - 1 - half) / 2));
  const t0 = taps[half];
  for (let n = nLo; n <= nHi; n++) {
    const c = 2 * n - offset; // src 上の中心
    let acc = 0;
    if (n >= iLo && n <= iHi) {
      acc = t0 * src[c];
      for (let j = 1; j <= half; j++) acc += taps[half + j] * (src[c + j] + src[c - j]);
    } else {
      const j0 = Math.max(-half, -c);
      const j1 = Math.min(half, len - 1 - c);
      for (let j = j0; j <= j1; j++) acc += taps[j + half] * src[c + j];
    }
    dst[n] += acc * scale;
  }
}

// ---------------------------------------------------------------------------- Beat This! の入力
let _btFb = null;
function btMelFb() {
  if (_btFb) return _btFb;
  const sr = 22050, nFft = 1024, fMin = 30, fMax = 11000, nMels = 128;
  const fSp = 200 / 3, minLogHz = 1000, minLogMel = minLogHz / fSp, logstep = Math.log(6.4) / 27;
  const hzToMel = (f) => (f >= minLogHz ? minLogMel + Math.log(f / minLogHz) / logstep : f / fSp);
  const melToHz = (m) => (m >= minLogMel ? minLogHz * Math.exp(logstep * (m - minLogMel)) : fSp * m);
  const nFreqs = nFft / 2 + 1;
  const allFreqs = new Float64Array(nFreqs);
  for (let i = 0; i < nFreqs; i++) allFreqs[i] = ((Math.floor(sr / 2)) * i) / (nFreqs - 1);
  const mMin = hzToMel(fMin), mMax = hzToMel(fMax);
  const fPts = new Float64Array(nMels + 2);
  for (let i = 0; i < nMels + 2; i++) fPts[i] = melToHz(mMin + ((mMax - mMin) * i) / (nMels + 1));
  const fb = new Float64Array(nFreqs * nMels);
  for (let f = 0; f < nFreqs; f++) {
    for (let m = 0; m < nMels; m++) {
      const down = (allFreqs[f] - fPts[m]) / (fPts[m + 1] - fPts[m]);
      const up = (fPts[m + 2] - allFreqs[f]) / (fPts[m + 2] - fPts[m + 1]);
      fb[f * nMels + m] = Math.max(0, Math.min(down, up));
    }
  }
  _btFb = fb;
  return fb;
}

/** (L,) 22050Hz → (T, 128) の log1p(1000·メル振幅) */
export function btLogmel(y) {
  const nFft = 1024, hop = 441, nMels = 128, F = nFft / 2 + 1;
  const xp = reflectPad(y, nFft / 2, nFft / 2);
  const T = 1 + Math.floor((xp.length - nFft) / hop);
  const out = new Float32Array(T * nMels);
  const win = hann(nFft);
  const fft = FFT.get(nFft);
  const re = new Float64Array(nFft);
  const im = new Float64Array(nFft);
  const mag = new Float64Array(F);
  const fb = btMelFb();
  const norm = 1 / Math.sqrt(nFft);
  for (let t = 0; t < T; t++) {
    const s = t * hop;
    for (let i = 0; i < nFft; i++) {
      re[i] = xp[s + i] * win[i];
      im[i] = 0;
    }
    fft.forward(re, im);
    for (let k = 0; k < F; k++) mag[k] = Math.hypot(re[k], im[k]) * norm;
    for (let m = 0; m < nMels; m++) {
      let a = 0;
      for (let k = 0; k < F; k++) {
        const w = fb[k * nMels + m];
        if (w) a += w * mag[k];
      }
      out[t * nMels + m] = Math.log1p(1000 * a);
    }
  }
  return { data: out, T };
}

/** Postprocessor(type="minimal"): ±3 フレームの最大で logit > 0、隣り合う山はまとめる */
export function btPeaks(beat, down, fps = 50) {
  const pick = (x) => {
    const n = x.length;
    const fr = [];
    for (let i = 0; i < n; i++) {
      let mx = -Infinity;
      for (let j = Math.max(0, i - 3); j <= Math.min(n - 1, i + 3); j++) if (x[j] > mx) mx = x[j];
      if (x[i] === mx && x[i] > 0) fr.push(i);
    }
    const groups = [];
    for (const f of fr) {
      const g = groups[groups.length - 1];
      if (g && f - g[g.length - 1] <= 1) g.push(f);
      else groups.push([f]);
    }
    return groups.map((g) => g.reduce((a, b) => a + b, 0) / g.length / fps);
  };
  const bt = pick(beat);
  let dt = pick(down);
  if (bt.length) {
    dt = dt.map((d) => {
      let best = bt[0];
      for (const b of bt) if (Math.abs(b - d) < Math.abs(best - d)) best = b;
      return best;
    });
  }
  dt = [...new Set(dt)].sort((a, b) => a - b);
  return { beats: bt, downbeats: dt };
}

// ---------------------------------------------------------------------------- 前処理（audio.py と同じ）
/** 前後の無音を切る（20ms 窓の最大値が ピーク-50dB を超える範囲 ± 0.15 秒） */
export function trimSilence(chs, sr, thresholdDb = -50, pad = 0.15) {
  const n = chs[0].length;
  let peak = 0;
  const mono = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    let s = 0;
    for (const c of chs) s += c[i];
    mono[i] = Math.abs(s / chs.length);
    if (mono[i] > peak) peak = mono[i];
  }
  if (!n || !peak) return { chs, offset: 0 };
  const thr = peak * Math.pow(10, thresholdDb / 20);
  const win = Math.max(1, Math.floor(sr * 0.02));
  const blocks = Math.floor(n / win);
  let first = -1, last = -1;
  for (let b = 0; b < blocks; b++) {
    let mx = 0;
    for (let i = b * win; i < (b + 1) * win; i++) if (mono[i] > mx) mx = mono[i];
    if (mx > thr) {
      if (first < 0) first = b;
      last = b;
    }
  }
  if (first < 0) return { chs, offset: 0 };
  const a = Math.max(0, first * win - Math.floor(pad * sr));
  const b = Math.min(n, (last + 1) * win + Math.floor(pad * sr));
  return { chs: chs.map((c) => c.slice(a, b)), offset: a / sr };
}

export function normalize(chs, peakDb = -1) {
  let peak = 0;
  for (const c of chs) for (let i = 0; i < c.length; i++) peak = Math.max(peak, Math.abs(c[i]));
  if (peak < 1e-6) return chs;
  const g = Math.pow(10, peakDb / 20) / peak;
  return chs.map((c) => {
    const o = new Float32Array(c.length);
    for (let i = 0; i < c.length; i++) o[i] = c[i] * g;
    return o;
  });
}

// ---------------------------------------------------------------------------- WAV（16bit モノラル）
export function wavBlob(x, sr) {
  const n = x.length;
  const buf = new ArrayBuffer(44 + n * 2);
  const v = new DataView(buf);
  const w = (o, s) => [...s].forEach((ch, i) => v.setUint8(o + i, ch.charCodeAt(0)));
  w(0, "RIFF");
  v.setUint32(4, 36 + n * 2, true);
  w(8, "WAVE");
  w(12, "fmt ");
  v.setUint32(16, 16, true);
  v.setUint16(20, 1, true);
  v.setUint16(22, 1, true);
  v.setUint32(24, sr, true);
  v.setUint32(28, sr * 2, true);
  v.setUint16(32, 2, true);
  v.setUint16(34, 16, true);
  w(36, "data");
  v.setUint32(40, n * 2, true);
  let peak = 0;
  for (let i = 0; i < n; i++) peak = Math.max(peak, Math.abs(x[i]));
  const g = peak > 0.99 ? 0.99 / peak : 1;
  for (let i = 0; i < n; i++) v.setInt16(44 + i * 2, Math.max(-32768, Math.min(32767, Math.round(x[i] * g * 32767))), true);
  return new Blob([buf], { type: "audio/wav" });
}

export async function wavRead(blob) {
  const v = new DataView(await blob.arrayBuffer());
  const sr = v.getUint32(24, true);
  const n = v.getUint32(40, true) / 2;
  const x = new Float32Array(n);
  for (let i = 0; i < n; i++) x[i] = v.getInt16(44 + i * 2, true) / 32767;
  return { x, sr };
}
