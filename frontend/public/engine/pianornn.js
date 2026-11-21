// ByteDance ピアノ採譜モデルの後半（双方向 GRU と全結合）を WebGPU で計算する。
//
// onnxruntime-web の WebGPU は GRU を 1 時刻ごとに GPU を 2 回呼ぶ形で動かすため、
// 1 区間（1001 フレーム）で約 13 万回の呼び出しになり、1 区間に 14 秒かかる。
// ここでは 7 つの音響モデル × 2 方向（14 本）の GRU を 1 時刻 1 回の呼び出しでまとめて進め、
// 入力側の行列積はまとめて 1 回で計算する。計算は fp32。
//
// 入力: 7 つの音響モデルの GRU 入力（各 T×768、piano_frontend.onnx の出力）
// 出力: PyTorch の Note_pedal と同じ 7 つの出力

export const AMS = [
  "frame_model",
  "reg_onset_model",
  "reg_offset_model",
  "velocity_model",
  "reg_pedal_onset_model",
  "reg_pedal_offset_model",
  "reg_pedal_frame_model",
];
const H = 256;
const G = 3 * H; // r, z, n

// ---------------------------------------------------------------------------- 重みの読み込み
function halfToFloat(h) {
  const s = (h & 0x8000) >> 15;
  const e = (h & 0x7c00) >> 10;
  const f = h & 0x03ff;
  if (e === 0) return (s ? -1 : 1) * Math.pow(2, -14) * (f / 1024);
  if (e === 0x1f) return f ? NaN : (s ? -1 : 1) * Infinity;
  return (s ? -1 : 1) * Math.pow(2, e - 15) * (1 + f / 1024);
}

export function loadWeights(buffer, manifest) {
  const u16 = new Uint16Array(buffer);
  const table = new Float32Array(65536);
  for (let i = 0; i < 65536; i++) table[i] = halfToFloat(i);
  const W = {};
  for (const [name, t] of Object.entries(manifest.tensors)) {
    const n = t.shape.reduce((a, b) => a * b, 1);
    const off = t.offset / 2;
    const a = new Float32Array(n);
    for (let i = 0; i < n; i++) a[i] = table[u16[off + i]];
    W[name] = { data: a, shape: t.shape };
  }
  return W;
}

// ---------------------------------------------------------------------------- シェーダ
// 行列積: C[b][r][o] = act( Σ_i A[ba][r][i] * B[b][o][i] + bias[b][o] )
//   A のバッチ番号は b / aDiv（同じ入力を 2 方向で使い回すため）
const MATMUL = /* wgsl */ `
struct P { rows: u32, K: u32, N: u32, aDiv: u32, act: u32, aStride: u32, cStride: u32, pad: u32 };
@group(0) @binding(0) var<storage, read> A: array<f32>;
@group(0) @binding(1) var<storage, read> B: array<f32>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> C: array<f32>;
@group(0) @binding(4) var<uniform> p: P;
var<workgroup> ta: array<array<f32, 17>, 16>;
var<workgroup> tb: array<array<f32, 17>, 16>;
@compute @workgroup_size(16, 16)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) li: vec3<u32>) {
  let b = wg.z;
  let r = wg.y * 16u + li.y;
  let o = wg.x * 16u + li.x;
  let aBase = (b / p.aDiv) * p.aStride;
  let bBase = b * p.N * p.K;
  var acc = 0.0;
  var k0 = 0u;
  loop {
    if (k0 >= p.K) { break; }
    let ka = k0 + li.x;
    let rr = wg.y * 16u + li.y;
    if (rr < p.rows && ka < p.K) { ta[li.y][li.x] = A[aBase + rr * p.K + ka]; } else { ta[li.y][li.x] = 0.0; }
    let oo = wg.x * 16u + li.y;
    if (oo < p.N && ka < p.K) { tb[li.y][li.x] = B[bBase + oo * p.K + ka]; } else { tb[li.y][li.x] = 0.0; }
    workgroupBarrier();
    for (var k = 0u; k < 16u; k++) { acc += ta[li.y][k] * tb[li.x][k]; }
    workgroupBarrier();
    k0 += 16u;
  }
  if (r < p.rows && o < p.N) {
    var v = acc + bias[b * p.N + o];
    if (p.act == 1u) { v = 1.0 / (1.0 + exp(-v)); }
    C[b * p.cStride + r * p.N + o] = v;
  }
}`;

// GRU の 1 時刻: 系列 s = (モデル m, 方向 d) ごとに 1 ワークグループ（256 スレッド = 隠れ状態の各要素）
//   Xp[s][t][0..768) は入力側の線形変換済み（r, z, n の順）
//   WhhT[s][i][g] は再帰の重みを転置したもの（読み出しをまとめるため）
//   Hout[m][t][d*256 + j] に出力。前の時刻の状態も Hout から読む。
const GRU_STEP = /* wgsl */ `
struct S { step: u32, T: u32, pad0: u32, pad1: u32 };
@group(0) @binding(0) var<storage, read> Xp: array<f32>;
@group(0) @binding(1) var<storage, read> WhhT: array<f32>;
@group(0) @binding(2) var<storage, read> bhh: array<f32>;
@group(0) @binding(3) var<storage, read_write> Hout: array<f32>;
@group(0) @binding(4) var<uniform> st: S;
var<workgroup> h: array<f32, 256>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) li: vec3<u32>) {
  let s = wg.x;
  let m = s / 2u;
  let d = s % 2u;
  let j = li.x;
  let T = st.T;
  var t: u32;
  var tp: i32;
  if (d == 0u) { t = st.step; tp = i32(st.step) - 1; } else { t = T - 1u - st.step; tp = i32(t) + 1; }
  if (st.step == 0u) { h[j] = 0.0; } else { h[j] = Hout[(m * T + u32(tp)) * 512u + d * 256u + j]; }
  workgroupBarrier();
  let wb = s * 256u * 768u;
  var gr = bhh[s * 768u + j];
  var gz = bhh[s * 768u + 256u + j];
  var gn = bhh[s * 768u + 512u + j];
  for (var i = 0u; i < 256u; i++) {
    let hv = h[i];
    let row = wb + i * 768u;
    gr += hv * WhhT[row + j];
    gz += hv * WhhT[row + 256u + j];
    gn += hv * WhhT[row + 512u + j];
  }
  let xb = (s * T + t) * 768u;
  let r = 1.0 / (1.0 + exp(-(Xp[xb + j] + gr)));
  let z = 1.0 / (1.0 + exp(-(Xp[xb + 256u + j] + gz)));
  let n = tanh(Xp[xb + 512u + j] + r * gn);
  Hout[(m * T + t) * 512u + d * 256u + j] = (1.0 - z) * n + z * h[j];
}`;

// ---------------------------------------------------------------------------- WebGPU 版
export class PianoRNNGpu {
  static async create(device, W) {
    const self = new PianoRNNGpu();
    self.device = device;
    self.W = W;
    const mm = device.createShaderModule({ code: MATMUL });
    const gs = device.createShaderModule({ code: GRU_STEP });
    self.mmPipe = await device.createComputePipelineAsync({ layout: "auto", compute: { module: mm, entryPoint: "main" } });
    self.gruPipe = await device.createComputePipelineAsync({ layout: "auto", compute: { module: gs, entryPoint: "main" } });
    // 重みを層ごとにまとめて GPU に置く
    self.layers = {};
    const amLayer = (l, inDim) => self.packGru(AMS.map((n) => `${n}.l${l}.`), inDim);
    self.layers.am0 = amLayer(0, 768);
    self.layers.am1 = amLayer(1, 512);
    self.layers.onset = self.packGru(["reg_onset_gru.l0."], 176);
    self.layers.frame = self.packGru(["frame_gru.l0."], 264);
    self.fc = {};
    for (const n of AMS) self.fc[n] = self.packFc(`${n}.fc.`);
    self.fc.reg_onset = self.packFc("reg_onset_fc.");
    self.fc.frame = self.packFc("frame_fc.");
    return self;
  }

  buf(data, usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST) {
    const b = this.device.createBuffer({ size: Math.max(16, data.byteLength), usage });
    this.device.queue.writeBuffer(b, 0, data);
    return b;
  }

  empty(bytes, usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST) {
    return this.device.createBuffer({ size: Math.max(16, bytes), usage });
  }

  packGru(prefixes, inDim) {
    const S = prefixes.length * 2;
    const wih = new Float32Array(S * G * inDim);
    const whhT = new Float32Array(S * H * G);
    const bih = new Float32Array(S * G);
    const bhh = new Float32Array(S * G);
    prefixes.forEach((p, m) => {
      ["", "_r"].forEach((tag, d) => {
        const s = m * 2 + d;
        wih.set(this.W[`${p}weight_ih${tag}`].data, s * G * inDim);
        const hh = this.W[`${p}weight_hh${tag}`].data; // (768, 256)
        for (let g = 0; g < G; g++) for (let i = 0; i < H; i++) whhT[s * H * G + i * G + g] = hh[g * H + i];
        bih.set(this.W[`${p}bias_ih${tag}`].data, s * G);
        bhh.set(this.W[`${p}bias_hh${tag}`].data, s * G);
      });
    });
    return { S, inDim, wih: this.buf(wih), whhT: this.buf(whhT), bih: this.buf(bih), bhh: this.buf(bhh) };
  }

  packFc(prefix) {
    const w = this.W[`${prefix}weight`];
    return { out: w.shape[0], w: this.buf(w.data), b: this.buf(this.W[`${prefix}bias`].data) };
  }

  uniform(u32s) {
    const a = new Uint32Array(u32s);
    return this.buf(a, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
  }

  // C = act(A Bᵀ + bias)。batch 個の行列積を 1 回で。
  matmul(enc, { A, B, bias, C, rows, K, N, batch, aDiv = 1, aStride, cStride, act = 0 }) {
    const u = this.uniform([rows, K, N, aDiv, act, aStride ?? rows * K, cStride ?? rows * N, 0]);
    const bg = this.device.createBindGroup({
      layout: this.mmPipe.getBindGroupLayout(0),
      entries: [A, B, bias, C, u].map((b, i) => ({ binding: i, resource: { buffer: b } })),
    });
    const pass = enc.beginComputePass();
    pass.setPipeline(this.mmPipe);
    pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(Math.ceil(N / 16), Math.ceil(rows / 16), batch);
    pass.end();
    this.tmp.push(u);
  }

  // 双方向 GRU 1 層。X: (M, T, inDim) → Hout: (M, T, 512)
  gru(enc, layer, X, T) {
    const M = layer.S / 2;
    const Xp = this.empty(layer.S * T * G * 4);
    this.matmul(enc, { A: X, B: layer.wih, bias: layer.bih, C: Xp, rows: T, K: layer.inDim, N: G, batch: layer.S, aDiv: 2 });
    const Hout = this.empty(M * T * 512 * 4);
    // 時刻ごとの番号は、動的オフセット付きの uniform で渡す
    const stride = 256;
    const steps = new Uint32Array((T * stride) / 4);
    for (let k = 0; k < T; k++) {
      steps[(k * stride) / 4] = k;
      steps[(k * stride) / 4 + 1] = T;
    }
    const ub = this.buf(steps, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
    const layout = this.gruPipe.getBindGroupLayout(0);
    const bg = this.device.createBindGroup({
      layout,
      entries: [
        { binding: 0, resource: { buffer: Xp } },
        { binding: 1, resource: { buffer: layer.whhT } },
        { binding: 2, resource: { buffer: layer.bhh } },
        { binding: 3, resource: { buffer: Hout } },
        { binding: 4, resource: { buffer: ub, size: 16 } },
      ],
    });
    // 動的オフセットは "auto" レイアウトでは使えないので、時刻ごとに bind group を作る代わりに
    // uniform を区切った bind group をあらかじめ作っておく
    const groups = [];
    for (let k = 0; k < T; k++) {
      groups.push(
        k === 0
          ? bg
          : this.device.createBindGroup({
              layout,
              entries: [
                { binding: 0, resource: { buffer: Xp } },
                { binding: 1, resource: { buffer: layer.whhT } },
                { binding: 2, resource: { buffer: layer.bhh } },
                { binding: 3, resource: { buffer: Hout } },
                { binding: 4, resource: { buffer: ub, offset: k * stride, size: 16 } },
              ],
            }),
      );
    }
    const pass = enc.beginComputePass();
    pass.setPipeline(this.gruPipe);
    for (let k = 0; k < T; k++) {
      pass.setBindGroup(0, groups[k]);
      pass.dispatchWorkgroups(layer.S);
    }
    pass.end();
    this.tmp.push(Xp, ub);
    return Hout;
  }

  async read(buffer, bytes) {
    const rb = this.device.createBuffer({ size: bytes, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const enc = this.device.createCommandEncoder();
    enc.copyBufferToBuffer(buffer, 0, rb, 0, bytes);
    this.device.queue.submit([enc.finish()]);
    await rb.mapAsync(GPUMapMode.READ);
    const out = new Float32Array(rb.getMappedRange().slice(0));
    rb.unmap();
    rb.destroy();
    return out;
  }

  // feats: 7 個の Float32Array（各 T×768）
  async run(feats, T) {
    this.tmp = [];
    const dev = this.device;
    const X1 = this.empty(7 * T * 768 * 4);
    feats.forEach((f, m) => dev.queue.writeBuffer(X1, m * T * 768 * 4, f));
    const enc = dev.createCommandEncoder();
    const H1 = this.gru(enc, this.layers.am0, X1, T);
    const H2 = this.gru(enc, this.layers.am1, H1, T);
    // 各音響モデルの全結合 + sigmoid（モデル m の行だけを切り出して計算する）
    const outs = {};
    for (let m = 0; m < 7; m++) {
      const fc = this.fc[AMS[m]];
      const out = this.empty(T * fc.out * 4);
      const slice = this.empty(T * 512 * 4);
      enc.copyBufferToBuffer(H2, m * T * 512 * 4, slice, 0, T * 512 * 4);
      this.matmul(enc, { A: slice, B: fc.w, bias: fc.b, C: out, rows: T, K: 512, N: fc.out, batch: 1, act: 1 });
      outs[AMS[m]] = { fc, out };
      this.tmp.push(slice, out);
    }
    dev.queue.submit([enc.finish()]);
    const res = {};
    for (const n of AMS) res[n] = await this.read(outs[n].out, T * outs[n].fc.out * 4);
    // 打鍵（onset）の条件付け GRU：[onset, sqrt(onset) * velocity]
    const ro = res.reg_onset_model;
    const vel = res.velocity_model;
    const x2 = new Float32Array(T * 176);
    for (let t = 0; t < T; t++) {
      for (let k = 0; k < 88; k++) {
        const o = ro[t * 88 + k];
        x2[t * 176 + k] = o;
        x2[t * 176 + 88 + k] = Math.sqrt(o) * vel[t * 88 + k];
      }
    }
    const regOnset = await this.condGru(this.layers.onset, this.fc.reg_onset, x2, T, 176);
    const x3 = new Float32Array(T * 264);
    const fr = res.frame_model;
    const off = res.reg_offset_model;
    for (let t = 0; t < T; t++) {
      for (let k = 0; k < 88; k++) {
        x3[t * 264 + k] = fr[t * 88 + k];
        x3[t * 264 + 88 + k] = regOnset[t * 88 + k];
        x3[t * 264 + 176 + k] = off[t * 88 + k];
      }
    }
    const frame = await this.condGru(this.layers.frame, this.fc.frame, x3, T, 264);
    for (const b of [X1, H1, H2, ...this.tmp]) b.destroy();
    this.tmp = [];
    return {
      reg_onset_output: regOnset,
      reg_offset_output: off,
      frame_output: frame,
      velocity_output: vel,
      reg_pedal_onset_output: res.reg_pedal_onset_model,
      reg_pedal_offset_output: res.reg_pedal_offset_model,
      pedal_frame_output: res.reg_pedal_frame_model,
    };
  }

  async condGru(layer, fc, x, T, inDim) {
    const dev = this.device;
    const X = this.buf(x);
    const enc = dev.createCommandEncoder();
    const Hb = this.gru(enc, layer, X, T);
    const out = this.empty(T * 88 * 4);
    this.matmul(enc, { A: Hb, B: fc.w, bias: fc.b, C: out, rows: T, K: 512, N: 88, batch: 1, act: 1 });
    dev.queue.submit([enc.finish()]);
    const r = await this.read(out, T * 88 * 4);
    for (const b of [X, Hb, out]) b.destroy();
    return r;
  }
}
