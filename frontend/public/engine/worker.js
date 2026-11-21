// 楽譜AI ブラウザ版エンジン（Web Worker）。
//
// 閲覧している人の PC の中だけで、音源分離・AI 採譜・拍の推定・編曲・楽譜の書き出しまでを行う。
//   ・AI の推論  : onnxruntime-web（WebGPU。無ければ WASM）
//   ・ピアノ採譜の GRU : 自前の WebGPU シェーダ（pianornn.js）
//   ・それ以外   : PC 版と同じ Python のコード（Pyodide。py/ の zip）
// 曲のデータは Pyodide の仮想ファイルシステム（IndexedDB に保存）に PC 版と同じ形で置き、
// 音声は別の IndexedDB に置く。外部へは何も送らない。

const ORT_VERSION = "1.31.1";
const PYODIDE_VERSION = "0.29.5";
const MODEL_CACHE = "gakufu-models-v1";
const MAX_ANALYZE_SEC = 600;

const here = (p) => new URL(p, import.meta.url).href;
// worker.js?v=<版> で読み込まれる。同じ版の部品を読む（公開の更新直後に古い部品と混ざらないように）
const VQ = new URL(import.meta.url).search;

// 部品を読み込む前に届いた依頼は、読み込みが終わるまで待たせる
let dsp, AMS, PianoRNNGpu, loadWeights;
let resolveReady;
const ready = new Promise((r) => (resolveReady = r));
self.onmessage = async (e) => {
  await ready;
  onMessage(e);
};
dsp = await import(`./dsp.js${VQ}`);
({ AMS, PianoRNNGpu, loadWeights } = await import(`./pianornn.js${VQ}`));

// GPU を使わない（WebGPU の無い PC と同じ動き）。動作確認用：Worker の名前で指定する
const FORCE_CPU = self.name === "gakufu-cpu";

// ---------------------------------------------------------------------------- GPU
let gpuInfoP = null;
function gpuInfo() {
  if (!gpuInfoP) {
    gpuInfoP = (async () => {
      try {
        if (FORCE_CPU || !self.navigator?.gpu) return { webgpu: false, name: null };
        const a = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
        if (!a) return { webgpu: false, name: null };
        const info = a.info || {};
        const name = info.description || [info.vendor, info.architecture].filter(Boolean).join(" ") || "WebGPU";
        return { webgpu: true, name, adapter: a };
      } catch {
        return { webgpu: false, name: null };
      }
    })();
  }
  return gpuInfoP;
}

let rnnDeviceP = null;
async function rnnDevice() {
  if (!rnnDeviceP) {
    rnnDeviceP = (async () => {
      const g = await gpuInfo();
      if (!g.webgpu) return null;
      const a = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
      const lim = a.limits;
      return a.requestDevice({
        requiredLimits: { maxStorageBufferBindingSize: lim.maxStorageBufferBindingSize, maxBufferSize: lim.maxBufferSize },
      });
    })();
  }
  return rnnDeviceP;
}

// ---------------------------------------------------------------------------- onnxruntime-web
let ortP = null;
function ortLib() {
  if (!ortP) {
    ortP = import(`https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist/ort.webgpu.min.mjs`).then((ort) => {
      ort.env.wasm.wasmPaths = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist/`;
      ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(8, navigator.hardwareConcurrency || 4) : 1;
      return ort;
    });
  }
  return ortP;
}

// ---------------------------------------------------------------------------- モデル（初回だけ取得してブラウザに保存）
const MODELS = {
  demucs: { file: "htdemucs_6s.onnx", label: "音源分離AI (Demucs 6-stem)", mb: 60 },
  pianoFront: { file: "piano_frontend.onnx", label: "ピアノ採譜AI（前半）", mb: 44 },
  pianoRnn: { file: "piano_rnn.bin", label: "ピアノ採譜AI（後半）", mb: 42 },
  pianoRnnJson: { file: "piano_rnn.json", label: "ピアノ採譜AI（設定）", mb: 0 },
  pianoRnnOnnx: { file: "piano_rnn.onnx", label: "ピアノ採譜AI（後半・GPU なし用）", mb: 42 },
  beat: { file: "beat_this.onnx", label: "拍・小節頭AI (Beat This!)", mb: 44 },
  basic: { file: "basic_pitch.onnx", label: "楽器採譜AI (Basic Pitch)", mb: 0.23 },
};

// モデルを更新したら MODEL_CACHE の版を上げる。古い版はブラウザから消す
let cleaned = false;
async function cleanupOldModels() {
  if (cleaned) return;
  cleaned = true;
  try {
    for (const k of await caches.keys()) if (k.startsWith("gakufu-models-") && k !== MODEL_CACHE) await caches.delete(k);
  } catch {
    /* Cache API が使えない */
  }
}

async function modelBytes(key, onProgress) {
  const m = MODELS[key];
  const url = here(`./models/${m.file}`);
  await cleanupOldModels();
  let cache = null;
  try {
    cache = await caches.open(MODEL_CACHE);
    const hit = await cache.match(url);
    if (hit) return await hit.arrayBuffer();
  } catch {
    cache = null; // Cache API が使えない（プライベートモードなど）
  }
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${m.label} を取得できませんでした（${res.status}）`);
  // GitHub Pages は圧縮して送ることがあり、そのとき content-length は圧縮後の大きさなので使わない
  const total = (!res.headers.get("content-encoding") && Number(res.headers.get("content-length"))) || m.mb * 1e6;
  const reader = res.body.getReader();
  const parts = [];
  let got = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    parts.push(value);
    got += value.length;
    onProgress?.(Math.min(1, got / total), `${m.label} を取得中 ${(got / 1e6).toFixed(0)}/${(total / 1e6).toFixed(0)} MB（初回だけ）`);
  }
  const blob = new Blob(parts);
  if (cache) {
    try {
      await cache.put(url, new Response(blob, { headers: { "content-type": "application/octet-stream" } }));
    } catch {
      /* 容量不足などは無視（次回また取得する） */
    }
  }
  return await blob.arrayBuffer();
}

async function modelCached(key) {
  try {
    const cache = await caches.open(MODEL_CACHE);
    return !!(await cache.match(here(`./models/${MODELS[key].file}`)));
  } catch {
    return false;
  }
}

const sessions = {};
async function session(key, onProgress) {
  if (!sessions[key]) {
    sessions[key] = (async () => {
      const ort = await ortLib();
      const bytes = await modelBytes(key, onProgress);
      onProgress?.(1, `${MODELS[key].label} を準備中`);
      const g = await gpuInfo();
      return ort.InferenceSession.create(new Uint8Array(bytes), {
        executionProviders: g.webgpu ? ["webgpu", "wasm"] : ["wasm"],
        graphOptimizationLevel: "all",
      });
    })().catch((e) => {
      delete sessions[key];
      throw e;
    });
  }
  return sessions[key];
}

let pianoRnnP = null;
async function pianoRnn(onProgress) {
  if (!pianoRnnP) {
    pianoRnnP = (async () => {
      const device = await rnnDevice();
      if (device) {
        try {
          const man = JSON.parse(new TextDecoder().decode(await modelBytes("pianoRnnJson")));
          const W = loadWeights(await modelBytes("pianoRnn", onProgress), man);
          const gpu = await PianoRNNGpu.create(device, W);
          return { run: (feats, T) => gpu.run(feats, T), kind: "webgpu" };
        } catch (e) {
          console.warn("WebGPU の GRU を使えないので CPU で計算します", e);
        }
      }
      // GPU が無いとき：GRU は onnxruntime の WASM 版で（JS で書くより何倍も速い）
      const sess = await session("pianoRnnOnnx", onProgress);
      const ort = await ortLib();
      return {
        kind: "cpu",
        run: async (feats, T) => {
          const feeds = {};
          AMS.forEach((n, i) => (feeds[n] = new ort.Tensor("float32", feats[i], [T, 768])));
          const o = await sess.run(feeds);
          const out = {};
          for (const k of Object.keys(o)) out[k] = o[k].data;
          return out;
        },
      };
    })().catch((e) => {
      pianoRnnP = null;
      throw e;
    });
  }
  return pianoRnnP;
}

// ---------------------------------------------------------------------------- Python（Pyodide）
// 曲のデータ（JSON）は Pyodide の仮想ファイル /data/jobs/<id>/ に PC 版と同じ形で置き、
// 曲ごとに IndexedDB（ストア "fs"、キー＝ファイルのパス）へ保存する。
// （ファイルシステム丸ごとの同期だと、別のタブで作った曲を消してしまうことがあるため）
let pyP = null;
function py() {
  if (!pyP) {
    pyP = (async () => {
      const { loadPyodide } = await import(`https://cdn.jsdelivr.net/pyodide/v${PYODIDE_VERSION}/full/pyodide.mjs`);
      const pyodide = await loadPyodide({
        indexURL: `https://cdn.jsdelivr.net/pyodide/v${PYODIDE_VERSION}/full/`,
        stdout: (s) => console.log("[py]", s),
        stderr: (s) => console.warn("[py]", s),
      });
      await pyodide.loadPackage(["numpy", "packaging"]);
      const man = await (await fetch(here("./py/manifest.json"), { cache: "no-cache" })).json();
      const zip = await (await fetch(here(`./py/${man.zip}`))).arrayBuffer();
      pyodide.unpackArchive(zip, "zip", { extractDir: "/lib/gakufu" });
      const FS = pyodide.FS;
      FS.mkdirTree("/data/jobs");
      for (const [path, data] of await fsAll()) writeFsFile(FS, path, data);
      // 別のタブが処理中の曲（Web Locks で印を付けている）は、中断扱いにしない
      let live = [];
      try {
        live = ((await navigator.locks.query()).held || []).map((l) => l.name).filter((n) => n.startsWith("gakufu-job:")).map((n) => n.slice(11));
      } catch {
        live = [];
      }
      pyodide.globals.set("_live_jobs", live.filter((x) => /^[0-9a-f]{12}$/.test(x)).join(","));
      pyodide.runPython(`
import os, sys
os.environ["GAKUFU_DATA"] = "/data"
os.environ["GAKUFU_MODELS"] = "/tmp/models"
os.environ["GAKUFU_LIVE_JOBS"] = _live_jobs
sys.path.insert(0, "/lib/gakufu")
`);
      const W = pyodide.pyimport("app.webapi");
      const fixed = JSON.parse(W.boot_fixed());
      const ctx = { pyodide, W, FS };
      for (const jid of fixed) await saveJobFiles(ctx, jid);
      return ctx;
    })().catch((e) => {
      pyP = null;
      throw e;
    });
  }
  return pyP;
}

function writeFsFile(FS, path, data) {
  FS.mkdirTree(path.slice(0, path.lastIndexOf("/")));
  FS.writeFile(path, data instanceof Uint8Array ? data : new Uint8Array(data));
}

function jobFiles(FS, jid) {
  const dir = `/data/jobs/${jid}`;
  let names = [];
  try {
    names = FS.readdir(dir).filter((n) => n !== "." && n !== "..");
  } catch {
    return [];
  }
  const out = [];
  for (const n of names) {
    const p = `${dir}/${n}`;
    try {
      if (FS.isFile(FS.stat(p).mode)) out.push([p, FS.readFile(p)]);
    } catch {
      /* 途中で消えた */
    }
  }
  return out;
}

async function saveJobFiles(ctx, jid) {
  const dir = `/data/jobs/${jid}/`;
  const entries = jobFiles(ctx.FS, jid);
  await idb("readwrite", (st) => {
    st.delete(IDBKeyRange.bound(dir, dir + "￿"));
    for (const [k, v] of entries) st.put(v, k);
  }, "fs");
}

// 曲のファイルを IndexedDB に書き戻す。続けて呼ばれたら少し待ってまとめる。
const persistTimers = new Map();
let persistChain = Promise.resolve();
function persistJob(jid, now = false) {
  if (!jid) return Promise.resolve();
  clearTimeout(persistTimers.get(jid));
  const run = () => {
    persistTimers.delete(jid);
    persistChain = persistChain.then(async () => saveJobFiles(await py(), jid)).catch((e) => console.warn("保存に失敗", e));
    return persistChain;
  };
  if (now) return run();
  persistTimers.set(jid, setTimeout(run, 500));
  return Promise.resolve();
}

// 別のタブが書き換えた曲を読み直す（このタブで処理中の曲は除く）
async function reloadJob(jid) {
  if (current?.jid === jid) return;
  if (persistTimers.has(jid)) {
    // このタブの書き込みが保存待ち＝このタブが書き手。読み直さずに保存を先に済ませる
    await persistJob(jid, true);
    return;
  }
  const ctx = await py();
  const dir = `/data/jobs/${jid}/`;
  const files = await idb("readonly", (st) => getAllRange(st, IDBKeyRange.bound(dir, dir + "￿")), "fs");
  for (const [p] of jobFiles(ctx.FS, jid)) {
    try {
      ctx.FS.unlink(p);
    } catch {
      /* 無い */
    }
  }
  for (const [p, d] of files) writeFsFile(ctx.FS, p, d);
  ctx.W.refresh(jid);
}

let lastListReload = 0;
async function reloadList() {
  if (Date.now() - lastListReload < 1500) return;
  lastListReload = Date.now();
  const ctx = await py();
  const keys = await idb("readonly", (st) => new Promise((res) => {
    const r = st.getAllKeys();
    r.onsuccess = () => res(r.result);
    r.onerror = () => res([]);
  }), "fs");
  const ids = [...new Set(keys.map((k) => /^\/data\/jobs\/([0-9a-f]{12})\/job\.json$/.exec(k)?.[1]).filter(Boolean))];
  const busy = current?.jid || "";
  const jsons = await idb("readonly", (st) => Promise.all(ids.filter((j) => j !== busy).map((j) => new Promise((res) => {
    const r = st.get(`/data/jobs/${j}/job.json`);
    r.onsuccess = () => res([j, r.result]);
    r.onerror = () => res([j, null]);
  }))), "fs");
  for (const [j, data] of jsons) if (data) writeFsFile(ctx.FS, `/data/jobs/${j}/job.json`, data);
  ctx.W.refresh_all(JSON.stringify(ids), busy);
}

function getAllRange(st, range) {
  return new Promise((res) => {
    const out = [];
    const r = st.openCursor(range);
    r.onsuccess = () => {
      const c = r.result;
      if (!c) return res(out);
      out.push([c.key, c.value]);
      c.continue();
    };
    r.onerror = () => res(out);
  });
}

async function fsAll() {
  return idb("readonly", (st) => getAllRange(st, null), "fs");
}

async function pycall(name, ...args) {
  const { W } = await py();
  const fn = W[name];
  const r = fn(...args);
  if (r && typeof r.destroy === "function") {
    const v = r.toJs ? r.toJs() : r;
    r.destroy();
    return v;
  }
  return r;
}

const pyjson = async (name, ...args) => {
  const s = await pycall(name, ...args);
  return s == null ? null : JSON.parse(s);
};

// ---------------------------------------------------------------------------- 保存先（IndexedDB）
//   files: 音声（元のファイル・採譜にかけた音など）  キー "<id>:<名前>"
//   fs   : 曲のデータ（JSON）                         キー "/data/jobs/<id>/<ファイル名>"
const DB_NAME = "gakufu-engine";
let dbP = null;
function db() {
  if (!dbP) {
    dbP = new Promise((res, rej) => {
      const r = indexedDB.open(DB_NAME, 1);
      r.onupgradeneeded = () => {
        for (const s of ["files", "fs"]) if (!r.result.objectStoreNames.contains(s)) r.result.createObjectStore(s);
      };
      r.onsuccess = () => res(r.result);
      r.onerror = () => rej(r.error);
    });
  }
  return dbP;
}
async function idb(mode, fn, store = "files") {
  const d = await db();
  return new Promise((res, rej) => {
    const tx = d.transaction(store, mode);
    const st = tx.objectStore(store);
    let out;
    Promise.resolve(fn(st)).then((v) => (out = v));
    tx.oncomplete = () => res(out);
    tx.onerror = () => rej(tx.error);
    tx.onabort = () => rej(tx.error || new Error("保存に失敗しました（ブラウザの空き容量を確認してください）"));
  });
}
const putFile = (key, val) => idb("readwrite", (st) => void st.put(val, key));
const getFile = (key) =>
  idb("readonly", (st) => new Promise((res) => {
    const r = st.get(key);
    r.onsuccess = () => res(r.result);
    r.onerror = () => res(undefined);
  }));
const deleteJobFiles = async (jid) => {
  await idb("readwrite", (st) => void st.delete(IDBKeyRange.bound(`${jid}:`, `${jid}:￿`)));
  const dir = `/data/jobs/${jid}/`;
  clearTimeout(persistTimers.get(jid));
  persistTimers.delete(jid);
  await idb("readwrite", (st) => void st.delete(IDBKeyRange.bound(dir, dir + "￿")), "fs");
};

// ---------------------------------------------------------------------------- 処理の順番待ち
const queue = [];
let current = null;
const canceled = new Set();

class Canceled extends Error {}

function checkCancel(jid) {
  if (canceled.has(jid)) throw new Canceled();
}

function enqueue(task) {
  queue.push(task);
  void pump();
}

async function pump() {
  if (current) return;
  const task = queue.shift();
  if (!task) return;
  current = task;
  try {
    // 処理中の印（別のタブがこの曲を「中断した」と誤って扱わないように）
    if (navigator.locks?.request) await navigator.locks.request(`gakufu-job:${task.jid}`, () => task.run());
    else await task.run();
  } catch (e) {
    if (e instanceof Canceled) {
      await pycall("job_canceled", task.jid);
    } else {
      console.error(e);
      const msg = e?.userMessage || `処理中にエラーが発生しました: ${e?.message || e}`;
      try {
        await pycall("job_error", task.jid, msg);
      } catch (e2) {
        console.error(e2);
      }
    }
  } finally {
    canceled.delete(task.jid);
    await persistJob(task.jid, true);
    current = null;
    void pump();
  }
}

function userError(msg) {
  const e = new Error(msg);
  e.userMessage = msg;
  return e;
}

// ---------------------------------------------------------------------------- 進捗
function progressCb(jid, key) {
  let last = 0;
  return (frac, msg = "") => {
    checkCancel(jid);
    const now = Date.now();
    if (now - last < 250 && frac < 1) return;
    last = now;
    void pycall("job_progress", jid, key, frac, msg).then(() => persistJob(jid));
  };
}

// 工程の区切り（Python の job_step）を記録して保存する
async function step(jid, key, state, detail = "") {
  await pycall("job_step", jid, key, state, detail);
  persistJob(jid);
}
const sub = (cb, lo, hi) => (f, m) => cb(lo + (hi - lo) * f, m);
const yieldNow = () => new Promise((r) => setTimeout(r, 0));

// ---------------------------------------------------------------------------- 音源分離（Demucs htdemucs_6s）
const QUALITY = {
  fast: { shifts: 0, overlap: 0.25, label: "高速" },
  high: { shifts: 2, overlap: 0.25, label: "高品質" },
  ultra: { shifts: 5, overlap: 0.5, label: "最高（Virtuoso 用）" },
};
const SOURCES = ["drums", "bass", "other", "vocals", "guitar", "piano"];
const SEG = 343980; // 7.8 秒 × 44.1kHz

function mulberry32(a) {
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/**
 * demucs.apply.apply_model と同じ区切り方（重なりのある 7.8 秒区間・三角窓の重み・時間ずらしの平均）で分離し、
 * 各 Stem をモノラル 22.05kHz で返す。エネルギー（種類の判定用）は 44.1kHz のまま測る。
 */
async function separate(jid, L, R, quality, cb) {
  const q = QUALITY[quality] || QUALITY.high;
  const sess = await session("demucs", sub(cb, 0, 0.1));
  const n = L.length;
  // 曲全体のモノラルの平均・標準偏差で正規化（torch の std は不偏）
  let mean = 0;
  for (let i = 0; i < n; i++) mean += (L[i] + R[i]) / 2;
  mean /= n;
  let v = 0;
  for (let i = 0; i < n; i++) {
    const d = (L[i] + R[i]) / 2 - mean;
    v += d * d;
  }
  const std = Math.sqrt(v / Math.max(1, n - 1)) + 1e-8;
  const passes = q.shifts > 0 ? q.shifts : 1;
  const maxShift = q.shifts > 0 ? Math.floor(0.5 * 44100) : 0;
  // 正規化済みの波形（時間ずらし用に前後を 0 で埋めておく）
  const baseLen = n + 2 * maxShift;
  const bL = new Float32Array(baseLen);
  const bR = new Float32Array(baseLen);
  for (let i = 0; i < n; i++) {
    bL[maxShift + i] = (L[i] - mean) / std;
    bR[maxShift + i] = (R[i] - mean) / std;
  }
  const outLen = Math.ceil(n / 2);
  const acc = SOURCES.map(() => new Float32Array(outLen));
  const energy = SOURCES.map(() => 0);
  const stride = Math.floor((1 - q.overlap) * SEG);
  const weight = new Float32Array(SEG);
  const halfSeg = Math.floor(SEG / 2);
  for (let i = 0; i < halfSeg; i++) weight[i] = i + 1;
  for (let i = halfSeg; i < SEG; i++) weight[i] = SEG - i;
  let wmax = 0;
  for (let i = 0; i < SEG; i++) wmax = Math.max(wmax, weight[i]);
  for (let i = 0; i < SEG; i++) weight[i] /= wmax;
  const rand = mulberry32(12345);
  const shifts = [];
  for (let p = 0; p < passes; p++) shifts.push(maxShift ? Math.floor(rand() * (maxShift + 1)) : 0);
  const viewLens = shifts.map((o) => n + maxShift - o);
  // 区間の一覧（時間ずらしの回 p ごとに、重なりのある 7.8 秒区間）
  const tasks = [];
  for (let p = 0; p < passes; p++) {
    const o = shifts[p];
    const Lv = viewLens[p];
    const offsets = [];
    for (let off = 0; off < Lv; off += stride) offsets.push(off);
    const lens = offsets.map((off) => Math.min(Lv - off, SEG));
    for (let k = 0; k < offsets.length; k++) tasks.push({ o, k, offsets, lens, energy: p === 0 });
  }
  const ort = await ortLib();

  // 推論を始める（GPU に投げるだけで待たない）
  const startRun = (task) => {
    const { o, k, lens, offsets } = task;
    const len = lens[k];
    // TensorChunk.padded(SEG)：短い区間は前の音も含めて中央に置き、足りない所は 0
    const delta = SEG - len;
    const start = o + offsets[k] - Math.floor(delta / 2);
    const inL = new Float32Array(SEG);
    const inR = new Float32Array(SEG);
    const a = Math.max(0, start);
    const b = Math.min(baseLen, start + SEG);
    inL.set(bL.subarray(a, b), a - start);
    inR.set(bR.subarray(a, b), a - start);
    const spec = dsp.demucsSpec(inL, inR, SEG);
    const le = spec.length / 4 / 2048;
    const runP = sess.run({
      mix: new ort.Tensor("float32", interleave2(inL, inR), [1, 2, SEG]),
      spec: new ort.Tensor("float32", spec, [1, 4, 2048, le]),
    });
    return { task, runP, le, delta };
  };

  // 推論結果を Stem に戻して足し込む（JS の計算。次の区間の推論と並行して行う）
  const finish = ({ task, le, delta }, res) => {
    const { o, k, offsets, lens } = task;
    const off = offsets[k];
    const len = lens[k];
    const specOut = res.spec_out.data;
    const waveOut = res.wave_out.data;
    // 重なり部分の重みの合計（この区間の範囲だけ）
    const sw = new Float32Array(len);
    for (let j = Math.max(0, k - 3); j < Math.min(offsets.length, k + 4); j++) {
      const oj = offsets[j];
      const lj = lens[j];
      const s0 = Math.max(off, oj);
      const s1 = Math.min(off + len, oj + lj);
      for (let t = s0; t < s1; t++) sw[t - off] += weight[t - oj];
    }
    const trim = Math.floor(delta / 2);
    // view 上の位置 off+t → 元の曲の位置 off + t - (maxShift - o)
    const g0 = off - (maxShift - o);
    const contrib = new Float32Array(len);
    const add = (s, mono) => {
      for (let t = 0; t < len; t++) {
        const g = g0 + t;
        contrib[t] = g >= 0 && g < n ? (weight[t] / sw[t] / passes) * (mono[trim + t] * std + mean) : 0;
      }
      dsp.addDecimated(acc[s], contrib, g0);
    };
    if (task.energy) {
      // 1 回目の時間ずらしでは左右別々に戻し、種類の判定に使うエネルギーも測る
      for (let s = 0; s < SOURCES.length; s++) {
        const [yL, yR] = dsp.demucsStem(specOut, waveOut, s, le, SEG);
        let e = 0;
        const mono = new Float32Array(SEG);
        for (let t = 0; t < len; t++) {
          const a2 = weight[t] / sw[t];
          const l = yL[trim + t] * std + mean;
          const r = yR[trim + t] * std + mean;
          const g = g0 + t;
          if (g >= 0 && g < n) e += a2 * 0.5 * (l * l + r * r);
        }
        for (let i = 0; i < SEG; i++) mono[i] = 0.5 * (yL[i] + yR[i]);
        energy[s] += e;
        add(s, mono);
      }
    } else {
      // 2 回目以降はモノラルだけ（2 つの Stem を 1 回の iFFT でまとめて戻す）
      for (let s = 0; s < SOURCES.length; s += 2) {
        const [m1, m2] = dsp.demucsMonoPair(specOut, waveOut, s, s + 1 < SOURCES.length ? s + 1 : null, le, SEG);
        add(s, m1);
        if (m2) add(s + 1, m2);
      }
    }
  };

  let inflight = null;
  for (let i = 0; i <= tasks.length; i++) {
    checkCancel(jid);
    const prev = inflight;
    const res = prev ? await prev.runP : null;
    inflight = i < tasks.length ? startRun(tasks[i]) : null;
    // GPU に仕事を渡す時間を与えてから、前の区間の後処理をする
    if (inflight) await yieldNow();
    if (prev) {
      finish(prev, res);
      cb(0.1 + (0.9 * i) / tasks.length, `音源分離 htdemucs_6s（${q.label}）${i}/${tasks.length}`);
    }
  }
  const stems = {};
  const energies = {};
  SOURCES.forEach((s, i) => {
    stems[s] = acc[i];
    energies[s] = energy[i] / n;
  });
  return { stems, energies, label: q.label };
}

function interleave2(a, b) {
  const out = new Float32Array(a.length * 2);
  out.set(a, 0);
  out.set(b, a.length);
  return out;
}

// ---------------------------------------------------------------------------- 採譜
async function transcribeByteDance(jid, y22, cb) {
  const y = dsp.resample(y22, 22050, 16000);
  const seg = 160000;
  const front = await session("pianoFront", sub(cb, 0, 0.05));
  const rnn = await pianoRnn(sub(cb, 0.05, 0.1));
  const ort = await ortLib();
  const total = Math.ceil(y.length / seg) * seg;
  const audio = new Float32Array(total);
  audio.set(y);
  const starts = [];
  for (let s = 0; s + seg <= total; s += seg / 2) starts.push(s);
  const N = starts.length;
  const T = 1001;
  const frames = N === 1 ? T : 750 + (N - 2) * 500 + 750;
  const out88 = ["reg_onset_output", "reg_offset_output", "frame_output", "velocity_output"];
  const out1 = ["reg_pedal_onset_output", "reg_pedal_offset_output", "pedal_frame_output"];
  const res = {};
  for (const k of out88) res[k] = new Float32Array(frames * 88);
  for (const k of out1) res[k] = new Float32Array(frames);
  for (let i = 0; i < N; i++) {
    checkCancel(jid);
    const fo = await front.run({ audio: new ort.Tensor("float32", audio.slice(starts[i], starts[i] + seg), [1, seg]) });
    const feats = AMS.map((n) => fo[n].data);
    const o = await rnn.run(feats, T);
    // PianoTranscription.deframe：最後の 1 フレームを除き、前後 1/4 を捨ててつなぐ
    let a, b, dst;
    if (N === 1) [a, b, dst] = [0, T, 0];
    else if (i === 0) [a, b, dst] = [0, 750, 0];
    else if (i === N - 1) [a, b, dst] = [250, 1000, 750 + (i - 1) * 500];
    else [a, b, dst] = [250, 750, 750 + (i - 1) * 500];
    for (const k of out88) res[k].set(o[k].subarray(a * 88, b * 88), dst * 88);
    for (const k of out1) res[k].set(o[k].subarray(a, b), dst);
    cb(0.1 + (0.85 * (i + 1)) / N, `ピアノ採譜AI 推論中 ${i + 1}/${N} 区間（${rnn.kind === "webgpu" ? "GPU" : "CPU"}）`);
    await yieldNow();
  }
  cb(0.97, "音符に変換中");
  return pyjson("bytedance_notes", frames, ...out88.map((k) => res[k]), ...out1.map((k) => res[k]));
}

async function transcribeBasicPitch(jid, y22, profile, cb) {
  const sess = await session("basic", sub(cb, 0, 0.05));
  const ort = await ortLib();
  const N_SAMPLES = 43844, OVERLAP = 30 * 256, HOP = N_SAMPLES - OVERLAP, FRAMES = 172;
  const audio = new Float32Array(OVERLAP / 2 + y22.length);
  audio.set(y22, OVERLAP / 2);
  const starts = [];
  for (let i = 0; i < audio.length; i += HOP) starts.push(i);
  const note = new Float32Array(starts.length * FRAMES * 88);
  const onset = new Float32Array(starts.length * FRAMES * 88);
  const inName = sess.inputNames[0];
  for (let w = 0; w < starts.length; w++) {
    checkCancel(jid);
    const win = new Float32Array(N_SAMPLES);
    win.set(audio.subarray(starts[w], Math.min(audio.length, starts[w] + N_SAMPLES)));
    const o = await sess.run({ [inName]: new ort.Tensor("float32", win, [1, N_SAMPLES, 1]) });
    note.set(o["StatefulPartitionedCall:1"].data, w * FRAMES * 88);
    onset.set(o["StatefulPartitionedCall:2"].data, w * FRAMES * 88);
    if (w % 8 === 0) {
      cb(0.05 + (0.9 * (w + 1)) / starts.length, `Basic Pitch 推論中 ${w + 1}/${starts.length}`);
      await yieldNow();
    }
  }
  cb(0.96, "音符に変換中");
  const notes = await pyjson("basicpitch_notes", starts.length, y22.length, note, onset, profile);
  return { notes, pedals: [] };
}

async function beatThis(jid, mix22, cb) {
  try {
    const sess = await session("beat", sub(cb, 0, 0.3));
    const ort = await ortLib();
    const { data: spec, T } = dsp.btLogmel(mix22);
    const CH = 1500, BORDER = 6;
    const starts = [];
    for (let s = -BORDER; s < T - BORDER; s += CH - 2 * BORDER) starts.push(s);
    if (T > CH - 2 * BORDER) starts[starts.length - 1] = T - (CH - BORDER);
    const beat = new Float32Array(T).fill(-1000);
    const down = new Float32Array(T).fill(-1000);
    for (let i = starts.length - 1; i >= 0; i--) {
      checkCancel(jid);
      const s = starts[i];
      const chunk = new Float32Array(CH * 128);
      const a = Math.max(s, 0), b = Math.min(s + CH, T);
      chunk.set(spec.subarray(a * 128, b * 128), (a - s) * 128);
      const o = await sess.run({ spect: new ort.Tensor("float32", chunk, [1, CH, 128]) });
      const lo = s + BORDER, hi = Math.min(s + CH - BORDER, T);
      beat.set(o.beat.data.subarray(BORDER, BORDER + hi - lo), lo);
      down.set(o.downbeat.data.subarray(BORDER, BORDER + hi - lo), lo);
      cb(0.3 + (0.6 * (starts.length - i)) / starts.length, "拍と小節頭を推定中 (Beat This!)");
    }
    return dsp.btPeaks(beat, down);
  } catch (e) {
    if (e instanceof Canceled) throw e;
    console.warn("Beat This!", e);
    return { error: String(e?.message || e) };
  }
}

// ---------------------------------------------------------------------------- 1 曲の処理（pipeline.run と同じ流れ）
async function runJob(jid, input) {
  const { req, pcm } = input;
  const mode = req.mode || "standard";
  // 1. 取得
  await step(jid, "download", "running");
  await step(jid, "download", "done", input.sourceLabel);
  checkCancel(jid);

  // 2. 前処理
  await step(jid, "preprocess", "running");
  let chs = pcm.channels.length === 1 ? [pcm.channels[0], pcm.channels[0]] : pcm.channels.slice(0, 2);
  const trimmed = dsp.trimSilence(chs, 44100);
  chs = dsp.normalize(trimmed.chs);
  const [L, R] = chs;
  const analyzed = L.length / 44100;
  if (analyzed < 2) throw userError("音声が短すぎるか、ほぼ無音でした。");
  const offset = (input.startSec || 0) + trimmed.offset;
  persistJob(jid);
  await pycall("job_meta", jid, JSON.stringify({ analyzedDuration: Math.round(analyzed * 100) / 100, offset: Math.round(offset * 1000) / 1000, duration: input.duration || analyzed }));
  if (input.original) await putFile(`${jid}:original`, { blob: input.original, offset });
  // 解析用のモノラル 22.05kHz
  const mono = new Float32Array(L.length);
  for (let i = 0; i < L.length; i++) mono[i] = 0.5 * (L[i] + R[i]);
  const mix22 = new Float32Array(Math.ceil(L.length / 2));
  dsp.addDecimated(mix22, mono, 0);
  await step(jid, "preprocess", "done", `${analyzed.toFixed(1)}秒 / 44.1kHz ステレオ${input.note || ""}`);
  checkCancel(jid);

  // 3. 音源分離
  await step(jid, "separate", "running");
  const g = await gpuInfo();
  let sepQ = req.separation || "auto";
  if (sepQ === "auto") sepQ = { easy: "fast", standard: "high", virtuoso: "ultra" }[mode] || "high";
  let sep = null;
  let sepNote = "";
  if (sepQ !== "off") {
    if (!g.webgpu && req.separation === "auto") {
      sepNote = "（GPU が使えないため分離を省略）";
    } else {
      sep = await separate(jid, L, R, sepQ, progressCb(jid, "separate"));
    }
  }
  const plan = await pyjson("plan", sep ? JSON.stringify(sep.energies) : "", req.sourceType || "auto", req.engine || "auto");
  const sepInfo = { quality: sepQ, engine: sep ? `Demucs htdemucs_6s（${sep.label}）` : null };
  // 採譜にかける音
  let target;
  if (!plan.use) target = mix22;
  else {
    target = new Float32Array(mix22.length);
    for (const s of plan.use) {
      const x = sep.stems[s];
      for (let i = 0; i < target.length; i++) target[i] += x[i];
    }
  }
  await putFile(`${jid}:target`, dsp.wavBlob(target, 22050));
  let vocals16 = null, bass16 = null;
  if (sep) {
    vocals16 = dsp.resample(sep.stems.vocals, 22050, 16000);
    bass16 = dsp.resample(sep.stems.bass, 22050, 16000);
    await putFile(`${jid}:vocals16`, dsp.wavBlob(vocals16, 16000));
    await putFile(`${jid}:bass16`, dsp.wavBlob(bass16, 16000));
  }
  await putFile(`${jid}:info`, { stems: sep ? Object.keys(sep.energies) : [] });
  await step(jid, "separate", sep ? "done" : "skipped", `${sepInfo.engine || "分離なし" + sepNote} → 判定: ${plan.kindLabel}（${plan.ratioText}）`);
  checkCancel(jid);

  // 4. 採譜
  const t = await transcribeStep(jid, plan, target, vocals16, bass16, "transcribe");
  const detail = await pycall("save_raw", jid, JSON.stringify(t.notes), JSON.stringify(t.pedals), t.engine, plan.engineId, plan.kind, plan.label);
  await step(jid, "transcribe", "done", detail);
  checkCancel(jid);

  // 5. 解析
  await step(jid, "analyze", "running");
  const cbA = progressCb(jid, "analyze");
  const dnn = await beatThis(jid, mix22, cbA);
  cbA(0.92, "テンポ・拍子・調を解析中");
  const an = await pyjson("analyze", jid, mix22, plan.kind, JSON.stringify(plan.ratios), JSON.stringify(dnn));
  await step(jid, "analyze", "done", an.detail);
  checkCancel(jid);

  // 6. アレンジ
  await step(jid, "arrange", "running");
  const ar = await pyjson("arrange_first", jid);
  await step(jid, "arrange", "done", ar.detail);
  await pycall("job_finish", jid, JSON.stringify(sepInfo), plan.kind, t.engine, plan.engineId, plan.label);
}

async function transcribeStep(jid, plan, target, vocals16, bass16, stepKey) {
  await step(jid, stepKey, "running");
  const cb = progressCb(jid, stepKey);
  const hi = plan.song ? 0.75 : 1.0;
  let main;
  if (plan.engineId === "bytedance") main = await transcribeByteDance(jid, target, sub(cb, 0, hi));
  else main = await transcribeBasicPitch(jid, target, plan.profile, sub(cb, 0, hi));
  let mel = "", bass = "";
  if (plan.song && vocals16) {
    cb(0.8, "歌メロの音高追跡 (pYIN)");
    await yieldNow();
    mel = await pycall("track_part", vocals16, "vocals");
    if (plan.bass && bass16) {
      cb(0.9, "ベースの音高追跡 (pYIN)");
      await yieldNow();
      bass = await pycall("track_part", bass16, "bass");
    } else bass = "[]";
  }
  const asm = await pyjson("assemble_notes", plan.engineId, plan.kind, JSON.stringify(main.notes), mel, bass);
  return { notes: asm.notes, pedals: main.pedals, engine: asm.engine };
}

async function runRetranscribe(jid, engine, start) {
  const target = await getFile(`${jid}:target`);
  if (!target) throw userError("このブラウザに音声が残っていないため、採譜をやり直せません。");
  const info = (await getFile(`${jid}:info`)) || { stems: [] };
  const y = (await dsp.wavRead(target)).x;
  const v = await getFile(`${jid}:vocals16`);
  const b = await getFile(`${jid}:bass16`);
  const vocals16 = v ? (await dsp.wavRead(v)).x : null;
  const bass16 = b ? (await dsp.wavRead(b)).x : null;
  const p = await pyjson("retranscribe_plan", start.kind, JSON.stringify(start.ratios), JSON.stringify(info.stems), engine);
  const plan = { ...p, kind: start.kind, label: start.target };
  const t = await transcribeStep(jid, plan, y, vocals16, bass16, "transcribe");
  await pycall("finish_retranscribe", jid, JSON.stringify(t.notes), JSON.stringify(t.pedals), t.engine, plan.engineId, plan.kind, plan.label);
}

// ---------------------------------------------------------------------------- 窓口
async function systemInfo() {
  const g = await gpuInfo();
  const cached = await modelCached("pianoFront");
  return {
    version: "browser",
    browser: true,
    device: g.webgpu ? "webgpu" : "wasm",
    gpu: g.name,
    ffmpeg: false,
    ytdlp: null,
    engines: {
      bytedance: { available: true, weights: cached },
      demucs: g.webgpu,
      basicpitch: true,
      builtin: false,
    },
    limits: { maxAnalyzeSec: MAX_ANALYZE_SEC, maxSourceSec: 3600, maxUploadMB: 2048 },
  };
}

async function api({ method, path, query, body }) {
  const m = /^\/api\/jobs\/([0-9a-f]{12})(\/.*)?$/.exec(path);
  const jid = m?.[1];
  // 一覧と曲の状態は、別のタブでの変更を読み直してから返す
  if (method === "GET" && path === "/api/jobs") await reloadList();
  else if (method === "GET" && jid && !m[2]) await reloadJob(jid);
  const r = await pyjson("handle", method, path, query || "", body || "");
  // 楽譜を開く・保存する・編曲し直すとファイルが増減するので、その曲を保存する
  if (jid && method !== "DELETE") persistJob(jid);
  return r;
}

const handlers = {
  async system() {
    void py(); // Python の準備を先に始めておく
    return systemInfo();
  },

  api,

  async upload({ req, meta, pcm, original, sourceLabel, note, startSec, duration }) {
    const job = await pyjson("job_create", JSON.stringify(req), JSON.stringify(meta));
    await persistJob(job.id, true);
    enqueue({ jid: job.id, run: () => runJob(job.id, { req, pcm, original, sourceLabel, note, startSec, duration }) });
    return job;
  },

  async cancel({ jid }) {
    if (current?.jid === jid) canceled.add(jid);
    const i = queue.findIndex((t) => t.jid === jid);
    if (i >= 0) {
      queue.splice(i, 1);
      await pycall("job_canceled", jid);
      await persistJob(jid, true);
    }
    return { ok: true };
  },

  async remove({ jid }) {
    if (current?.jid === jid) canceled.add(jid);
    const r = await api({ method: "DELETE", path: `/api/jobs/${jid}` });
    await deleteJobFiles(jid);
    return r;
  },

  async retranscribe({ jid, engine }) {
    const start = await pyjson("job_retranscribe_start", jid, engine);
    if (start.error) return { status: start.error.status, json: { detail: start.error.detail } };
    await persistJob(jid, true);
    enqueue({ jid, run: () => runRetranscribe(jid, engine, start) });
    return { status: 200, json: start.job };
  },

  async audio({ jid, which }) {
    if (which === "original") {
      const o = await getFile(`${jid}:original`);
      if (o) return { blob: o.blob, offset: o.offset };
    }
    const t = await getFile(`${jid}:target`);
    if (!t) return null;
    return { blob: t, offset: 0 };
  },
};

async function onMessage(e) {
  const { id, type, payload } = e.data || {};
  const h = handlers[type];
  if (!h) return;
  try {
    const result = await h(payload || {});
    self.postMessage({ id, ok: true, result });
  } catch (err) {
    console.error(err);
    self.postMessage({ id, ok: false, error: err?.userMessage || err?.message || String(err) });
  }
}

resolveReady();
