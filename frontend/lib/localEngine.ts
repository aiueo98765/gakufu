/**
 * ブラウザ版エンジン（public/engine/worker.js）との仲立ち。
 *
 * 採譜はすべて閲覧している人の PC の中（WebGPU / WASM）で行う。動画・音声ファイルの読み込み（デコード）は
 * Worker では使えない Web Audio API が要るので、ここ（画面側）で行ってから Worker に渡す。
 */
import type { ArrangeOptions, Job, Mode } from "./types";

const BASE = process.env.NEXT_PUBLIC_BASE_PATH || "";
const BUILD = process.env.NEXT_PUBLIC_BUILD_ID || "dev";

type Pending = { resolve: (v: unknown) => void; reject: (e: Error) => void };
let worker: Worker | null = null;
let seq = 0;
const pending = new Map<number, Pending>();

function getWorker(): Worker {
  if (!worker) {
    // gakufu.forceCpu=1 のときは GPU を使わない（WebGPU の無い PC での動き方の確認用）
    let cpu = false;
    try {
      cpu = localStorage.getItem("gakufu.forceCpu") === "1";
    } catch {
      cpu = false;
    }
    worker = new Worker(`${BASE}/engine/worker.js?v=${encodeURIComponent(BUILD)}`, { type: "module", name: cpu ? "gakufu-cpu" : "gakufu" });
    worker.onmessage = (e: MessageEvent) => {
      const { id, ok, result, error } = e.data || {};
      const p = pending.get(id);
      if (!p) return;
      pending.delete(id);
      if (ok) p.resolve(result);
      else p.reject(new Error(error || "ブラウザ内エンジンでエラーが発生しました"));
    };
    worker.onerror = (e) => {
      const err = new Error(`ブラウザ内エンジンを起動できませんでした: ${e.message || "不明なエラー"}`);
      for (const p of pending.values()) p.reject(err);
      pending.clear();
    };
  }
  return worker;
}

export function call<T>(type: string, payload?: unknown, transfer: Transferable[] = []): Promise<T> {
  const id = ++seq;
  return new Promise<T>((resolve, reject) => {
    pending.set(id, { resolve: resolve as (v: unknown) => void, reject });
    getWorker().postMessage({ id, type, payload }, transfer);
  });
}

// 開発中だけ、コンソールからエンジンを直接呼べるようにする（動作確認用）
if (typeof window !== "undefined" && process.env.NODE_ENV !== "production") {
  (window as unknown as { __gakufuCall: typeof call }).__gakufuCall = call;
}

export interface LocalResponse {
  status: number;
  json?: unknown;
  text?: string;
  b64?: string;
  filename?: string;
  type?: string;
}

/** fetch の代わり：/api/... を Worker の中の Python に渡す */
export async function localRequest(path: string, init?: RequestInit): Promise<LocalResponse> {
  const u = new URL(path, "http://local");
  const method = (init?.method || "GET").toUpperCase();
  const body = typeof init?.body === "string" ? init.body : "";
  if (u.pathname === "/api/system") return { status: 200, json: await call("system") };
  const m = /^\/api\/jobs\/([0-9a-f]{12})(\/.*)?$/.exec(u.pathname);
  if (m && m[2] === "/cancel" && method === "POST") return { status: 200, json: await call("cancel", { jid: m[1] }) };
  if (m && !m[2] && method === "DELETE") {
    const r = await call<LocalResponse>("remove", { jid: m[1] });
    return r;
  }
  if (m && m[2] === "/retranscribe" && method === "POST") {
    const { engine } = JSON.parse(body || "{}");
    return call<LocalResponse>("retranscribe", { jid: m[1], engine: engine || "auto" });
  }
  return call<LocalResponse>("api", { method, path: u.pathname, query: u.search.replace(/^\?/, ""), body });
}

export function b64ToBlob(b64: string, type: string): Blob {
  const bin = atob(b64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return new Blob([bytes], { type });
}

// ---------------------------------------------------------------------------- 音声の読み込み
export const MAX_ANALYZE_SEC = 600;

export interface JobFields {
  mode: Mode;
  sourceType: string;
  separation: string;
  start?: number | null;
  end?: number | null;
  options?: ArrangeOptions;
}

async function decode(blob: Blob): Promise<AudioBuffer> {
  const buf = await blob.arrayBuffer();
  // 44.1kHz に変換しながら読む（動画の音声トラックも読める）
  const ctx = new OfflineAudioContext(2, 1, 44100);
  try {
    return await ctx.decodeAudioData(buf);
  } catch {
    throw new Error("このファイルの音声を読み込めませんでした。mp3 / m4a / wav / mp4 などを試してください。");
  }
}

/**
 * 音声・動画（ファイルや録音）を読み込み、Worker に渡して採譜を始める。
 * 区間指定と長さの上限はここで切り出す。
 */
export async function startLocalJob(
  blob: Blob,
  fields: JobFields,
  info: { title: string; platform: string; filename?: string; url?: string; sourceLabel: string },
) {
  // AI モデルと曲のデータをブラウザに置き続けられるよう頼む（容量が足りないときに消されにくくなる）
  void navigator.storage?.persist?.().catch(() => false);
  const audio = await decode(blob);
  const sr = audio.sampleRate;
  const total = audio.length / sr;
  let s = Math.max(0, fields.start ?? 0);
  let e = fields.end && fields.end > s ? Math.min(fields.end, total) : total;
  let note = "";
  if (e - s > MAX_ANALYZE_SEC) {
    e = s + MAX_ANALYZE_SEC;
    note = `（長いので${fields.start ? "指定位置から" : "先頭"}${MAX_ANALYZE_SEC / 60}分を解析）`;
  }
  const a = Math.floor(s * sr);
  const b = Math.max(a + 1, Math.floor(e * sr));
  const channels: Float32Array[] = [];
  for (let c = 0; c < Math.min(2, audio.numberOfChannels); c++) channels.push(audio.getChannelData(c).slice(a, b));
  const req = {
    kind: "upload",
    filename: info.filename ?? info.title,
    url: info.url,
    platform: info.platform,
    mode: fields.mode,
    sourceType: fields.sourceType,
    separation: fields.separation,
    start: fields.start ?? null,
    end: fields.end ?? null,
    options: fields.options ?? {},
  };
  const meta = { title: info.title, platform: info.platform, duration: Math.round(total * 100) / 100, webpage_url: info.url };
  return call<{ id: string }>(
    "upload",
    {
      req,
      meta,
      pcm: { sr: 44100, channels },
      original: blob,
      sourceLabel: info.sourceLabel,
      note,
      startSec: s,
      duration: total,
    },
    channels.map((c) => c.buffer),
  );
}

/**
 * 中断した曲などを、ブラウザに保存してある元の音声から同じ設定でもう一度実行する。
 * 新しい曲として始め、元の曲は消す。
 */
export async function rerunLocalJob(job: Job): Promise<{ id: string }> {
  const o = await call<{ blob: Blob; offset: number } | null>("audio", { jid: job.id, which: "source" });
  if (!o) throw new Error("この曲の元の音声がブラウザに残っていないため、もう一度実行できません。ファイルを選び直してください。");
  const r = job.request;
  const created = await startLocalJob(
    o.blob,
    { mode: r.mode, sourceType: r.sourceType, separation: r.separation, start: r.start ?? null, end: r.end ?? null, options: r.options ?? {} },
    {
      title: job.meta.title || r.filename || "曲",
      platform: r.platform,
      filename: r.filename,
      url: r.url,
      sourceLabel: job.steps.download?.detail || "保存してあった音声",
    },
  );
  await call("remove", { jid: job.id });
  return created;
}

/** 再生用の音声（元の音・採譜にかけた音）。offset は楽譜の 0 秒が音声の何秒目か */
export async function localAudio(jid: string, which: "original" | "target"): Promise<{ url: string; offset: number }> {
  const r = await call<{ blob: Blob; offset: number } | null>("audio", { jid, which });
  if (!r) throw new Error("このブラウザに音声が残っていません");
  return { url: URL.createObjectURL(r.blob), offset: r.offset || 0 };
}

// ---------------------------------------------------------------------------- タブの音の録音（URL の動画用）
export function canCaptureTab(): boolean {
  return typeof navigator !== "undefined" && !!navigator.mediaDevices?.getDisplayMedia && typeof MediaRecorder !== "undefined";
}

export interface TabRecording {
  stop: () => Promise<Blob>;
  cancel: () => void;
  level: () => number;
  elapsed: () => number;
  onEnded?: () => void;
}

/**
 * 別のタブで再生している動画の音を録音する（Chrome / Edge の「タブを共有」＋「タブの音声を共有」）。
 * 映像は使わないのですぐ止める。
 */
export async function recordTab(): Promise<TabRecording> {
  const stream = await navigator.mediaDevices.getDisplayMedia({
    video: true,
    audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false } as MediaTrackConstraints,
    // @ts-expect-error Chrome の拡張オプション
    preferCurrentTab: false,
    selfBrowserSurface: "exclude",
    surfaceSwitching: "exclude",
    systemAudio: "exclude",
  });
  const audioTracks = stream.getAudioTracks();
  stream.getVideoTracks().forEach((t) => t.stop());
  if (!audioTracks.length) {
    stream.getTracks().forEach((t) => t.stop());
    throw new Error("音声が共有されませんでした。共有するときに「タブの音声も共有する」をオンにしてください。");
  }
  const audioStream = new MediaStream(audioTracks);
  const mime = ["audio/webm;codecs=opus", "audio/webm", "audio/ogg;codecs=opus"].find((t) => MediaRecorder.isTypeSupported(t)) || "";
  const rec = new MediaRecorder(audioStream, mime ? { mimeType: mime, audioBitsPerSecond: 256000 } : undefined);
  const chunks: Blob[] = [];
  rec.ondataavailable = (e) => e.data.size && chunks.push(e.data);
  // 音量メーター
  const ctx = new AudioContext();
  const src = ctx.createMediaStreamSource(audioStream);
  const an = ctx.createAnalyser();
  an.fftSize = 1024;
  src.connect(an);
  const buf = new Float32Array(an.fftSize);
  const t0 = performance.now();
  rec.start(1000);
  const handle: TabRecording = {
    level: () => {
      an.getFloatTimeDomainData(buf);
      let m = 0;
      for (let i = 0; i < buf.length; i++) m = Math.max(m, Math.abs(buf[i]));
      return m;
    },
    elapsed: () => (performance.now() - t0) / 1000,
    stop: () =>
      new Promise<Blob>((resolve) => {
        rec.onstop = () => {
          audioTracks.forEach((t) => t.stop());
          void ctx.close();
          resolve(new Blob(chunks, { type: rec.mimeType || "audio/webm" }));
        };
        if (rec.state !== "inactive") rec.stop();
        else rec.onstop?.(new Event("stop"));
      }),
    cancel: () => {
      try {
        rec.stop();
      } catch {
        /* 止まっている */
      }
      audioTracks.forEach((t) => t.stop());
      void ctx.close();
    },
  };
  // 共有を止められたら（共有バーの「共有を停止」）知らせる
  audioTracks[0].addEventListener("ended", () => handle.onEnded?.());
  return handle;
}
