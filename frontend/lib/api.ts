import type { ArrangeOptions, DisplayOptions, Instrument, Job, JobSummary, Mode, ProbeResult, Score, SystemInfo } from "./types";
import { b64ToBlob, localAudio, localRequest, startLocalJob } from "./localEngine";

/** 各自の PC で動く「楽譜AIエンジン」（インストール版）の既定の場所 */
export const DEFAULT_ENGINE = "http://127.0.0.1:8541";
const ENGINE_KEY = "gakufu.engine";
const MODE_KEY = "gakufu.engineMode";

/**
 * 処理する場所。
 *  - browser: このブラウザの中（WebGPU）。インストール不要。既定。
 *  - server : インストール版の楽譜AIエンジン（URL から直接取り込める・PyTorch で動く）
 */
export type EngineMode = "browser" | "server";

export function engineMode(): EngineMode {
  if (typeof window === "undefined") return "browser";
  try {
    const v = localStorage.getItem(MODE_KEY);
    if (v === "browser" || v === "server") return v;
  } catch {
    /* 保存できない環境 */
  }
  // インストール版エンジン自身が配っている画面（:8541）だけは、そのエンジンを使う
  return window.location.port === "8541" ? "server" : "browser";
}

export function setEngineMode(m: EngineMode) {
  try {
    localStorage.setItem(MODE_KEY, m);
  } catch {
    /* 保存できない環境 */
  }
}

const isBrowserMode = () => engineMode() === "browser";

/**
 * 公開サイト（GitHub Pages など）から開かれているか。
 * 自分の PC の画面は「開発サーバー :8540」か「エンジン自身 :8541」だけなので、それ以外は公開サイト扱い。
 */
export function isPublicSite(): boolean {
  if (typeof window === "undefined") return false;
  const { hostname: h, port } = window.location;
  const loop = h === "localhost" || h === "127.0.0.1" || h === "[::1]";
  return !(loop && (port === "8540" || port === "8541"));
}

export function engineUrl(): string {
  try {
    return (localStorage.getItem(ENGINE_KEY) || DEFAULT_ENGINE).replace(/\/$/, "");
  } catch {
    return DEFAULT_ENGINE;
  }
}

export function setEngineUrl(url: string) {
  try {
    if (!url || url === DEFAULT_ENGINE) localStorage.removeItem(ENGINE_KEY);
    else localStorage.setItem(ENGINE_KEY, url.replace(/\/$/, ""));
  } catch {
    /* 保存できない環境では既定のまま */
  }
}

/**
 * API の置き場所。
 *  - 公開サイトから開いたとき：その人の PC で動いているエンジン（http://127.0.0.1:8541）
 *  - 開発中（:8540）：:8541
 *  - エンジン自身が配っている画面（:8541）：同じオリジン
 */
export function apiBase(): string {
  const env = process.env.NEXT_PUBLIC_API_BASE;
  if (env) return env.replace(/\/$/, "");
  if (typeof window === "undefined") return "";
  if (isPublicSite()) return engineUrl();
  const { protocol, hostname, port } = window.location;
  if (port === "8540") return `${protocol}//${hostname}:8541`;
  return "";
}

export class ApiError extends Error {
  status: number;
  constructor(status: number, message: string) {
    super(message);
    this.status = status;
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  if (isBrowserMode()) {
    const r = await localRequest(path, init);
    if (r.status >= 400) {
      const d = (r.json as { detail?: unknown } | undefined)?.detail;
      throw new ApiError(r.status, typeof d === "string" ? d : d ? JSON.stringify(d) : `エラー ${r.status}`);
    }
    return (r.json !== undefined ? r.json : r.text) as T;
  }
  let res: Response;
  try {
    res = await fetch(apiBase() + path, init);
  } catch {
    throw new ApiError(0, "楽譜AIエンジンにつながりません。あなたのPCでエンジンが起動しているか確認してください。");
  }
  if (!res.ok) {
    let msg = `${res.status} ${res.statusText}`;
    try {
      const j = await res.json();
      if (j?.detail) msg = typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail);
    } catch {
      /* 本文が JSON でない */
    }
    throw new ApiError(res.status, msg);
  }
  const ct = res.headers.get("content-type") || "";
  if (ct.includes("application/json")) return (await res.json()) as T;
  return (await res.text()) as unknown as T;
}

const json = (body: unknown): RequestInit => ({
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(body),
});

async function fileResponse(path: string, init?: RequestInit, fallbackName = "download"): Promise<{ blob: Blob; name: string }> {
  if (isBrowserMode()) {
    const r = await localRequest(path, init);
    if (r.status >= 400 || !r.b64) {
      const d = (r.json as { detail?: string } | undefined)?.detail;
      throw new ApiError(r.status, d || "書き出しに失敗しました");
    }
    return { blob: b64ToBlob(r.b64, r.type || "application/octet-stream"), name: r.filename || fallbackName };
  }
  const res = await fetch(`${apiBase()}${path}`, init);
  if (!res.ok) throw new ApiError(res.status, "書き出しに失敗しました");
  const disp = res.headers.get("content-disposition") || "";
  const m = /filename\*=UTF-8''([^;]+)/i.exec(disp);
  return { blob: await res.blob(), name: m ? decodeURIComponent(m[1]) : fallbackName };
}

export const api = {
  system: () => request<SystemInfo>("/api/system"),
  probe: (url: string) => request<ProbeResult>("/api/probe", json({ url })),
  /** ブラウザ版：録音やファイルの音声から採譜を始める */
  startFromBlob: (
    blob: Blob,
    fields: { mode: Mode; sourceType: string; separation: string; start?: number | null; end?: number | null; options?: ArrangeOptions },
    info: { title: string; platform: string; filename?: string; url?: string; sourceLabel: string },
  ) => startLocalJob(blob, fields, info) as Promise<Job>,
  createJob: (body: {
    url: string;
    mode: Mode;
    sourceType: string;
    separation: string;
    start?: number | null;
    end?: number | null;
    options?: ArrangeOptions;
  }) => request<Job>("/api/jobs", json(body)),
  uploadJob: (
    file: File,
    fields: { mode: Mode; sourceType: string; separation: string; start?: number | null; end?: number | null; options?: ArrangeOptions },
  ) => {
    if (isBrowserMode()) {
      const name = file.name || "upload";
      return startLocalJob(file, fields, {
        title: name.replace(/\.[^.]+$/, ""),
        platform: "upload",
        filename: name,
        sourceLabel: `ファイル: ${name}`,
      }) as Promise<Job>;
    }
    const fd = new FormData();
    fd.append("file", file);
    fd.append("mode", fields.mode);
    fd.append("sourceType", fields.sourceType);
    fd.append("separation", fields.separation);
    if (fields.start != null) fd.append("start", String(fields.start));
    if (fields.end != null) fd.append("end", String(fields.end));
    fd.append("options", JSON.stringify(fields.options ?? {}));
    return request<Job>("/api/jobs/upload", { method: "POST", body: fd });
  },
  listJobs: () => request<JobSummary[]>("/api/jobs"),
  job: (id: string) => request<Job>(`/api/jobs/${id}`),
  cancel: (id: string) => request<{ ok: boolean }>(`/api/jobs/${id}/cancel`, { method: "POST" }),
  retranscribe: (id: string, engine: "auto" | "bytedance" | "basicpitch") => request<Job>(`/api/jobs/${id}/retranscribe`, json({ engine })),
  remove: (id: string) => request<{ ok: boolean }>(`/api/jobs/${id}`, { method: "DELETE" }),
  score: (id: string, mode: Mode, instrument: Instrument = "piano") =>
    request<Score>(`/api/jobs/${id}/score?mode=${mode}&instrument=${instrument}`),
  saveScore: (id: string, mode: Mode, score: Score, instrument: Instrument = "piano") =>
    request<{ ok: boolean }>(`/api/jobs/${id}/score?mode=${mode}&instrument=${instrument}`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ score }),
    }),
  rearrange: (
    id: string,
    mode: Mode,
    options: ArrangeOptions,
    instrument: Instrument = "piano",
    guitar: { tuning?: string; capo?: number | "auto" } = {},
  ) => request<Score>(`/api/jobs/${id}/arrange`, json({ mode, options, instrument, guitar })),
  reset: (id: string, mode: Mode, instrument: Instrument = "piano") =>
    request<Score>(`/api/jobs/${id}/reset?mode=${mode}&instrument=${instrument}`, { method: "POST" }),
  musicxml: (score: Score, options: Partial<DisplayOptions> & { subtitle?: string }) =>
    request<string>("/api/notation/musicxml", json({ score, options })),
  fingering: (score: Score) => request<Score>("/api/notation/fingering", json({ score })),
  /** 再生用の音声。ブラウザ版は保存してある音声（と楽譜の 0 秒の位置）を返す */
  audioSource: (id: string, which: "original" | "target"): string | Promise<{ url: string; offset: number }> =>
    isBrowserMode() ? localAudio(id, which) : `${apiBase()}/api/jobs/${id}/audio/${which}`,
  rawMidi: (id: string) => fileResponse(`/api/jobs/${id}/export/raw-midi`, undefined, "raw.mid"),
  exportBlob: (kind: "midi" | "musicxml" | "tab", score: Score, options?: Partial<DisplayOptions>) =>
    fileResponse(`/api/export/${kind}`, json({ score, options: options ?? {} }), `score.${kind === "midi" ? "mid" : kind === "tab" ? "txt" : "musicxml"}`),
};
