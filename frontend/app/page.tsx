"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import {
  ChevronDown,
  Clock,
  Cpu,
  LoaderCircle,
  Piano,
  SlidersHorizontal,
  Trash2,
  TriangleAlert,
  Upload,
  Wand2,
} from "lucide-react";
import { api, engineMode, setEngineMode, type EngineMode } from "@/lib/api";
import { formatSec, parseTime } from "@/lib/music";
import type { JobSummary, Mode, ProbeResult, SystemInfo } from "@/lib/types";
import { ModeCards, MODE_INFO } from "@/components/ModeCards";
import { EngineGate } from "@/components/EngineGate";
import { PlatformIcon, PLATFORM_LABEL, detectPlatform } from "@/components/PlatformIcon";
import { TabCapture } from "@/components/TabCapture";

const SOURCE_TYPES = [
  { id: "auto", label: "自動判定", hint: "Stem の比率から判定" },
  { id: "piano", label: "ピアノ演奏", hint: "弾いてみた・ピアノカバー" },
  { id: "song", label: "歌もの", hint: "歌メロ＋伴奏でピアノ化" },
  { id: "instrumental", label: "器楽", hint: "ドラム以外を全部採譜" },
];
const SEPARATIONS = [
  { id: "auto", label: "難易度に合わせる" },
  { id: "fast", label: "高速" },
  { id: "high", label: "高品質" },
  { id: "ultra", label: "最高（5回アンサンブル）" },
  { id: "off", label: "分離しない" },
];
const LEFT_HANDS = [
  { id: "auto", label: "おまかせ" },
  { id: "root", label: "ルート音のみ" },
  { id: "fifth", label: "ルート＋5度" },
  { id: "triad", label: "3和音" },
  { id: "block", label: "ベース＋和音" },
  { id: "broken", label: "分散和音（8分）" },
  { id: "alberti", label: "アルベルティ・バス" },
  { id: "original", label: "原曲どおり" },
];

export default function Home() {
  const router = useRouter();
  const [tab, setTab] = useState<"url" | "file">("url");
  const [url, setUrl] = useState("");
  const [file, setFile] = useState<File | null>(null);
  const [mode, setMode] = useState<Mode>("standard");
  const [sourceType, setSourceType] = useState("auto");
  const [separation, setSeparation] = useState("auto");
  const [start, setStart] = useState("");
  const [end, setEnd] = useState("");
  const [timeSig, setTimeSig] = useState("auto");
  const [leftHand, setLeftHand] = useState("auto");
  const [easyKey, setEasyKey] = useState(false);
  const [advanced, setAdvanced] = useState(false);
  const [probe, setProbe] = useState<ProbeResult | null>(null);
  const [probing, setProbing] = useState(false);
  const [probeErr, setProbeErr] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [jobs, setJobs] = useState<JobSummary[]>([]);
  const [sys, setSys] = useState<SystemInfo | null>(null);
  const [sysErr, setSysErr] = useState<string | null>(null);
  const [drag, setDrag] = useState(false);
  const fileInput = useRef<HTMLInputElement>(null);
  // 処理する場所（このブラウザ / インストール版のエンジン）。画面を作った後に読む（サーバー側の描画とそろえるため）
  const [where, setWhere] = useState<EngineMode>("browser");
  const browser = where === "browser";

  const platform = useMemo(() => detectPlatform(url), [url]);

  const [checking, setChecking] = useState(false);
  const connect = async () => {
    setChecking(true);
    try {
      setSys(await api.system());
      setSysErr(null);
    } catch (e) {
      setSysErr((e as Error).message);
    } finally {
      setChecking(false);
    }
  };

  useEffect(() => {
    setWhere(engineMode());
  }, []);

  useEffect(() => {
    setSys(null);
    setSysErr(null);
    setJobs([]);
    void connect();
    const load = () => api.listJobs().then(setJobs).catch(() => undefined);
    load();
    const t = setInterval(load, 4000);
    return () => clearInterval(t);
  }, [where]);

  const switchWhere = (w: EngineMode) => {
    setEngineMode(w);
    setWhere(w);
  };

  // エンジンにつながっていない間は、数秒ごとにそっと確かめる（起動したら自動で切り替わる）
  useEffect(() => {
    if (!sysErr) return;
    const t = setInterval(() => {
      api
        .system()
        .then((s) => {
          setSys(s);
          setSysErr(null);
        })
        .catch(() => undefined);
    }, 4000);
    return () => clearInterval(t);
  }, [sysErr]);

  // URL を貼ったら題名と長さを先に見せる
  useEffect(() => {
    setProbe(null);
    setProbeErr(null);
    if (!platform || browser) return;
    const t = setTimeout(async () => {
      setProbing(true);
      try {
        setProbe(await api.probe(url));
      } catch (e) {
        setProbeErr((e as Error).message);
      } finally {
        setProbing(false);
      }
    }, 600);
    return () => clearTimeout(t);
  }, [url, platform, browser]);

  function range(): { s: number | null; en: number | null } | null {
    const s = parseTime(start);
    const en = parseTime(end);
    let msg = "";
    if (start && s === null) msg = "開始位置は 1:23 や 83 の形式で入力してください";
    else if (end && en === null) msg = "終了位置は 1:23 や 83 の形式で入力してください";
    else if (s !== null && en !== null && en <= s) msg = "終了位置は開始位置より後にしてください";
    if (msg) {
      setError(msg);
      return null;
    }
    return { s, en };
  }

  // ブラウザ版：別タブの動画を録音した音から始める
  async function startRecorded(blob: Blob, info: { title: string; platform: string; url: string; seconds: number }) {
    setError(null);
    const r = range();
    if (!r) return;
    setSubmitting(true);
    try {
      const options = { timeSig: timeSig as "auto", leftHand, easyKey };
      const job = await api.startFromBlob(
        blob,
        { mode, sourceType, separation, start: r.s, end: r.en, options },
        { title: info.title, platform: info.platform, url: info.url, sourceLabel: `タブの録音 ${formatSec(info.seconds)}` },
      );
      router.push(`/job/?id=${job.id}`);
    } catch (err) {
      setError((err as Error).message);
      setSubmitting(false);
    }
  }

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    setError(null);
    const r = range();
    if (!r) return;
    const { s, en } = r;
    setSubmitting(true);
    try {
      const options = { timeSig: timeSig as "auto", leftHand, easyKey };
      const job =
        tab === "url"
          ? await api.createJob({ url, mode, sourceType, separation, start: s, end: en, options })
          : await api.uploadJob(file as File, { mode, sourceType, separation, start: s, end: en, options });
      router.push(`/job/?id=${job.id}`);
    } catch (err) {
      setError((err as Error).message);
      setSubmitting(false);
    }
  }

  const canSubmit = tab === "url" ? !!platform : !!file;

  return (
    <main className="bg-aurora min-h-screen">
      <div className="mx-auto max-w-6xl px-4 pb-24 pt-10 sm:px-6">
        <header className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="grid size-10 place-items-center rounded-xl bg-gradient-to-br from-indigo-500 to-fuchsia-500 shadow-lg shadow-fuchsia-500/30">
              <Piano className="size-6 text-white" />
            </div>
            <div>
              <div className="text-xl font-black tracking-tight">楽譜AI</div>
              <div className="text-xs text-ink-300">Video → Piano Score</div>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <SystemBadge sys={sys} err={sysErr} />
            <div className="flex rounded-full border border-white/10 bg-white/5 p-0.5 text-[11px]" title="採譜を行う場所">
              <button
                type="button"
                onClick={() => switchWhere("browser")}
                className={`rounded-full px-2.5 py-1 ${browser ? "bg-indigo-500/80 text-white" : "text-ink-300 hover:text-ink-100"}`}
              >
                ブラウザで処理
              </button>
              <button
                type="button"
                onClick={() => switchWhere("server")}
                className={`rounded-full px-2.5 py-1 ${!browser ? "bg-indigo-500/80 text-white" : "text-ink-300 hover:text-ink-100"}`}
              >
                PCエンジン
              </button>
            </div>
          </div>
        </header>

        <section className="mt-14 text-center">
          <h1 className="text-4xl font-black leading-tight tracking-tight sm:text-6xl">
            動画のURLから、
            <br className="sm:hidden" />
            <span className="text-gradient">ピアノ譜</span>を。
          </h1>
          <p className="mx-auto mt-5 max-w-2xl text-ink-300">
            YouTube・X・Instagram の動画から音声を取り出し、音源分離と AI 採譜でピアノ譜とギターのタブ譜に。
            初心者向けの簡単アレンジから、雑音だけを除いて一音も削らない超絶技巧の完全再現まで。
          </p>
        </section>

        {!browser && sysErr && !sys ? (
          <div className="mt-10">
            <EngineGate onRetry={connect} checking={checking} />
          </div>
        ) : (
        <form onSubmit={submit} className="glass mx-auto mt-10 max-w-4xl rounded-3xl p-5 sm:p-7">
          <div className="mb-4 flex gap-2">
            <button type="button" onClick={() => setTab("url")} className={`btn btn-ghost ${tab === "url" ? "btn-active" : ""}`}>
              動画URL
            </button>
            <button type="button" onClick={() => setTab("file")} className={`btn btn-ghost ${tab === "file" ? "btn-active" : ""}`}>
              <Upload className="size-4" /> ファイル
            </button>
          </div>

          {tab === "url" && browser ? (
            <TabCapture onRecorded={startRecorded} busy={submitting} />
          ) : tab === "url" ? (
            <div>
              <div className="relative">
                <div className="pointer-events-none absolute left-3 top-1/2 -translate-y-1/2">
                  <PlatformIcon platform={platform ?? undefined} className="size-6" />
                </div>
                <input
                  className="field w-full !py-3.5 !pl-12 text-base"
                  placeholder="https://www.youtube.com/watch?v=…  /  https://x.com/…/status/…  /  https://www.instagram.com/reel/…"
                  value={url}
                  onChange={(e) => setUrl(e.target.value)}
                  inputMode="url"
                  autoComplete="off"
                  spellCheck={false}
                />
              </div>
              <div className="mt-2 flex min-h-6 flex-wrap items-center gap-x-4 gap-y-1 text-xs text-ink-300">
                {url && !platform && <span className="text-amber-300">YouTube・X (Twitter)・Instagram の URL に対応しています</span>}
                {platform && <span>{PLATFORM_LABEL[platform]} を検出</span>}
                {probing && (
                  <span className="flex items-center gap-1">
                    <LoaderCircle className="size-3 animate-spin" />
                    動画情報を確認中…
                  </span>
                )}
                {probeErr && <span className="text-rose-300">{probeErr}</span>}
              </div>
              {probe && (
                <div className="mt-3 flex items-center gap-3 rounded-xl border border-white/8 bg-black/20 p-2.5">
                  {probe.thumbnail ? (
                    // eslint-disable-next-line @next/next/no-img-element
                    <img src={probe.thumbnail} alt="" className="h-14 w-24 rounded-lg object-cover" referrerPolicy="no-referrer" />
                  ) : (
                    <div className="grid h-14 w-24 place-items-center rounded-lg bg-white/5">
                      <PlatformIcon platform={probe.platform} />
                    </div>
                  )}
                  <div className="min-w-0">
                    <div className="truncate font-semibold">{probe.title}</div>
                    <div className="text-xs text-ink-300">
                      {probe.uploader} ・ {formatSec(probe.duration)}
                      {sys && probe.duration > sys.limits.maxAnalyzeSec && !end && (
                        <span className="ml-2 text-amber-300">長いので先頭 {Math.round(sys.limits.maxAnalyzeSec / 60)} 分を解析します（区間指定も可）</span>
                      )}
                    </div>
                  </div>
                </div>
              )}
            </div>
          ) : (
            <div
              onDragOver={(e) => {
                e.preventDefault();
                setDrag(true);
              }}
              onDragLeave={() => setDrag(false)}
              onDrop={(e) => {
                e.preventDefault();
                setDrag(false);
                const f = e.dataTransfer.files?.[0];
                if (f) setFile(f);
              }}
              onClick={() => fileInput.current?.click()}
              className={`grid cursor-pointer place-items-center rounded-2xl border-2 border-dashed px-4 py-10 text-center transition ${
                drag ? "border-indigo-400 bg-indigo-500/10" : "border-white/12 hover:bg-white/[0.03]"
              }`}
            >
              <Upload className="size-8 text-ink-300" />
              <div className="mt-2 font-semibold">{file ? file.name : "音声・動画ファイルをドロップ"}</div>
              <div className="mt-1 text-xs text-ink-300">
                mp3 / wav / m4a / flac / ogg / mp4 / mov / webm ・ {sys?.limits.maxUploadMB ?? 300}MB まで
              </div>
              <input
                ref={fileInput}
                type="file"
                className="hidden"
                accept="audio/*,video/*,.mp3,.wav,.m4a,.flac,.ogg,.opus,.webm,.mp4,.mov,.mkv,.aac"
                onChange={(e) => setFile(e.target.files?.[0] ?? null)}
              />
            </div>
          )}

          <div className="mt-6">
            <div className="label mb-2">難易度</div>
            <ModeCards value={mode} onChange={setMode} />
          </div>

          <button type="button" onClick={() => setAdvanced((v) => !v)} className="mt-5 flex items-center gap-1.5 text-sm text-ink-300 hover:text-ink-100">
            <SlidersHorizontal className="size-4" />
            詳細設定
            <ChevronDown className={`size-4 transition ${advanced ? "rotate-180" : ""}`} />
          </button>
          {advanced && (
            <div className="mt-3 grid gap-4 rounded-2xl border border-white/8 bg-black/15 p-4 sm:grid-cols-2 lg:grid-cols-3">
              <div>
                <div className="label mb-1.5">音源の種類</div>
                <select className="field w-full" value={sourceType} onChange={(e) => setSourceType(e.target.value)}>
                  {SOURCE_TYPES.map((s) => (
                    <option key={s.id} value={s.id}>
                      {s.label} — {s.hint}
                    </option>
                  ))}
                </select>
              </div>
              <div>
                <div className="label mb-1.5">音源分離 (Stem)</div>
                <select className="field w-full" value={separation} onChange={(e) => setSeparation(e.target.value)}>
                  {SEPARATIONS.map((s) => (
                    <option key={s.id} value={s.id}>
                      {s.label}
                    </option>
                  ))}
                </select>
              </div>
              <div>
                <div className="label mb-1.5">解析する区間</div>
                <div className="flex items-center gap-2">
                  <input className="field w-full" placeholder="開始 0:00" value={start} onChange={(e) => setStart(e.target.value)} />
                  <span className="text-ink-400">〜</span>
                  <input className="field w-full" placeholder="終了 (最後まで)" value={end} onChange={(e) => setEnd(e.target.value)} />
                </div>
              </div>
              <div>
                <div className="label mb-1.5">拍子</div>
                <select className="field w-full" value={timeSig} onChange={(e) => setTimeSig(e.target.value)}>
                  <option value="auto">自動判定</option>
                  <option value="4/4">4/4</option>
                  <option value="3/4">3/4</option>
                  <option value="2/4">2/4</option>
                </select>
              </div>
              <div>
                <div className="label mb-1.5">左手の伴奏型</div>
                <select className="field w-full" value={leftHand} onChange={(e) => setLeftHand(e.target.value)}>
                  {LEFT_HANDS.map((s) => (
                    <option key={s.id} value={s.id}>
                      {s.label}
                    </option>
                  ))}
                </select>
              </div>
              <label className="flex items-center gap-2 self-end pb-2 text-sm">
                <input type="checkbox" checked={easyKey} onChange={(e) => setEasyKey(e.target.checked)} className="size-4 accent-indigo-500" />
                ハ長調 / イ短調に移調（黒鍵を減らす）
              </label>
            </div>
          )}

          {browser && sys?.browser && !sys.gpu && (
            <div className="mt-4 flex items-start gap-2 rounded-xl border border-amber-400/30 bg-amber-500/10 px-3 py-2 text-xs text-amber-100">
              <TriangleAlert className="mt-0.5 size-4 shrink-0" />
              <span>
                このブラウザでは GPU（WebGPU）が使えないため、CPU で処理します。1分の曲で5分ほどかかり、音源分離は省略します。
                パソコンの Chrome / Edge（最新版）で開くと GPU で数十倍速く処理できます。
              </span>
            </div>
          )}

          {error && (
            <div className="mt-4 flex items-start gap-2 rounded-xl border border-rose-400/30 bg-rose-500/10 px-3 py-2 text-sm text-rose-200">
              <TriangleAlert className="mt-0.5 size-4 shrink-0" />
              {error}
            </div>
          )}

          <div className="mt-6 flex flex-col items-center gap-2 sm:flex-row sm:justify-between">
            <p className="text-xs text-ink-400">
              {browser
                ? "音源分離・採譜はすべてこのブラウザの中（あなたのPCのGPU）で行われ、音声はどこにも送られません。初回だけAIモデル（約190MB）を読み込みます。著作物は個人の練習の範囲でご利用ください。"
                : "動画の取得・音源分離・採譜はすべてあなたのPCの中で行われ、外には送られません。著作物は個人の練習の範囲でご利用ください。"}
            </p>
            {!(tab === "url" && browser) ? (
              <button type="submit" disabled={!canSubmit || submitting} className="btn btn-primary w-full !px-6 !py-3 text-base sm:w-auto">
                {submitting ? <LoaderCircle className="size-5 animate-spin" /> : <Wand2 className="size-5" />}
                {MODE_INFO[mode].label}モードで楽譜にする
              </button>
            ) : (
              submitting && (
                <span className="flex items-center gap-2 text-sm text-ink-200">
                  <LoaderCircle className="size-4 animate-spin" /> 録音を読み込んでいます…
                </span>
              )
            )}
          </div>
        </form>
        )}

        <RecentJobs jobs={jobs} onDelete={async (id) => {
          await api.remove(id);
          setJobs((js) => js.filter((j) => j.id !== id));
        }} />
      </div>
    </main>
  );
}

function SystemBadge({ sys, err }: { sys: SystemInfo | null; err: string | null }) {
  if (err)
    return (
      <div className="flex items-center gap-2 rounded-full border border-rose-400/30 bg-rose-500/10 px-3 py-1.5 text-xs text-rose-200">
        <TriangleAlert className="size-3.5" /> エンジン未接続
      </div>
    );
  if (!sys) return <div className="h-7 w-40 animate-pulse rounded-full bg-white/5" />;
  if (sys.browser)
    return (
      <div
        className="hidden items-center gap-2 rounded-full border border-white/10 bg-white/5 px-3 py-1.5 text-xs text-ink-200 sm:flex"
        title={sys.gpu ? "WebGPU で、このPCのGPUを使って処理します" : "WebGPU が使えないので CPU で処理します（時間がかかります）"}
      >
        <Cpu className="size-3.5" />
        {sys.gpu ? <span className="max-w-48 truncate">{sys.gpu}</span> : <span className="text-amber-300">CPU（GPU 不可）</span>}
        <span className="text-ink-500">|</span>
        インストール不要
      </div>
    );
  const bd = sys.engines.bytedance.available;
  return (
    <div className="hidden items-center gap-2 rounded-full border border-white/10 bg-white/5 px-3 py-1.5 text-xs text-ink-200 sm:flex" title={`yt-dlp ${sys.ytdlp ?? "-"}`}>
      <Cpu className="size-3.5" />
      {sys.gpu ? sys.gpu : "CPU"}
      <span className="text-ink-500">|</span>
      {bd ? "ピアノ採譜AI" : "内蔵採譜"}
      <span className="text-ink-500">|</span>
      {sys.engines.demucs ? "Demucs 6-stem" : "HPSS"}
    </div>
  );
}

const STATUS_LABEL: Record<string, string> = {
  queued: "待機中",
  running: "処理中",
  done: "完成",
  error: "エラー",
  canceled: "中止",
};

function RecentJobs({ jobs, onDelete }: { jobs: JobSummary[]; onDelete: (id: string) => void }) {
  const router = useRouter();
  if (!jobs.length) return null;
  return (
    <section className="mx-auto mt-12 max-w-4xl">
      <h2 className="mb-3 flex items-center gap-2 text-sm font-bold text-ink-300">
        <Clock className="size-4" /> 最近の採譜
      </h2>
      <div className="grid gap-2">
        {jobs.slice(0, 12).map((j) => (
          <div
            key={j.id}
            role="button"
            tabIndex={0}
            onClick={() => router.push(`/job/?id=${j.id}`)}
            onKeyDown={(e) => e.key === "Enter" && router.push(`/job/?id=${j.id}`)}
            className="glass flex cursor-pointer items-center gap-3 rounded-xl px-3 py-2.5 transition hover:bg-white/[0.06]"
          >
            <PlatformIcon platform={j.platform} />
            <div className="min-w-0 flex-1">
              <div className="truncate text-sm font-semibold">{j.title}</div>
              <div className="text-xs text-ink-400">
                {new Date(j.created * 1000).toLocaleString("ja-JP")} ・ {MODE_INFO[j.mode]?.label ?? j.mode}
                {j.error && <span className="ml-2 text-rose-300">{j.error.slice(0, 60)}</span>}
              </div>
            </div>
            <span
              className={`rounded-full px-2 py-0.5 text-xs font-semibold ${
                j.status === "done"
                  ? "bg-emerald-500/15 text-emerald-300"
                  : j.status === "error"
                    ? "bg-rose-500/15 text-rose-300"
                    : j.status === "running" || j.status === "queued"
                      ? "bg-indigo-500/15 text-indigo-200"
                      : "bg-white/5 text-ink-300"
              }`}
            >
              {STATUS_LABEL[j.status]}
              {j.status === "running" && ` ${Math.round(j.progress * 100)}%`}
            </span>
            <button
              type="button"
              className="rounded-lg p-1.5 text-ink-400 hover:bg-white/10 hover:text-rose-300"
              title="削除"
              onClick={(e) => {
                e.stopPropagation();
                if (confirm("この採譜結果を削除しますか？（元に戻せません）")) onDelete(j.id);
              }}
            >
              <Trash2 className="size-4" />
            </button>
          </div>
        ))}
      </div>
    </section>
  );
}
