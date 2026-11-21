"use client";

import { useEffect, useRef, useState } from "react";
import { CircleStop, ExternalLink, LoaderCircle, Mic, RotateCcw, TriangleAlert } from "lucide-react";
import { canCaptureTab, recordTab, type TabRecording } from "@/lib/localEngine";
import { formatSec } from "@/lib/music";
import { PlatformIcon, PLATFORM_LABEL, detectPlatform } from "./PlatformIcon";

/**
 * URL の動画を、別のタブで再生してその音を録音する（ブラウザ版）。
 *
 * サーバーを持たないので、YouTube・X・Instagram から動画を直接ダウンロードすることはできない。
 * 代わりに Chrome / Edge の「タブの共有（音声つき）」で、再生中のタブの音をそのまま受け取る。
 */
export function TabCapture({
  onRecorded,
  busy,
}: {
  onRecorded: (blob: Blob, info: { title: string; platform: string; url: string; seconds: number }) => void;
  busy: boolean;
}) {
  const [url, setUrl] = useState("");
  const [title, setTitle] = useState("");
  const [rec, setRec] = useState<TabRecording | null>(null);
  const [starting, setStarting] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [elapsed, setElapsed] = useState(0);
  const [level, setLevel] = useState(0);
  const [silentFor, setSilentFor] = useState(0);
  const raf = useRef(0);
  const platform = detectPlatform(url);
  // 録音できるかはブラウザで確かめる（静的書き出し時の描画とそろえるため、最初は「できる」として描く）
  const [supported, setSupported] = useState(true);
  useEffect(() => setSupported(canCaptureTab()), []);

  useEffect(() => {
    if (!rec) return;
    let quiet = 0;
    let last = performance.now();
    const tick = () => {
      const now = performance.now();
      const lv = rec.level();
      setElapsed(rec.elapsed());
      setLevel(lv);
      quiet = lv < 0.003 ? quiet + (now - last) / 1000 : 0;
      last = now;
      setSilentFor(quiet);
      raf.current = requestAnimationFrame(tick);
    };
    raf.current = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf.current);
  }, [rec]);

  const defaultTitle = () => {
    const d = new Date();
    const p = platform ? PLATFORM_LABEL[platform] : "動画";
    return `${p} の録音 ${d.getMonth() + 1}/${d.getDate()} ${String(d.getHours()).padStart(2, "0")}:${String(d.getMinutes()).padStart(2, "0")}`;
  };

  async function start() {
    setErr(null);
    setStarting(true);
    try {
      const r = await recordTab();
      r.onEnded = () => void finish(r);
      setRec(r);
    } catch (e) {
      const m = (e as Error).message || "";
      setErr(/Permission|NotAllowed|denied/i.test(m) ? "共有がキャンセルされました。もう一度「録音を始める」を押して、動画のタブを選んでください。" : m);
    } finally {
      setStarting(false);
    }
  }

  async function finish(r = rec) {
    if (!r) return;
    const secs = r.elapsed();
    const blob = await r.stop();
    setRec(null);
    if (secs < 3 || blob.size < 2000) {
      setErr("録音が短すぎました。動画を再生しながら録音してください。");
      return;
    }
    onRecorded(blob, { title: title.trim() || defaultTitle(), platform: platform || "upload", url, seconds: secs });
  }

  function cancel() {
    rec?.cancel();
    setRec(null);
  }

  if (!supported)
    return (
      <div className="rounded-2xl border border-amber-400/30 bg-amber-500/10 p-4 text-sm text-amber-100">
        <div className="flex items-center gap-2 font-bold">
          <TriangleAlert className="size-4" /> このブラウザでは動画の音を録音できません
        </div>
        <p className="mt-1 text-amber-100/80">
          パソコンの Chrome か Edge で開くと、URL の動画の音を取り込めます。または「ファイル」から音声・動画ファイルを選んでください。
        </p>
      </div>
    );

  return (
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
          disabled={!!rec}
        />
      </div>
      <div className="mt-2 min-h-5 text-xs text-ink-300">
        {url && !platform && <span className="text-amber-300">YouTube・X (Twitter)・Instagram の URL に対応しています</span>}
        {platform && <span>{PLATFORM_LABEL[platform]} を検出</span>}
      </div>

      <ol className="mt-3 grid gap-2 text-sm sm:grid-cols-3">
        <li className="rounded-xl border border-white/8 bg-black/20 p-3">
          <div className="text-xs font-bold text-ink-400">1</div>
          <button
            type="button"
            className="btn btn-ghost mt-1 w-full"
            disabled={!platform || !!rec}
            onClick={() => window.open(/^https?:\/\//i.test(url.trim()) ? url.trim() : `https://${url.trim()}`, "_blank", "noopener")}
          >
            <ExternalLink className="size-4" /> 動画を別のタブで開く
          </button>
          <p className="mt-1.5 text-[11px] text-ink-400">広告があれば先に終わらせ、曲の頭で一時停止しておく</p>
        </li>
        <li className="rounded-xl border border-white/8 bg-black/20 p-3">
          <div className="text-xs font-bold text-ink-400">2</div>
          <button type="button" className="btn btn-primary mt-1 w-full" disabled={!!rec || starting || busy} onClick={start}>
            {starting ? <LoaderCircle className="size-4 animate-spin" /> : <Mic className="size-4" />} 録音を始める
          </button>
          <p className="mt-1.5 text-[11px] text-ink-400">
            共有する画面で<b>動画のタブ</b>を選び、<b>「タブの音声も共有する」</b>をオンにして共有
          </p>
        </li>
        <li className="rounded-xl border border-white/8 bg-black/20 p-3">
          <div className="text-xs font-bold text-ink-400">3</div>
          <button type="button" className="btn btn-ghost mt-1 w-full" disabled={!rec} onClick={() => void finish()}>
            <CircleStop className="size-4" /> 止めて楽譜にする
          </button>
          <p className="mt-1.5 text-[11px] text-ink-400">動画を最後まで（または欲しい所まで）再生してから押す</p>
        </li>
      </ol>

      {rec && (
        <div className="mt-3 flex flex-wrap items-center gap-3 rounded-xl border border-rose-400/30 bg-rose-500/10 px-3 py-2 text-sm">
          <span className="flex items-center gap-1.5 font-bold text-rose-200">
            <span className="size-2.5 animate-pulse rounded-full bg-rose-400" /> 録音中 {formatSec(elapsed)}
          </span>
          <div className="h-2 w-40 overflow-hidden rounded-full bg-white/10">
            <div className="h-full bg-emerald-400 transition-[width] duration-75" style={{ width: `${Math.min(100, Math.sqrt(level) * 100)}%` }} />
          </div>
          {silentFor > 4 && <span className="text-xs text-amber-200">音が来ていません。動画のタブで再生してください</span>}
          <button type="button" className="btn btn-ghost ml-auto !py-1 text-xs" onClick={cancel}>
            <RotateCcw className="size-3.5" /> やめる
          </button>
        </div>
      )}

      <div className="mt-3 flex items-center gap-2">
        <span className="text-xs text-ink-400">曲名</span>
        <input className="field flex-1 !py-1.5 text-sm" placeholder="（空欄なら自動）" value={title} onChange={(e) => setTitle(e.target.value)} />
      </div>
      {err && (
        <div className="mt-3 flex items-start gap-2 rounded-xl border border-rose-400/30 bg-rose-500/10 px-3 py-2 text-sm text-rose-200">
          <TriangleAlert className="mt-0.5 size-4 shrink-0" />
          {err}
        </div>
      )}
      <p className="mt-3 text-[11px] leading-relaxed text-ink-400">
        このサイトはサーバーを持たないので、動画を直接ダウンロードしません。別のタブで再生している音を、あなたのブラウザが受け取って録音します（Chrome / Edge のパソコン版）。
        音は外部に送られません。動画プレーヤーの音量は大きめのままにしてください（下げると録音も小さくなります）。
      </p>
    </div>
  );
}
