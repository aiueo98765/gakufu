"use client";

import { Check, CircleX, LoaderCircle, Minus, RotateCcw, Square } from "lucide-react";
import Link from "next/link";
import type { Job } from "@/lib/types";
import { engineMode } from "@/lib/api";
import { formatSec } from "@/lib/music";
import { MODE_INFO } from "./ModeCards";
import { PlatformIcon, PLATFORM_LABEL } from "./PlatformIcon";

export function JobProgress({ job, onCancel }: { job: Job; onCancel: () => void }) {
  const pct = Math.round(job.progress * 100);
  const active = job.status === "queued" || job.status === "running";
  const title = job.meta.title || job.request.url || job.request.filename;
  return (
    <div className="glass rounded-3xl p-6">
      <div className="flex items-start gap-4">
        {job.meta.thumbnail ? (
          // eslint-disable-next-line @next/next/no-img-element
          <img src={job.meta.thumbnail} alt="" referrerPolicy="no-referrer" className="h-20 w-32 shrink-0 rounded-xl object-cover" />
        ) : (
          <div className="grid h-20 w-32 shrink-0 place-items-center rounded-xl bg-white/5">
            <PlatformIcon platform={job.request.platform} className="size-8" />
          </div>
        )}
        <div className="min-w-0 flex-1">
          <div className="flex items-center gap-2 text-xs text-ink-300">
            <PlatformIcon platform={job.request.platform} className="size-4" />
            {PLATFORM_LABEL[job.request.platform] ?? job.request.platform}
            <span>・</span>
            <span className="font-semibold">{MODE_INFO[job.request.mode].label}モード</span>
            {job.meta.duration ? <span>・ {formatSec(job.meta.duration)}</span> : null}
          </div>
          <h1 className="mt-1 line-clamp-2 text-lg font-bold">{title}</h1>
          {job.meta.uploader && <div className="text-sm text-ink-300">{job.meta.uploader}</div>}
        </div>
      </div>

      <div className="mt-6">
        <div className="flex items-end justify-between">
          <div className="text-sm text-ink-200">{job.status === "queued" ? "順番待ち…" : job.message}</div>
          <div className="font-mono text-2xl font-black tabular-nums">{pct}%</div>
        </div>
        <div className="mt-2 h-2.5 overflow-hidden rounded-full bg-white/8">
          <div
            className={`h-full rounded-full bg-gradient-to-r from-indigo-500 via-fuchsia-500 to-sky-400 transition-[width] duration-500 ${active ? "progress-shimmer" : ""}`}
            style={{ width: `${Math.max(2, pct)}%` }}
          />
        </div>
      </div>

      <ol className="mt-6 space-y-2.5">
        {job.stepOrder.map((k) => {
          const st = job.steps[k];
          return (
            <li key={k} className="flex items-start gap-3">
              <StepIcon state={st.state} />
              <div className="min-w-0">
                <div className={`text-sm font-semibold ${st.state === "pending" ? "text-ink-400" : ""}`}>{st.label}</div>
                {st.detail && <div className="break-words text-xs text-ink-300">{st.detail}</div>}
              </div>
            </li>
          );
        })}
      </ol>

      {job.status === "error" && (
        <div className="mt-6 rounded-xl border border-rose-400/30 bg-rose-500/10 p-3 text-sm text-rose-100">
          {job.error}
          <div className="mt-3">
            <Link href="/" className="btn btn-ghost">
              <RotateCcw className="size-4" /> 設定を変えてやり直す
            </Link>
          </div>
        </div>
      )}
      {job.status === "canceled" && (
        <div className="mt-6 text-sm text-ink-300">
          キャンセルしました。<Link href="/" className="text-indigo-300 underline">トップへ戻る</Link>
        </div>
      )}
      {active && (
        <div className="mt-6 flex items-center justify-between gap-3">
          {engineMode() === "browser" ? (
            <p className="text-[11px] leading-relaxed text-ink-400">
              このタブの中で（あなたのPCの GPU を使って）処理しています。終わるまでこのタブを閉じたり再読み込みしたりしないでください。
              ほかのタブを見ていても大丈夫です。
            </p>
          ) : (
            <span />
          )}
          <button onClick={onCancel} className="btn btn-ghost shrink-0">
            <Square className="size-3.5" /> 中止
          </button>
        </div>
      )}
    </div>
  );
}

function StepIcon({ state }: { state: string }) {
  const base = "mt-0.5 grid size-5 shrink-0 place-items-center rounded-full";
  if (state === "done") return <span className={`${base} bg-emerald-500/20 text-emerald-300`}><Check className="size-3.5" /></span>;
  if (state === "running") return <span className={`${base} bg-indigo-500/20 text-indigo-200`}><LoaderCircle className="size-3.5 animate-spin" /></span>;
  if (state === "error") return <span className={`${base} bg-rose-500/20 text-rose-300`}><CircleX className="size-3.5" /></span>;
  if (state === "skipped") return <span className={`${base} bg-white/5 text-ink-400`}><Minus className="size-3.5" /></span>;
  return <span className={`${base} border border-white/12`} />;
}
