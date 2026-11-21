"use client";

import { Flame, LoaderCircle, Repeat, Wand2 } from "lucide-react";
import type { Score } from "@/lib/types";

export function DifficultyPanel({
  score,
  onLoop,
  onRecalc,
  stale,
  busy,
  guitar = false,
}: {
  score: Score;
  onLoop: (startMeasure: number, endMeasure: number) => void;
  onRecalc: () => void;
  stale: boolean;
  busy: boolean;
  guitar?: boolean;
}) {
  const d = score.difficulty;
  const pct = Math.min(100, (d.overall / 10) * 100);
  const color = d.overall >= 8 ? "from-fuchsia-500 to-rose-500" : d.overall >= 6 ? "from-amber-400 to-rose-400" : d.overall >= 4 ? "from-sky-400 to-indigo-400" : "from-emerald-400 to-teal-400";
  return (
    <div className="space-y-4">
      <div>
        <div className="flex items-end justify-between">
          <div>
            <div className="label">推定難易度</div>
            <div className="text-sm font-semibold text-ink-200">{d.label}</div>
          </div>
          <div className="font-mono text-3xl font-black tabular-nums">
            {d.overall.toFixed(1)}
            <span className="text-base text-ink-400">/10</span>
          </div>
        </div>
        <div className="mt-2 h-2 overflow-hidden rounded-full bg-white/8">
          <div className={`h-full rounded-full bg-gradient-to-r ${color}`} style={{ width: `${pct}%` }} />
        </div>
      </div>

      <button className={`btn w-full ${stale ? "btn-primary" : "btn-ghost"}`} onClick={onRecalc} disabled={busy}>
        {busy ? <LoaderCircle className="size-4 animate-spin" /> : <Wand2 className="size-4" />}
        {guitar ? "弦・フレット・難所を" : "運指・難所を"}
        {stale ? "再計算（編集あり）" : "再計算"}
      </button>

      <div>
        <div className="label mb-2 flex items-center gap-1">
          <Flame className="size-3.5 text-rose-300" /> 難所 {d.spans.length} か所
        </div>
        {d.spans.length === 0 && <div className="text-xs text-ink-400">とくに難しい箇所は見つかりませんでした。</div>}
        <div className="space-y-2">
          {d.spans.map((s) => (
            <div key={`${s.startMeasure}-${s.endMeasure}`} className="rounded-xl border border-rose-400/15 bg-rose-500/[0.06] p-2.5">
              <div className="flex items-center justify-between gap-2">
                <div className="text-sm font-bold">
                  {s.startMeasure + 1}
                  {s.endMeasure > s.startMeasure && `〜${s.endMeasure + 1}`} 小節
                  <span className="ml-2 text-xs font-normal text-rose-200">{"●".repeat(Math.max(1, Math.round(s.peak * 5)))}</span>
                </div>
                <button className="btn btn-ghost !px-2 !py-1 text-xs" onClick={() => onLoop(s.startMeasure, s.endMeasure)} title="この区間をループ再生">
                  <Repeat className="size-3.5" /> 練習
                </button>
              </div>
              <ul className="mt-1 space-y-0.5 text-[11px] text-ink-300">
                {s.reasons.slice(0, 4).map((r) => (
                  <li key={r}>・{r}</li>
                ))}
              </ul>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
