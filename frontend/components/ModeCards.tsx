"use client";

import { Flame, Music2, Sparkles } from "lucide-react";
import type { Mode } from "@/lib/types";

export const MODE_INFO: Record<Mode, { label: string; en: string; color: string; ring: string; desc: string; points: string[]; icon: typeof Music2 }> = {
  easy: {
    label: "簡単",
    en: "EASY",
    color: "from-emerald-400/25 to-emerald-500/5",
    ring: "ring-emerald-400/70",
    desc: "片手ずつ弾ける入門アレンジ",
    points: ["右手はメロディだけ（8分音符まで）", "左手はコードのルート / 5度 / 3和音", "弾きやすい音域へ移動・ハ長調への移調も"],
    icon: Music2,
  },
  standard: {
    label: "標準",
    en: "STANDARD",
    color: "from-sky-400/25 to-sky-500/5",
    ring: "ring-sky-400/70",
    desc: "原曲の雰囲気そのまま、中級者向けに最適化",
    points: ["和音は片手3音・1オクターブ以内", "トリルは tr 記号、前打音は省略", "歌ものは分散和音などの伴奏型に"],
    icon: Sparkles,
  },
  virtuoso: {
    label: "超プロ",
    en: "VIRTUOSO",
    color: "from-fuchsia-500/30 via-rose-500/10 to-amber-400/5",
    ring: "ring-fuchsia-400/80",
    desc: "原曲完全再現。一音も削らない",
    points: ["音源分離を最高品質（5回アンサンブル）で", "32分・6連符・3連符を拍ごとに判定", "10度超の和音はアルペジオ、ペダル・強弱つき"],
    icon: Flame,
  },
};

export function ModeCards({ value, onChange }: { value: Mode; onChange: (m: Mode) => void }) {
  return (
    <div className="grid gap-3 sm:grid-cols-3">
      {(Object.keys(MODE_INFO) as Mode[]).map((m) => {
        const info = MODE_INFO[m];
        const Icon = info.icon;
        const active = value === m;
        return (
          <button
            key={m}
            type="button"
            onClick={() => onChange(m)}
            className={`group relative overflow-hidden rounded-2xl border p-4 text-left transition ${
              active ? `border-transparent ring-2 ${info.ring} bg-gradient-to-br ${info.color}` : "border-white/8 bg-white/[0.03] hover:bg-white/[0.06]"
            }`}
          >
            <div className="flex items-center justify-between">
              <span className="flex items-center gap-2 text-lg font-bold">
                <Icon className="size-5 opacity-90" />
                {info.label}
              </span>
              <span className={`text-[10px] font-black tracking-[0.2em] ${m === "virtuoso" ? "text-gradient" : "text-ink-300"}`}>{info.en}</span>
            </div>
            <p className="mt-1.5 text-sm text-ink-200">{info.desc}</p>
            <ul className="mt-3 space-y-1 text-xs text-ink-300">
              {info.points.map((p) => (
                <li key={p} className="flex gap-1.5">
                  <span className="mt-1 size-1 shrink-0 rounded-full bg-current opacity-70" />
                  {p}
                </li>
              ))}
            </ul>
          </button>
        );
      })}
    </div>
  );
}
