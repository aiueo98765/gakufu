"use client";

import { useEffect, useState } from "react";
import { LoaderCircle, RotateCcw, Wand2, X } from "lucide-react";
import type { ArrangeOptions, Instrument, Mode, Score } from "@/lib/types";
import { tempoLabel } from "@/lib/music";
import { MODE_INFO } from "./ModeCards";

const LEFT = [
  ["auto", "おまかせ"],
  ["original", "原曲どおり"],
  ["root", "ルート音のみ"],
  ["fifth", "ルート＋5度"],
  ["triad", "3和音"],
  ["block", "ベース＋和音"],
  ["broken", "分散和音（8分）"],
  ["alberti", "アルベルティ・バス"],
] as const;

export function ArrangePanel({
  open,
  onClose,
  score,
  mode,
  busy,
  onApply,
  onReset,
  instrument = "piano",
}: {
  open: boolean;
  onClose: () => void;
  score: Score;
  mode: Mode;
  busy: boolean;
  onApply: (o: ArrangeOptions, g?: { tuning?: string; capo?: number | "auto" }) => void;
  onReset: () => void;
  instrument?: Instrument;
}) {
  const [o, setO] = useState<ArrangeOptions>({});
  const [g, setG] = useState<{ tuning: string; capo: number | "auto" }>({ tuning: "standard", capo: "auto" });
  useEffect(() => {
    if (open && score.guitar) setG({ tuning: score.guitar.options?.tuning ?? score.guitar.tuning, capo: score.guitar.options?.capo ?? "auto" });
  }, [open, score.guitar]);
  const guitar = instrument === "guitar";
  useEffect(() => {
    if (open) setO({ tempoScale: "auto", bpm: null, timeSig: "auto", transpose: 0, easyKey: false, leftHand: "auto", ...(score.arrange || {}) });
  }, [open, score.arrange]);
  if (!open) return null;
  const set = (k: keyof ArrangeOptions, v: unknown) => setO((p) => ({ ...p, [k]: v }));
  return (
    <div className="fixed inset-0 z-50 grid place-items-center bg-black/60 p-4" onClick={onClose}>
      <div className="glass w-full max-w-lg rounded-3xl p-6" onClick={(e) => e.stopPropagation()}>
        <div className="flex items-center justify-between">
          <h2 className="text-lg font-bold">編曲の設定（{MODE_INFO[mode].label}）</h2>
          <button className="rounded-lg p-1 hover:bg-white/10" onClick={onClose}>
            <X className="size-5" />
          </button>
        </div>
        <p className="mt-1 text-xs text-ink-300">採譜した生データから作り直します（数秒）。テンポが倍・半分に取れてしまったときもここで直せます。</p>

        <div className="mt-5 grid gap-4 sm:grid-cols-2">
          <div className="sm:col-span-2">
            <div className="label mb-1.5">拍の取り方（いま {tempoLabel(score)}）</div>
            <div className="grid grid-cols-4 gap-1.5">
              {(
                [
                  ["auto", "自動"],
                  [0.5, "÷2"],
                  [1, "検出どおり"],
                  [2, "×2"],
                ] as const
              ).map(([v, l]) => (
                <button
                  key={String(v)}
                  className={`btn btn-ghost !px-1 ${String(o.tempoScale ?? "auto") === String(v) && !o.bpm ? "btn-active" : ""}`}
                  onClick={() => setO((p) => ({ ...p, tempoScale: v, bpm: null }))}
                >
                  {l}
                </button>
              ))}
            </div>
            {score.arrange?.autoTempo === "x2" && (
              <div className="mt-1 text-[11px] text-ink-400">音が細かすぎたので、自動で拍を ×2 に補正しています（「検出どおり」で元に戻せます）</div>
            )}
          </div>
          <div>
            <div className="label mb-1.5">テンポを固定（拍/分・複合拍子は付点4分）</div>
            <input
              className="field w-full"
              type="number"
              min={30}
              max={300}
              placeholder="自動（揺れに追従）"
              value={o.bpm ?? ""}
              onChange={(e) => set("bpm", e.target.value ? Number(e.target.value) : null)}
            />
          </div>
          <div>
            <div className="label mb-1.5">拍子</div>
            <select className="field w-full" value={o.timeSig ?? "auto"} onChange={(e) => set("timeSig", e.target.value)}>
              <option value="auto">自動判定</option>
              <option value="4/4">4/4</option>
              <option value="3/4">3/4</option>
              <option value="2/4">2/4</option>
            </select>
          </div>
          <div>
            <div className="label mb-1.5">移調（半音） {o.transpose ? (o.transpose > 0 ? `+${o.transpose}` : o.transpose) : "±0"}</div>
            <input
              type="range"
              className="range w-full"
              min={-6}
              max={6}
              value={o.transpose ?? 0}
              style={{ ["--fill" as string]: `${(((o.transpose ?? 0) + 6) / 12) * 100}%` }}
              onChange={(e) => set("transpose", Number(e.target.value))}
            />
          </div>
          <div>
            <div className="label mb-1.5">左手の伴奏型</div>
            <select className="field w-full" value={o.leftHand ?? "auto"} onChange={(e) => set("leftHand", e.target.value)}>
              {LEFT.map(([v, l]) => (
                <option key={v} value={v}>
                  {l}
                </option>
              ))}
            </select>
            {mode === "virtuoso" && <div className="mt-1 text-[11px] text-ink-400">超プロは常に原曲どおり</div>}
          </div>
          <label className="flex items-center gap-2 text-sm sm:col-span-2">
            <input type="checkbox" className="size-4 accent-indigo-500" checked={!!o.easyKey} onChange={(e) => set("easyKey", e.target.checked)} />
            ハ長調 / イ短調に移調する（黒鍵を減らす）
          </label>
        </div>

        {guitar && (
          <div className="mt-5 grid gap-4 rounded-2xl border border-white/8 bg-black/15 p-4 sm:grid-cols-2">
            <div>
              <div className="label mb-1.5">調弦</div>
              <select className="field w-full" value={g.tuning} onChange={(e) => setG((p) => ({ ...p, tuning: e.target.value }))}>
                <option value="standard">標準 (EADGBE)</option>
                <option value="halfdown">半音下げ (E♭)</option>
                <option value="dropd">ドロップD (DADGBE)</option>
                <option value="dadgad">DADGAD</option>
                <option value="openg">オープンG (DGDGBD)</option>
              </select>
            </div>
            <div>
              <div className="label mb-1.5">カポ</div>
              <select
                className="field w-full"
                value={String(g.capo)}
                onChange={(e) => setG((p) => ({ ...p, capo: e.target.value === "auto" ? "auto" : Number(e.target.value) }))}
              >
                <option value="auto">自動（開放弦を活かせる位置）</option>
                {[0, 1, 2, 3, 4, 5, 6, 7].map((c) => (
                  <option key={c} value={c}>
                    {c === 0 ? "なし" : `${c}フレット`}
                  </option>
                ))}
              </select>
              {score.guitar && <div className="mt-1 text-[11px] text-ink-400">いまはカポ {score.guitar.capo || "なし"}</div>}
            </div>
          </div>
        )}

        {score.edited && (
          <div className="mt-4 rounded-xl border border-amber-400/30 bg-amber-500/10 px-3 py-2 text-xs text-amber-100">
            このモードの楽譜は手で編集されています。作り直すと編集内容は消えます。
          </div>
        )}

        <div className="mt-6 flex flex-wrap justify-between gap-2">
          <button className="btn btn-ghost" onClick={onReset} disabled={busy || !score.edited} title="手で直した内容を捨てて、最後に作った状態に戻す">
            <RotateCcw className="size-4" /> 編集を破棄
          </button>
          <div className="flex gap-2">
            <button className="btn btn-ghost" onClick={onClose}>
              閉じる
            </button>
            <button className="btn btn-primary" disabled={busy} onClick={() => onApply(o, guitar ? g : undefined)}>
              {busy ? <LoaderCircle className="size-4 animate-spin" /> : <Wand2 className="size-4" />}
              この設定で作り直す
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
