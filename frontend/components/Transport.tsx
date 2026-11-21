"use client";

import { Pause, Play, Repeat, SkipBack, Square, Timer } from "lucide-react";

export type PlaySource = "score" | "original" | "target";

export interface TransportProps {
  playing: boolean;
  onToggle: () => void;
  onStop: () => void;
  onRewind: () => void;
  position: string;
  rate: number;
  setRate: (r: number) => void;
  tempoText: string;
  loopOn: boolean;
  setLoopOn: (v: boolean) => void;
  loopFrom: number;
  loopTo: number;
  setLoopRange: (from: number, to: number) => void;
  measures: number;
  metronome: boolean;
  setMetronome: (v: boolean) => void;
  muteR: boolean;
  muteL: boolean;
  setMute: (hand: "R" | "L", v: boolean) => void;
  source: PlaySource;
  setSource: (s: PlaySource) => void;
  targetLabel: string;
  loadingSound: boolean;
  guitar?: boolean;
  guitarTone?: "nylon" | "steel";
  setGuitarTone?: (t: "nylon" | "steel") => void;
}

const RATES = [0.25, 0.5, 0.6, 0.75, 0.9, 1, 1.25, 1.5];

export function Transport(p: TransportProps) {
  const pct = Math.round(p.rate * 100);
  return (
    <div className="glass flex flex-wrap items-center gap-x-4 gap-y-2 rounded-2xl px-3 py-2.5">
      <div className="flex items-center gap-1.5">
        <button className="btn btn-ghost !p-2" onClick={p.onRewind} title="先頭へ (Home)">
          <SkipBack className="size-4" />
        </button>
        <button
          className="grid size-11 place-items-center rounded-full bg-gradient-to-br from-indigo-500 to-fuchsia-500 text-white shadow-lg shadow-fuchsia-500/30 transition hover:brightness-110"
          onClick={p.onToggle}
          title="再生 / 一時停止 (Space)"
        >
          {p.loadingSound ? (
            <span className="size-4 animate-spin rounded-full border-2 border-white/40 border-t-white" />
          ) : p.playing ? (
            <Pause className="size-5" />
          ) : (
            <Play className="ml-0.5 size-5" />
          )}
        </button>
        <button className="btn btn-ghost !p-2" onClick={p.onStop} title="停止">
          <Square className="size-3.5" />
        </button>
      </div>

      <div className="min-w-[7.5rem] font-mono text-sm tabular-nums text-ink-100">{p.position}</div>

      <div className="flex items-center gap-2">
        <span className="text-xs text-ink-300">テンポ</span>
        <input
          type="range"
          className="range w-28"
          min={25}
          max={150}
          step={5}
          value={pct}
          style={{ ["--fill" as string]: `${((pct - 25) / 125) * 100}%` }}
          onChange={(e) => p.setRate(Number(e.target.value) / 100)}
        />
        <select className="field !py-1 !text-xs" value={RATES.includes(p.rate) ? p.rate : ""} onChange={(e) => p.setRate(Number(e.target.value))}>
          {!RATES.includes(p.rate) && <option value="">{pct}%</option>}
          {RATES.map((r) => (
            <option key={r} value={r}>
              {Math.round(r * 100)}%
            </option>
          ))}
        </select>
        <span className="w-16 text-xs tabular-nums text-ink-300">{p.tempoText}</span>
      </div>

      <div className="flex items-center gap-1.5">
        <button className={`btn btn-ghost !px-2 !py-1.5 ${p.loopOn ? "btn-active" : ""}`} onClick={() => p.setLoopOn(!p.loopOn)} title="小節ループ (L)">
          <Repeat className="size-4" /> ループ
        </button>
        <input
          type="number"
          min={1}
          max={p.measures}
          value={p.loopFrom}
          onChange={(e) => p.setLoopRange(Math.max(1, Number(e.target.value) || 1), Math.max(Number(e.target.value) || 1, p.loopTo))}
          className="field w-16 !py-1 text-center !text-xs"
          title="ループ開始小節"
        />
        <span className="text-xs text-ink-400">〜</span>
        <input
          type="number"
          min={1}
          max={p.measures}
          value={p.loopTo}
          onChange={(e) => p.setLoopRange(Math.min(p.loopFrom, Number(e.target.value) || 1), Math.max(1, Number(e.target.value) || 1))}
          className="field w-16 !py-1 text-center !text-xs"
          title="ループ終了小節"
        />
        <span className="text-xs text-ink-400">小節</span>
      </div>

      <button className={`btn btn-ghost !px-2 !py-1.5 ${p.metronome ? "btn-active" : ""}`} onClick={() => p.setMetronome(!p.metronome)} title="メトロノーム (M)">
        <Timer className="size-4" /> クリック
      </button>

      <div className="flex items-center gap-1">
        <button
          className={`btn btn-ghost !px-2 !py-1.5 ${p.muteR ? "line-through opacity-50" : ""}`}
          onClick={() => p.setMute("R", !p.muteR)}
          disabled={p.source !== "score"}
          title={p.guitar ? "メロディを消す（伴奏だけ練習）" : "右手を消す（左手だけ練習）"}
        >
          <span className="size-2 rounded-full bg-rh" /> {p.guitar ? "メロディ" : "右手"}
        </button>
        <button
          className={`btn btn-ghost !px-2 !py-1.5 ${p.muteL ? "line-through opacity-50" : ""}`}
          onClick={() => p.setMute("L", !p.muteL)}
          disabled={p.source !== "score"}
          title={p.guitar ? "伴奏を消す（メロディだけ練習）" : "左手を消す（右手だけ練習）"}
        >
          <span className="size-2 rounded-full bg-lh" /> {p.guitar ? "伴奏" : "左手"}
        </button>
      </div>

      <div className="ml-auto flex items-center gap-1.5">
        {p.guitar && p.setGuitarTone && (
          <select
            className="field !py-1 !text-xs"
            value={p.guitarTone}
            onChange={(e) => p.setGuitarTone?.(e.target.value as "nylon" | "steel")}
            title="ギターの音色（本物のギターの録音）"
          >
            <option value="nylon">ナイロン弦</option>
            <option value="steel">スチール弦</option>
          </select>
        )}
        <span className="text-xs text-ink-300">音源</span>
        <select className="field !py-1 !text-xs" value={p.source} onChange={(e) => p.setSource(e.target.value as PlaySource)}>
          <option value="score">{p.guitar ? "楽譜をギターで再生" : "楽譜をピアノで再生"}</option>
          <option value="original">原曲（元の音声）</option>
          <option value="target">採譜に使った音（{p.targetLabel}）</option>
        </select>
      </div>
    </div>
  );
}
