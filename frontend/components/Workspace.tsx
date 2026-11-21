"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import {
  ArrowLeft,
  ChevronDown,
  CircleCheck,
  Columns2,
  Download,
  ExternalLink,
  FileMusic,
  FileText,
  Guitar,
  Info,
  LoaderCircle,
  Music,
  Piano,
  Redo2,
  Rows3,
  Settings2,
  TriangleAlert,
  Undo2,
  Wand2,
  ZoomIn,
  ZoomOut,
} from "lucide-react";
import { api } from "@/lib/api";
import { AudioPlayer, ScorePlayer, getGuitarTone, setGuitarTone as applyGuitarTone, type Engine, type GuitarTone } from "@/lib/player";
import {
  GRID_OPTIONS,
  bars as barsOf,
  beatToSec,
  formatPos,
  formatSec,
  measureCount,
  measureIndexAt,
  measureStart,
  meterLabel,
  prettyChord,
  tempoLabel,
} from "@/lib/music";
import type { ArrangeOptions, DisplayOptions, Instrument, Job, Mode, Note, Score } from "@/lib/types";
import { refitFrets } from "@/lib/guitar";
import { MODE_INFO } from "./ModeCards";
import { PlatformIcon, PLATFORM_LABEL } from "./PlatformIcon";
import { ScoreView } from "./ScoreView";
import { PianoRoll } from "./PianoRoll";
import { Transport, type PlaySource } from "./Transport";
import { Inspector } from "./Inspector";
import { DifficultyPanel } from "./DifficultyPanel";
import { ArrangePanel } from "./ArrangePanel";

type View = "staff" | "roll" | "split";
type SideTab = "edit" | "hard" | "info";

function downloadBlob(blob: Blob, name: string) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = name;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 2000);
}

function safeName(s: string) {
  return (s || "score").replace(/[\\/:*?"<>|\r\n\t]+/g, " ").trim().slice(0, 80) || "score";
}

export function Workspace({ job }: { job: Job }) {
  const router = useRouter();
  const [mode, setMode] = useState<Mode>(job.result?.defaultMode ?? job.request.mode);
  const [instrument, setInstrument] = useState<Instrument>("piano");
  const [guitarTone, setGuitarToneState] = useState<GuitarTone>(getGuitarTone());
  const [score, setScoreState] = useState<Score | null>(null);
  const scoreRef = useRef<Score | null>(null);
  const setScore = useCallback((s: Score | null) => {
    scoreRef.current = s;
    setScoreState(s);
  }, []);
  const [loadErr, setLoadErr] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const history = useRef<{ past: Score[]; future: Score[] }>({ past: [], future: [] });
  const [, setHistVer] = useState(0);
  const [selection, setSelection] = useState<Set<string>>(new Set());
  const [view, setView] = useState<View>("split");
  const [side, setSide] = useState<SideTab>("hard");
  const [display, setDisplay] = useState<DisplayOptions>({ fingering: "difficult", chords: true, pedal: true, dynamics: true });
  const [zoom, setZoom] = useState(0.8);
  const [grid, setGrid] = useState(0.25);
  const [xml, setXml] = useState<string | null>(null);
  const [saveState, setSaveState] = useState<"saved" | "saving" | "dirty" | "error">("saved");
  const dirtyRef = useRef(false);
  const [stale, setStale] = useState(false);
  const [recalcBusy, setRecalcBusy] = useState(false);
  const [arrangeOpen, setArrangeOpen] = useState(false);
  const [arrangeBusy, setArrangeBusy] = useState(false);
  const [exportOpen, setExportOpen] = useState(false);
  const [exportMsg, setExportMsg] = useState<string | null>(null);
  const [toast, setToast] = useState<string | null>(null);
  const [focusBeat, setFocusBeat] = useState<number | null>(null);
  const [focusMeasure, setFocusMeasure] = useState<number | null>(null);

  // ---- 再生 ----
  const [playing, setPlaying] = useState(false);
  const playingRef = useRef(false);
  const [rate, setRate] = useState(1);
  const [loopOn, setLoopOn] = useState(false);
  const [loopFrom, setLoopFrom] = useState(1);
  const [loopTo, setLoopTo] = useState(4);
  const [metronome, setMetronome] = useState(false);
  const [muteR, setMuteR] = useState(false);
  const [muteL, setMuteL] = useState(false);
  const [source, setSource] = useState<PlaySource>("score");
  const sourceRef = useRef<PlaySource>("score");
  const [loadingSound, setLoadingSound] = useState(false);
  const [posText, setPosText] = useState("1小節 1拍");
  const engines = useRef<Partial<Record<PlaySource, Engine>>>({});
  const cursor = useRef(0);

  const measures = score ? measureCount(score) : 1;
  const mStart = useCallback((m: number) => (scoreRef.current ? measureStart(m, scoreRef.current) : m * 4), []);
  const loopRange = useMemo(
    () => (loopOn && score ? { start: measureStart(loopFrom - 1, score), end: measureStart(loopTo, score) } : null),
    [loopOn, loopFrom, loopTo, score],
  );
  const difficult = useMemo(() => new Set(score?.difficulty?.difficultMeasures ?? []), [score?.difficulty]);

  const flash = useCallback((msg: string) => {
    setToast(msg);
    setTimeout(() => setToast((t) => (t === msg ? null : t)), 2200);
  }, []);

  // ------------------------------------------------------------- 読み込み
  useEffect(() => {
    let cancel = false;
    setLoading(true);
    setLoadErr(null);
    api
      .score(job.id, mode, instrument)
      .then((s) => {
        if (cancel) return;
        setScore(s);
        history.current = { past: [], future: [] };
        setHistVer((v) => v + 1);
        setSelection(new Set());
        setStale(false);
        dirtyRef.current = false;
        setSaveState("saved");
        const ms = measureCount(s);
        setLoopFrom((f) => Math.min(f, ms));
        setLoopTo((t) => Math.min(Math.max(t, 1), ms));
      })
      .catch((e) => !cancel && setLoadErr(e.message))
      .finally(() => !cancel && setLoading(false));
    return () => {
      cancel = true;
    };
  }, [job.id, mode, instrument, setScore]);

  // ------------------------------------------------------------- 編集
  const applyEdit = useCallback(
    (fn: (notes: Note[]) => Note[], label: string) => {
      const prev = scoreRef.current;
      if (!prev) return;
      let notes = fn(prev.notes);
      if (notes === prev.notes) return;
      // ギターは音の高さが変わったら弦・フレットを合わせ直してタブ譜と矛盾させない
      notes = refitFrets(prev, notes);
      history.current.past.push(prev);
      if (history.current.past.length > 300) history.current.past.shift();
      history.current.future = [];
      setScore({ ...prev, notes, edited: true });
      dirtyRef.current = true;
      setSaveState("dirty");
      setStale(true);
      setHistVer((v) => v + 1);
      flash(label);
    },
    [setScore, flash],
  );

  const replaceScore = useCallback(
    (next: Score, label: string, keepHistory = true) => {
      const prev = scoreRef.current;
      if (keepHistory && prev) {
        history.current.past.push(prev);
        history.current.future = [];
      } else {
        history.current = { past: [], future: [] };
      }
      setScore(next);
      setHistVer((v) => v + 1);
      flash(label);
    },
    [setScore, flash],
  );

  const undo = useCallback(() => {
    const h = history.current;
    const prev = h.past.pop();
    if (!prev || !scoreRef.current) return;
    h.future.push(scoreRef.current);
    setScore({ ...prev, edited: true });
    dirtyRef.current = true;
    setSaveState("dirty");
    setStale(true);
    setHistVer((v) => v + 1);
    flash("元に戻しました");
  }, [setScore, flash]);

  const redo = useCallback(() => {
    const h = history.current;
    const next = h.future.pop();
    if (!next || !scoreRef.current) return;
    h.past.push(scoreRef.current);
    setScore({ ...next, edited: true });
    dirtyRef.current = true;
    setSaveState("dirty");
    setHistVer((v) => v + 1);
    flash("やり直しました");
  }, [setScore, flash]);

  const deleteSelected = useCallback(() => {
    if (!selection.size) return;
    const ids = new Set(selection);
    applyEdit((ns) => ns.filter((n) => !ids.has(n.id)), `${ids.size}音を削除`);
    setSelection(new Set());
  }, [selection, applyEdit]);

  // ------------------------------------------------------------- 保存
  const save = useCallback(async () => {
    const s = scoreRef.current;
    if (!s || !dirtyRef.current) return;
    dirtyRef.current = false;
    setSaveState("saving");
    try {
      await api.saveScore(job.id, s.mode, s, s.instrument ?? "piano");
      setSaveState(dirtyRef.current ? "dirty" : "saved");
    } catch {
      dirtyRef.current = true;
      setSaveState("error");
    }
  }, [job.id]);

  useEffect(() => {
    if (!dirtyRef.current) return;
    const t = setTimeout(save, 1500);
    return () => clearTimeout(t);
  }, [score, save]);

  useEffect(() => {
    const warn = (e: BeforeUnloadEvent) => {
      if (dirtyRef.current) {
        void save();
        e.preventDefault();
      }
    };
    window.addEventListener("beforeunload", warn);
    return () => window.removeEventListener("beforeunload", warn);
  }, [save]);

  // ------------------------------------------------------------- 五線譜（MusicXML）
  useEffect(() => {
    if (!score) return;
    const t = setTimeout(() => {
      api
        .musicxml(score, display)
        .then(setXml)
        .catch((e) => flash(`五線譜の生成に失敗: ${e.message}`));
    }, 350);
    return () => clearTimeout(t);
  }, [score, display, flash]);

  // ------------------------------------------------------------- 再生エンジン
  const engine = useCallback(
    (src: PlaySource): Engine => {
      let e = engines.current[src];
      if (!e) {
        e = src === "score" ? new ScorePlayer() : new AudioPlayer(api.audioSource(job.id, src));
        e.onEnd = () => {
          playingRef.current = false;
          setPlaying(false);
          cursor.current = 0;
        };
        if (scoreRef.current) e.setScore(scoreRef.current);
        engines.current[src] = e;
      }
      return e;
    },
    [job.id],
  );

  const getBeat = useCallback(() => {
    if (playingRef.current) {
      const e = engines.current[sourceRef.current];
      if (e) return e.getBeat();
    }
    return cursor.current;
  }, []);

  useEffect(() => {
    if (!score) return;
    const t = setTimeout(() => {
      for (const e of Object.values(engines.current)) e?.setScore(score);
    }, 150);
    return () => clearTimeout(t);
  }, [score]);

  useEffect(() => {
    for (const e of Object.values(engines.current)) {
      e?.setRate(rate);
      e?.setLoop(loopRange);
      e?.setMetronome(metronome);
      e?.setMuted("R", muteR);
      e?.setMuted("L", muteL);
    }
  }, [rate, loopRange, metronome, muteR, muteL]);

  useEffect(() => {
    const t = setInterval(() => {
      const s = scoreRef.current;
      if (!s) return;
      const b = Math.max(0, getBeat());
      const sec = beatToSec(b, s);
      setPosText(`${formatPos(b, s)}${sourceRef.current !== "score" ? `  ${formatSec(sec)}` : ""}`);
    }, 120);
    return () => clearInterval(t);
  }, [getBeat]);

  useEffect(
    () => () => {
      for (const e of Object.values(engines.current)) e?.dispose();
    },
    [],
  );

  const startEngine = useCallback(
    async (src: PlaySource, from: number) => {
      const e = engine(src);
      setLoadingSound(true);
      try {
        await e.ready();
        if (scoreRef.current) e.setScore(scoreRef.current);
        e.setRate(rate);
        e.setLoop(loopRange);
        e.setMetronome(metronome);
        e.setMuted("R", muteR);
        e.setMuted("L", muteL);
        await e.play(from);
        playingRef.current = true;
        setPlaying(true);
      } catch (err) {
        flash(`再生できませんでした: ${(err as Error).message}`);
      } finally {
        setLoadingSound(false);
      }
    },
    [engine, rate, loopRange, metronome, muteR, muteL, flash],
  );

  const togglePlay = useCallback(async () => {
    const src = sourceRef.current;
    if (playingRef.current) {
      const e = engine(src);
      cursor.current = e.getBeat();
      e.pause();
      playingRef.current = false;
      setPlaying(false);
      return;
    }
    await startEngine(src, cursor.current);
  }, [engine, startEngine]);

  const stop = useCallback(() => {
    engine(sourceRef.current).stop();
    playingRef.current = false;
    setPlaying(false);
    cursor.current = loopRange ? loopRange.start : 0;
  }, [engine, loopRange]);

  const seek = useCallback(
    (beat: number) => {
      cursor.current = Math.max(0, beat);
      if (playingRef.current) engine(sourceRef.current).seek(cursor.current);
    },
    [engine],
  );

  const changeSource = useCallback(
    async (src: PlaySource) => {
      const was = playingRef.current;
      const from = getBeat();
      if (was) {
        engine(sourceRef.current).pause();
        playingRef.current = false;
        setPlaying(false);
      }
      sourceRef.current = src;
      setSource(src);
      cursor.current = from;
      if (was) await startEngine(src, from);
    },
    [engine, getBeat, startEngine],
  );

  const setLoopRange = useCallback((from: number, to: number) => {
    setLoopFrom(from);
    setLoopTo(to);
  }, []);

  const practiceSpan = useCallback(
    (a: number, b: number) => {
      setLoopFrom(a + 1);
      setLoopTo(b + 1);
      setLoopOn(true);
      seek(mStart(a));
      setFocusMeasure(a);
      setFocusBeat(mStart(a));
      flash(`${a + 1}〜${b + 1}小節をループ`);
    },
    [mStart, seek, flash],
  );

  const onMeasureClick = useCallback(
    (m: number, shift: boolean) => {
      if (shift) {
        const a = Math.min(loopFrom - 1, m);
        const b = Math.max(loopFrom - 1, m);
        setLoopFrom(a + 1);
        setLoopTo(b + 1);
        setLoopOn(true);
        flash(`${a + 1}〜${b + 1}小節をループ`);
      } else {
        seek(mStart(m));
        setFocusBeat(mStart(m));
      }
    },
    [loopFrom, mStart, seek, flash],
  );

  // ------------------------------------------------------------- モード・編曲
  const changeMode = useCallback(
    async (m: Mode) => {
      if (m === mode) return;
      await save();
      stop();
      setMode(m);
    },
    [mode, save, stop],
  );

  const changeInstrument = useCallback(
    async (v: Instrument) => {
      if (v === instrument) return;
      await save();
      stop();
      setInstrument(v);
    },
    [instrument, save, stop],
  );

  const recalc = useCallback(async () => {
    const s = scoreRef.current;
    if (!s) return;
    setRecalcBusy(true);
    try {
      const next = await api.fingering(s);
      replaceScore({ ...next, edited: s.edited }, "運指と難所を計算し直しました");
      dirtyRef.current = true;
      setSaveState("dirty");
      setStale(false);
    } catch (e) {
      flash((e as Error).message);
    } finally {
      setRecalcBusy(false);
    }
  }, [replaceScore, flash]);

  const rearrange = useCallback(
    async (o: ArrangeOptions, g?: { tuning?: string; capo?: number | "auto" }) => {
      setArrangeBusy(true);
      try {
        stop();
        const next = await api.rearrange(job.id, mode, o, instrument, g ?? {});
        replaceScore(next, "作り直しました", false);
        dirtyRef.current = false;
        setSaveState("saved");
        setStale(false);
        setSelection(new Set());
        setArrangeOpen(false);
      } catch (e) {
        flash((e as Error).message);
      } finally {
        setArrangeBusy(false);
      }
    },
    [job.id, mode, instrument, replaceScore, stop, flash],
  );

  const resetEdits = useCallback(async () => {
    if (!confirm("手で直した内容を捨てて元に戻しますか？")) return;
    setArrangeBusy(true);
    try {
      const next = await api.reset(job.id, mode, instrument);
      replaceScore(next, "編集を破棄しました", false);
      dirtyRef.current = false;
      setSaveState("saved");
      setStale(false);
      setArrangeOpen(false);
    } finally {
      setArrangeBusy(false);
    }
  }, [job.id, mode, instrument, replaceScore]);

  // ------------------------------------------------------------- 書き出し
  const exportPdf = useCallback(async () => {
    const s = scoreRef.current;
    if (!s) return;
    setExportOpen(false);
    setExportMsg("PDF を準備しています…");
    try {
      const x = xml ?? (await api.musicxml(s, display));
      const { exportPdf } = await import("@/lib/pdf");
      await exportPdf(x, `${safeName(s.title)} (${s.modeLabel}).pdf`, setExportMsg);
    } catch (e) {
      flash(`PDF の作成に失敗: ${(e as Error).message}`);
    } finally {
      setExportMsg(null);
    }
  }, [xml, display, flash]);

  const exportFile = useCallback(
    async (kind: "midi" | "musicxml" | "tab") => {
      const s = scoreRef.current;
      if (!s) return;
      setExportOpen(false);
      setExportMsg(kind === "midi" ? "MIDI を書き出しています…" : kind === "tab" ? "TAB譜を書き出しています…" : "MusicXML を書き出しています…");
      try {
        const { blob, name } = await api.exportBlob(kind, s, display);
        downloadBlob(blob, name);
      } catch (e) {
        flash((e as Error).message);
      } finally {
        setExportMsg(null);
      }
    },
    [display, flash],
  );

  // ------------------------------------------------------------- キーボード
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const t = e.target as HTMLElement;
      if (t && (t.tagName === "INPUT" || t.tagName === "SELECT" || t.tagName === "TEXTAREA" || t.isContentEditable)) return;
      const s = scoreRef.current;
      if (!s) return;
      const ctrl = e.ctrlKey || e.metaKey;
      const sel = selection;
      const mapSel = (f: (n: Note) => Note, label: string) => applyEdit((ns) => ns.map((n) => (sel.has(n.id) ? f(n) : n)), label);
      const clampP = (p: number) => Math.min(108, Math.max(21, p));
      if (e.code === "Space") {
        e.preventDefault();
        void togglePlay();
      } else if (ctrl && e.key.toLowerCase() === "z" && !e.shiftKey) {
        e.preventDefault();
        undo();
      } else if (ctrl && (e.key.toLowerCase() === "y" || (e.key.toLowerCase() === "z" && e.shiftKey))) {
        e.preventDefault();
        redo();
      } else if (ctrl && e.key.toLowerCase() === "a") {
        e.preventDefault();
        setSelection(new Set(s.notes.map((n) => n.id)));
      } else if (e.key === "Escape") {
        setSelection(new Set());
      } else if (e.key === "Home") {
        seek(0);
      } else if (!sel.size && (e.key === "ArrowLeft" || e.key === "ArrowRight")) {
        const cur = measureIndexAt(getBeat(), s);
        seek(measureStart(Math.max(0, cur + (e.key === "ArrowRight" ? 1 : -1)), s));
      } else if (e.key.toLowerCase() === "l" && !ctrl) {
        setLoopOn((v) => !v);
      } else if (e.key.toLowerCase() === "m" && !ctrl) {
        setMetronome((v) => !v);
      } else if (sel.size) {
        if (e.key === "Delete" || e.key === "Backspace") {
          e.preventDefault();
          deleteSelected();
        } else if (e.key === "ArrowUp" || e.key === "ArrowDown") {
          e.preventDefault();
          const d = (e.key === "ArrowUp" ? 1 : -1) * (e.shiftKey ? 12 : 1);
          mapSel((n) => ({ ...n, pitch: clampP(n.pitch + d) }), e.shiftKey ? "オクターブ移動" : "半音移動");
        } else if (e.key === "ArrowLeft" || e.key === "ArrowRight") {
          e.preventDefault();
          const d = (e.key === "ArrowRight" ? 1 : -1) * (e.shiftKey ? 1 : grid);
          mapSel((n) => ({ ...n, start: Math.max(0, +(n.start + d).toFixed(6)) }), "位置を移動");
        } else if (e.key === "[" || e.key === "]") {
          const d = e.key === "]" ? grid : -grid;
          mapSel((n) => ({ ...n, dur: Math.max(grid, +(n.dur + d).toFixed(6)) }), "長さを変更");
        } else if (e.key.toLowerCase() === "h" && !ctrl) {
          mapSel((n) => ({ ...n, hand: n.hand === "R" ? "L" : "R" }), "手を入れ替え");
        } else if (/^[1-5]$/.test(e.key) && !ctrl) {
          const f = Number(e.key);
          mapSel((n) => ({ ...n, finger: f, fingerLocked: true }), `指番号 ${f}`);
        } else if (e.key === "0" && !ctrl) {
          mapSel((n) => ({ ...n, fingerLocked: false }), "指番号を自動に");
        }
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [selection, applyEdit, togglePlay, undo, redo, deleteSelected, seek, getBeat, grid]);

  // ------------------------------------------------------------- 表示
  if (loadErr) {
    return (
      <main className="bg-aurora grid min-h-screen place-items-center p-6">
        <div className="glass max-w-md rounded-2xl p-6 text-center">
          <TriangleAlert className="mx-auto size-8 text-rose-300" />
          <p className="mt-3">{loadErr}</p>
          <button className="btn btn-ghost mt-4" onClick={() => router.push("/")}>
            トップへ
          </button>
        </div>
      </main>
    );
  }

  const canUndo = history.current.past.length > 0;
  const canRedo = history.current.future.length > 0;
  const title = score?.title || job.meta.title || "無題";

  return (
    <main className="flex min-h-screen flex-col bg-ink-950 lg:h-screen">
      {/* ヘッダー */}
      <header className="flex flex-wrap items-center gap-x-3 gap-y-2 border-b border-white/6 bg-ink-900/80 px-3 py-2">
        <button className="btn btn-ghost !p-2" onClick={() => router.push("/")} title="トップへ">
          <ArrowLeft className="size-4" />
        </button>
        <PlatformIcon platform={job.request.platform} className="size-5 shrink-0" />
        <div className="min-w-0 flex-1">
          <div className="truncate text-sm font-bold">{title}</div>
          <div className="truncate text-[11px] text-ink-400">
            {job.meta.uploader || PLATFORM_LABEL[job.request.platform]}
            {score && ` ・ ${score.key.ja}（${score.key.en}） ・ ${meterLabel(score)} ・ ${tempoLabel(score)}`}
          </div>
        </div>
        <div className="flex rounded-xl border border-white/8 bg-black/30 p-0.5" title="楽器">
          {(
            [
              ["piano", "ピアノ", <Piano key="p" className="size-3.5" />],
              ["guitar", "ギター TAB", <Guitar key="g" className="size-3.5" />],
            ] as const
          ).map(([v, l, icon]) => (
            <button
              key={v}
              onClick={() => changeInstrument(v)}
              className={`flex items-center gap-1 rounded-lg px-2.5 py-1.5 text-xs font-bold transition ${
                instrument === v ? "bg-white text-ink-950" : "text-ink-300 hover:text-white"
              }`}
            >
              {icon}
              {l}
            </button>
          ))}
        </div>
        <div className="flex rounded-xl border border-white/8 bg-black/30 p-0.5">
          {(Object.keys(MODE_INFO) as Mode[]).map((m) => (
            <button
              key={m}
              onClick={() => changeMode(m)}
              className={`rounded-lg px-3 py-1.5 text-xs font-bold transition ${
                mode === m
                  ? m === "virtuoso"
                    ? "bg-gradient-to-r from-fuchsia-500 to-rose-500 text-white"
                    : m === "easy"
                      ? "bg-emerald-500 text-white"
                      : "bg-sky-500 text-white"
                  : "text-ink-300 hover:text-white"
              }`}
            >
              {MODE_INFO[m].label}
            </button>
          ))}
        </div>
        {score && (
          <div className="hidden items-center gap-1.5 rounded-xl border border-white/8 bg-black/30 px-2.5 py-1.5 text-xs md:flex" title="推定難易度">
            <span className="text-ink-400">難易度</span>
            <span className="font-mono font-black">{score.difficulty.overall.toFixed(1)}</span>
            <span className="text-ink-300">{score.difficulty.label}</span>
          </div>
        )}
        <div className="relative">
          <button className="btn btn-primary" onClick={() => setExportOpen((v) => !v)} disabled={!score}>
            <Download className="size-4" /> 書き出し <ChevronDown className="size-3.5" />
          </button>
          {exportOpen && (
            <div className="glass absolute right-0 top-full z-40 mt-2 w-72 rounded-2xl p-1.5 shadow-2xl" onMouseLeave={() => setExportOpen(false)}>
              <MenuItem icon={<FileText className="size-4" />} title="PDF（楽譜）" sub="A4 縦・ベクター。指番号/コードは表示設定どおり" onClick={exportPdf} />
              <MenuItem icon={<Music className="size-4" />} title="MIDI（楽譜どおり）" sub="右手/左手が別トラック・ペダル付き" onClick={() => exportFile("midi")} />
              <MenuItem
                icon={<Piano className="size-4" />}
                title="MIDI（AI の生データ）"
                sub="量子化前。原曲どおりの揺れたタイミング"
                onClick={async () => {
                  setExportOpen(false);
                  try {
                    const { blob, name } = await api.rawMidi(job.id);
                    downloadBlob(blob, name);
                  } catch (e) {
                    flash((e as Error).message);
                  }
                }}
              />
              <MenuItem icon={<FileMusic className="size-4" />} title="MusicXML" sub="MuseScore・Finale・Dorico・Guitar Pro で開けます" onClick={() => exportFile("musicxml")} />
              {instrument === "guitar" && (
                <MenuItem icon={<Guitar className="size-4" />} title="TAB譜（テキスト）" sub="e|--0--3--| 形式。そのまま貼り付けて共有できます" onClick={() => exportFile("tab")} />
              )}
            </div>
          )}
        </div>
      </header>

      {/* ツールバー */}
      <div className="flex flex-wrap items-center gap-2 border-b border-white/6 px-3 py-2 text-xs">
        <div className="flex rounded-lg border border-white/8 bg-black/20 p-0.5">
          {(
            [
              ["staff", "五線譜", <FileText key="s" className="size-3.5" />],
              ["split", "両方", <Rows3 key="b" className="size-3.5" />],
              ["roll", "ピアノロール", <Columns2 key="r" className="size-3.5" />],
            ] as const
          ).map(([v, l, icon]) => (
            <button key={v} onClick={() => setView(v)} className={`flex items-center gap-1 rounded-md px-2.5 py-1 font-semibold ${view === v ? "bg-white/12 text-white" : "text-ink-300"}`}>
              {icon}
              {l}
            </button>
          ))}
        </div>
        <label className={`flex items-center gap-1.5 ${instrument === "guitar" ? "hidden" : ""}`}>
          <span className="text-ink-400">指番号</span>
          <select className="field !py-1 !text-xs" value={display.fingering} onChange={(e) => setDisplay((d) => ({ ...d, fingering: e.target.value as DisplayOptions["fingering"] }))}>
            <option value="difficult">難所だけ</option>
            <option value="all">すべて</option>
            <option value="none">表示しない</option>
          </select>
        </label>
        <label className="flex items-center gap-1 text-ink-200">
          <input type="checkbox" className="accent-indigo-500" checked={display.chords} onChange={(e) => setDisplay((d) => ({ ...d, chords: e.target.checked }))} />
          コード
        </label>
        <label className={`flex items-center gap-1 text-ink-200 ${instrument === "guitar" ? "hidden" : ""}`}>
          <input type="checkbox" className="accent-indigo-500" checked={display.pedal} onChange={(e) => setDisplay((d) => ({ ...d, pedal: e.target.checked }))} />
          ペダル
        </label>
        <label className="flex items-center gap-1 text-ink-200">
          <input type="checkbox" className="accent-indigo-500" checked={display.dynamics} onChange={(e) => setDisplay((d) => ({ ...d, dynamics: e.target.checked }))} />
          強弱
        </label>
        {view !== "roll" && (
          <div className="flex items-center gap-1">
            <button className="btn btn-ghost !p-1.5" onClick={() => setZoom((z) => Math.max(0.4, +(z - 0.1).toFixed(2)))} title="縮小">
              <ZoomOut className="size-3.5" />
            </button>
            <span className="w-9 text-center tabular-nums text-ink-300">{Math.round(zoom * 100)}%</span>
            <button className="btn btn-ghost !p-1.5" onClick={() => setZoom((z) => Math.min(2, +(z + 0.1).toFixed(2)))} title="拡大">
              <ZoomIn className="size-3.5" />
            </button>
          </div>
        )}
        {view !== "staff" && (
          <label className="flex items-center gap-1.5">
            <span className="text-ink-400">グリッド</span>
            <select className="field !py-1 !text-xs" value={grid} onChange={(e) => setGrid(Number(e.target.value))}>
              {GRID_OPTIONS.map((g) => (
                <option key={g.label} value={g.value}>
                  {g.label}
                </option>
              ))}
            </select>
          </label>
        )}
        <button className="btn btn-ghost !px-2 !py-1" onClick={() => setArrangeOpen(true)} disabled={!score}>
          <Settings2 className="size-3.5" /> 編曲の設定
        </button>
        <div className="ml-auto flex items-center gap-1">
          <button className="btn btn-ghost !p-1.5" onClick={undo} disabled={!canUndo} title="元に戻す (Ctrl+Z)">
            <Undo2 className="size-3.5" />
          </button>
          <button className="btn btn-ghost !p-1.5" onClick={redo} disabled={!canRedo} title="やり直し (Ctrl+Y)">
            <Redo2 className="size-3.5" />
          </button>
          <span className="ml-1 flex w-24 items-center gap-1 text-ink-400">
            {saveState === "saving" && (
              <>
                <LoaderCircle className="size-3 animate-spin" /> 保存中…
              </>
            )}
            {saveState === "saved" && (
              <>
                <CircleCheck className="size-3 text-emerald-400" /> {score?.edited ? "保存済み" : "未編集"}
              </>
            )}
            {saveState === "dirty" && "未保存の変更"}
            {saveState === "error" && <span className="text-rose-300">保存に失敗</span>}
          </span>
        </div>
      </div>

      {/* 本体 */}
      <div className="flex min-h-0 flex-1 flex-col gap-3 p-3 lg:flex-row">
        <div className="flex min-h-[70vh] min-w-0 flex-1 flex-col gap-3 lg:min-h-0">
          {loading || !score ? (
            <div className="grid flex-1 place-items-center text-ink-300">
              <LoaderCircle className="size-6 animate-spin" />
            </div>
          ) : (
            <>
              {view !== "roll" && (
                <div className={view === "split" ? "min-h-0 flex-[1.25]" : "min-h-0 flex-1"}>
                  <ScoreView
                    xml={xml}
                    zoom={zoom}
                    bars={barsOf(score)}
                    getBeat={getBeat}
                    playing={playing}
                    loop={loopRange}
                    difficult={difficult}
                    onMeasureClick={onMeasureClick}
                    focusMeasure={focusMeasure}
                  />
                </div>
              )}
              {view !== "staff" && (
                <div className={view === "split" ? "min-h-[260px] flex-1" : "min-h-0 flex-1"}>
                  <PianoRoll
                    score={score}
                    selection={selection}
                    setSelection={setSelection}
                    onEdit={applyEdit}
                    getBeat={getBeat}
                    playing={playing}
                    loop={loopRange}
                    onSeek={seek}
                    onLoopSet={(a, b) => {
                      setLoopFrom(measureIndexAt(a, score) + 1);
                      setLoopTo(measureIndexAt(b - 1e-3, score) + 1);
                      setLoopOn(true);
                    }}
                    grid={grid}
                    fingering={display.fingering}
                    difficult={difficult}
                    focusBeat={focusBeat}
                  />
                </div>
              )}
            </>
          )}
        </div>

        <aside className="glass flex w-full shrink-0 flex-col rounded-2xl lg:w-80">
          <div className="flex border-b border-white/6 p-1">
            {(
              [
                ["hard", "難所・運指"],
                ["edit", `編集${selection.size ? ` (${selection.size})` : ""}`],
                ["info", "情報"],
              ] as const
            ).map(([k, l]) => (
              <button key={k} onClick={() => setSide(k)} className={`flex-1 rounded-lg py-1.5 text-xs font-bold ${side === k ? "bg-white/10 text-white" : "text-ink-400"}`}>
                {l}
              </button>
            ))}
          </div>
          <div className="scroll-thin min-h-0 flex-1 overflow-y-auto p-4">
            {score && side === "hard" && (
              <DifficultyPanel score={score} onLoop={practiceSpan} onRecalc={recalc} stale={stale} busy={recalcBusy} guitar={instrument === "guitar"} />
            )}
            {score && side === "edit" && <Inspector score={score} selection={selection} onEdit={applyEdit} onDelete={deleteSelected} />}
            {score && side === "info" && <InfoPanel job={job} score={score} />}
          </div>
        </aside>
      </div>

      {/* 再生バー */}
      {score && (
        <div className="px-3 pb-3">
          <Transport
            playing={playing}
            onToggle={togglePlay}
            onStop={stop}
            onRewind={() => seek(loopRange ? loopRange.start : 0)}
            position={posText}
            rate={rate}
            setRate={setRate}
            tempoText={tempoLabel(score, rate)}
            loopOn={loopOn}
            setLoopOn={setLoopOn}
            loopFrom={loopFrom}
            loopTo={loopTo}
            setLoopRange={setLoopRange}
            measures={measures}
            metronome={metronome}
            setMetronome={setMetronome}
            muteR={muteR}
            muteL={muteL}
            setMute={(h, v) => (h === "R" ? setMuteR(v) : setMuteL(v))}
            source={source}
            setSource={changeSource}
            targetLabel={job.result?.target?.split("（")[0] ?? "分離音"}
            guitar={instrument === "guitar"}
            guitarTone={guitarTone}
            setGuitarTone={(t) => {
              applyGuitarTone(t);
              setGuitarToneState(t);
              // 楽譜の再生エンジンを作り直して新しい音色を読み込む
              const e = engines.current.score;
              if (e && scoreRef.current) e.setScore(scoreRef.current);
              flash(t === "steel" ? "スチール弦の音に切り替えました" : "ナイロン弦の音に切り替えました");
            }}
            loadingSound={loadingSound}
          />
        </div>
      )}

      {score && (
        <ArrangePanel
          open={arrangeOpen}
          onClose={() => setArrangeOpen(false)}
          score={score}
          mode={mode}
          busy={arrangeBusy}
          onApply={rearrange}
          instrument={instrument}
          onReset={resetEdits}
        />
      )}

      {(toast || exportMsg) && (
        <div className="pointer-events-none fixed bottom-24 left-1/2 z-50 -translate-x-1/2 rounded-full bg-black/80 px-4 py-2 text-sm text-white shadow-xl">
          {exportMsg ? (
            <span className="flex items-center gap-2">
              <LoaderCircle className="size-4 animate-spin" /> {exportMsg}
            </span>
          ) : (
            toast
          )}
        </div>
      )}
    </main>
  );
}

const BEAT_LABEL: Record<string, string> = {
  audio: "音声の拍",
  notes: "音符の立ち上がりで再追跡",
  "audio+fit": "音声の拍を音符に吸着",
  "notes+fit": "再追跡＋音符に吸着",
  "audio+fit+const": "一定テンポに整列",
  "notes+fit+const": "一定テンポに整列",
};

function MenuItem({ icon, title, sub, onClick }: { icon: React.ReactNode; title: string; sub: string; onClick: () => void }) {
  return (
    <button className="flex w-full items-start gap-3 rounded-xl px-3 py-2.5 text-left hover:bg-white/8" onClick={onClick}>
      <span className="mt-0.5 text-indigo-300">{icon}</span>
      <span>
        <span className="block text-sm font-semibold">{title}</span>
        <span className="block text-[11px] text-ink-400">{sub}</span>
      </span>
    </button>
  );
}

function InfoPanel({ job, score }: { job: Job; score: Score }) {
  const r = job.result;
  const [engine, setEngine] = useState<"auto" | "bytedance" | "basicpitch">("auto");
  const [busy, setBusy] = useState(false);
  const redo = async () => {
    if (!confirm("採譜からやり直します（ダウンロードと音源分離はそのまま）。手で直した内容は消えます。よろしいですか？")) return;
    setBusy(true);
    try {
      await api.retranscribe(job.id, engine);
      // ページは読み込み直さない（ブラウザ版はこのタブの中で処理しているので、読み込み直すと止まる）
      window.dispatchEvent(new Event("gakufu:job-reload"));
    } catch (e) {
      alert((e as Error).message);
      setBusy(false);
    }
  };
  const rows: [string, React.ReactNode][] = [
    ...(score.instrument === "guitar" && score.guitar
      ? ([
          ["調弦", score.guitar.tuningLabel],
          ["カポ", score.guitar.capo ? `${score.guitar.capo}フレット` : "なし"],
        ] as [string, React.ReactNode][])
      : []),
    ["音源の判定", r?.sourceKindLabel ?? "-"],
    ["拍の取り方", (score.source as { beatMethod?: string })?.beatMethod ? BEAT_LABEL[(score.source as { beatMethod?: string }).beatMethod!] ?? "-" : "-"],
    ["取り除いた雑音", `${(score.source as { noiseRemoved?: number })?.noiseRemoved ?? 0}音`],
    ["採譜に使った音", r?.target ?? "-"],
    ["採譜エンジン", r?.engine ?? "-"],
    ["音源分離", r?.separation?.engine ?? "なし"],
    ["調", `${score.key.ja}（${score.key.en}）`],
    ["拍子・テンポ", `${meterLabel(score)} ・ ${tempoLabel(score)}`],
    ["音符数", `${score.notes.length}（右手 ${score.notes.filter((n) => n.hand === "R").length} / 左手 ${score.notes.filter((n) => n.hand === "L").length}）`],
    ["小節数", `${measureCount(score)}${(score.bars?.length ?? 0) > 1 && barsOf(score).some((x, i, a) => i > 0 && Math.abs(x - a[i - 1] - (score.measureBeats ?? 4)) > 1e-6 && i < a.length - 1) ? "（途中に拍子の変わる小節あり）" : ""}`],
    ["解析した長さ", job.meta.analyzedDuration ? formatSec(job.meta.analyzedDuration) : "-"],
  ];
  return (
    <div className="space-y-4 text-sm">
      <dl className="grid grid-cols-[auto_1fr] gap-x-3 gap-y-2">
        {rows.map(([k, v]) => (
          <div key={k} className="contents">
            <dt className="text-xs text-ink-400">{k}</dt>
            <dd className="text-xs text-ink-100">{v}</dd>
          </div>
        ))}
      </dl>
      {score.chords.length > 0 && (
        <div>
          <div className="label mb-1.5">コード進行（先頭）</div>
          <div className="flex flex-wrap gap-1">
            {score.chords.slice(0, 32).map((c, i) => (
              <span key={i} className="rounded-md bg-white/6 px-1.5 py-0.5 font-mono text-[11px]">
                {prettyChord(c.symbol)}
              </span>
            ))}
          </div>
        </div>
      )}
      <div className="rounded-xl border border-white/8 bg-black/15 p-3">
        <div className="label mb-1.5">採譜AIを選び直す</div>
        <select className="field w-full !py-1.5 !text-xs" value={engine} onChange={(e) => setEngine(e.target.value as typeof engine)}>
          <option value="auto">自動（ピアノならピアノ用、他の楽器なら汎用）</option>
          <option value="bytedance">ピアノ専用（ByteDance・ペダルも取れる）</option>
          <option value="basicpitch">楽器を問わない（Basic Pitch・ギター/合唱/バンド向け）</option>
        </select>
        <button className="btn btn-ghost mt-2 w-full !py-1.5 text-xs" onClick={redo} disabled={busy}>
          {busy ? <LoaderCircle className="size-3.5 animate-spin" /> : <Wand2 className="size-3.5" />}
          この採譜AIで採譜からやり直す
        </button>
      </div>
      {job.meta.webpage_url && (
        <a href={job.meta.webpage_url} target="_blank" rel="noreferrer" className="btn btn-ghost w-full">
          <ExternalLink className="size-4" /> 元の動画を開く
        </a>
      )}
      <div className="flex items-start gap-2 rounded-xl bg-white/[0.03] p-3 text-[11px] text-ink-400">
        <Info className="mt-0.5 size-3.5 shrink-0" />
        AI 採譜は完璧ではありません。音のずれやオクターブはピアノロールで直せます（自動保存）。
      </div>
    </div>
  );
}
