"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { previewNote } from "@/lib/player";
import { bars as barsOf, beatLen, isBlack, measureIndexAt, measureStart, nextNoteId, noteName, prettyChord, snap } from "@/lib/music";
import type { Note, Score } from "@/lib/types";

const KEY_W = 58;
/** 1弦〜6弦の色（ギター表示） */
export const STRING_COLORS = ["#f472b6", "#fb923c", "#facc15", "#4ade80", "#38bdf8", "#a78bfa"];
function rgba(hex: string, a: number) {
  const v = parseInt(hex.slice(1), 16);
  return `rgba(${(v >> 16) & 255},${(v >> 8) & 255},${v & 255},${a})`;
}
const RULER_H = 42;
const MIN_P = 21;
const MAX_P = 108;

export interface PianoRollProps {
  score: Score;
  selection: Set<string>;
  setSelection: (s: Set<string>) => void;
  onEdit: (fn: (notes: Note[]) => Note[], label: string) => void;
  getBeat: () => number;
  playing: boolean;
  loop: { start: number; end: number } | null;
  onSeek: (beat: number) => void;
  onLoopSet: (startBeat: number, endBeat: number) => void;
  grid: number;
  fingering: "all" | "difficult" | "none";
  difficult: Set<number>;
  focusBeat?: number | null;
}

type Drag =
  | { kind: "move"; x0: number; y0: number; anchor: Note; ids: Set<string>; dBeat: number; dPitch: number }
  | { kind: "resize"; x0: number; anchor: Note; ids: Set<string>; dDur: number }
  | { kind: "marquee"; x0: number; y0: number; x1: number; y1: number; base: Set<string> }
  | { kind: "loop"; m0: number; m1: number }
  | { kind: "seek" };

export function PianoRoll(props: PianoRollProps) {
  const wrapRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const propsRef = useRef(props);
  propsRef.current = props;
  const view = useRef({ scrollX: 0, scrollY: 0, ppb: 56, rowH: 10, w: 800, h: 400 });
  const drag = useRef<Drag | null>(null);
  const dirty = useRef(true);
  const [ppb, setPpb] = useState(56);
  const [scrollInfo, setScrollInfo] = useState({ x: 0, max: 0 });
  const initialized = useRef(false);

  const totalBeats = Math.max(props.score.totalBeats, ...props.score.notes.map((n) => n.start + n.dur), 4);

  const contentW = () => totalBeats * view.current.ppb + 200;
  const contentH = () => (MAX_P - MIN_P + 1) * view.current.rowH;

  const clampScroll = useCallback(() => {
    const v = view.current;
    v.scrollX = Math.max(0, Math.min(v.scrollX, Math.max(0, contentW() - (v.w - KEY_W))));
    v.scrollY = Math.max(0, Math.min(v.scrollY, Math.max(0, contentH() - (v.h - RULER_H))));
    setScrollInfo({ x: v.scrollX, max: Math.max(0, contentW() - (v.w - KEY_W)) });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [totalBeats]);

  // 座標変換
  const bx = (b: number) => KEY_W + b * view.current.ppb - view.current.scrollX;
  const xb = (x: number) => (x - KEY_W + view.current.scrollX) / view.current.ppb;
  const py = (p: number) => RULER_H + (MAX_P - p) * view.current.rowH - view.current.scrollY;
  const yp = (y: number) => MAX_P - Math.floor((y - RULER_H + view.current.scrollY) / view.current.rowH);

  // 初期位置：音の多い音域を真ん中に
  useEffect(() => {
    if (initialized.current || !props.score.notes.length) return;
    const ps = props.score.notes.map((n) => n.pitch).sort((a, b) => a - b);
    const med = ps[Math.floor(ps.length / 2)];
    const v = view.current;
    v.scrollY = (MAX_P - med) * v.rowH - (v.h - RULER_H) / 2;
    clampScroll();
    initialized.current = true;
    dirty.current = true;
  }, [props.score.notes, clampScroll]);

  useEffect(() => {
    dirty.current = true;
  }, [props.score, props.selection, props.loop, props.fingering, props.difficult, props.grid]);

  useEffect(() => {
    const fb = props.focusBeat;
    if (fb == null) return;
    const v = view.current;
    v.scrollX = Math.max(0, fb * v.ppb - (v.w - KEY_W) * 0.2);
    clampScroll();
    dirty.current = true;
  }, [props.focusBeat, clampScroll]);

  // サイズ追従
  useEffect(() => {
    const wrap = wrapRef.current;
    const canvas = canvasRef.current;
    if (!wrap || !canvas) return;
    const ro = new ResizeObserver(() => {
      const r = wrap.getBoundingClientRect();
      const dpr = window.devicePixelRatio || 1;
      view.current.w = r.width;
      view.current.h = r.height;
      canvas.width = Math.round(r.width * dpr);
      canvas.height = Math.round(r.height * dpr);
      canvas.style.width = `${r.width}px`;
      canvas.style.height = `${r.height}px`;
      clampScroll();
      dirty.current = true;
    });
    ro.observe(wrap);
    return () => ro.disconnect();
  }, [clampScroll]);

  // 描画ループ
  useEffect(() => {
    let raf = 0;
    let lastBeat = -1;
    const loop = () => {
      const p = propsRef.current;
      const beat = p.getBeat();
      if (p.playing) {
        const v = view.current;
        const x = bx(beat);
        if (x > v.w * 0.85 || x < KEY_W) {
          v.scrollX = Math.max(0, beat * v.ppb - (v.w - KEY_W) * 0.25);
          clampScroll();
        }
      }
      if (dirty.current || Math.abs(beat - lastBeat) > 1e-4) {
        draw(beat);
        lastBeat = beat;
        dirty.current = false;
      }
      raf = requestAnimationFrame(loop);
    };
    raf = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(raf);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [clampScroll]);

  function draw(beat: number) {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const p = propsRef.current;
    const s = p.score;
    const v = view.current;
    const dpr = window.devicePixelRatio || 1;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    const W = v.w;
    const H = v.h;
    const bl = beatLen(s);
    const barList = barsOf(s);
    const barSet = new Set(barList.map((x) => Math.round(x * 1000)));
    const isBarPos = (x: number) => barSet.has(Math.round(x * 1000));
    const mStart = (m: number) => measureStart(m, s);
    ctx.fillStyle = "#0b0d15";
    ctx.fillRect(0, 0, W, H);

    // 行（黒鍵の段を暗く）
    const pTop = Math.min(MAX_P, yp(RULER_H));
    const pBot = Math.max(MIN_P, yp(H));
    for (let pitch = pBot; pitch <= pTop; pitch++) {
      const y = py(pitch);
      ctx.fillStyle = isBlack(pitch) ? "#0e111b" : "#131726";
      ctx.fillRect(KEY_W, y, W - KEY_W, v.rowH);
      if (pitch % 12 === 0) {
        ctx.fillStyle = "rgba(255,255,255,0.10)";
        ctx.fillRect(KEY_W, y + v.rowH - 1, W - KEY_W, 1);
      }
    }

    const b0 = Math.max(0, Math.floor(xb(KEY_W)));
    const b1 = Math.ceil(xb(W));
    // 難所の小節
    const meas0 = measureIndexAt(b0, s);
    const meas1 = measureIndexAt(b1, s) + 1;
    for (let m = meas0; m <= meas1; m++) {
      if (p.difficult.has(m)) {
        ctx.fillStyle = "rgba(244,63,94,0.06)";
        ctx.fillRect(bx(mStart(m)), RULER_H, (mStart(m + 1) - mStart(m)) * v.ppb, H - RULER_H);
      }
    }
    // ループ範囲
    if (p.loop) {
      ctx.fillStyle = "rgba(99,102,241,0.12)";
      ctx.fillRect(bx(p.loop.start), RULER_H, (p.loop.end - p.loop.start) * v.ppb, H - RULER_H);
    }
    // 縦線
    const sub = p.grid;
    const subPx = sub * v.ppb;
    // 拍線は「拍」（複合拍子は付点4分）ごと、小節線は太く、4分音符刻みは薄く
    for (let b = b0; b <= b1; b++) {
      const x = bx(b);
      const isBar = isBarPos(b);
      const isBeat = Math.abs(b / bl - Math.round(b / bl)) < 1e-6;
      ctx.fillStyle = isBar ? "rgba(255,255,255,0.22)" : isBeat ? "rgba(255,255,255,0.08)" : "rgba(255,255,255,0.045)";
      ctx.fillRect(x, RULER_H, 1, H - RULER_H);
      if (bl !== 1 && !isBar) {
        // 付点4分の拍頭（整数でない位置）
        const half = b + 0.5;
        if (Math.abs(half / bl - Math.round(half / bl)) < 1e-6 && !isBarPos(half)) {
          ctx.fillStyle = "rgba(255,255,255,0.08)";
          ctx.fillRect(bx(half), RULER_H, 1, H - RULER_H);
        }
      }
      if (subPx >= 9 && sub < 1) {
        ctx.fillStyle = "rgba(255,255,255,0.035)";
        for (let k = 1; k * sub < 1 - 1e-6; k++) ctx.fillRect(bx(b + k * sub), RULER_H, 1, H - RULER_H);
      }
    }

    // ドラッグ中のずれ
    const d = drag.current;
    const guitar = s.instrument === "guitar";
    const showFinger = (n: Note) =>
      p.fingering === "all" || (p.fingering === "difficult" && p.difficult.has(measureIndexAt(n.start, s)));
    const sounding = new Set<number>();
    ctx.font = "600 9px Inter, system-ui, sans-serif";
    ctx.textBaseline = "middle";
    for (const n of s.notes) {
      let start = n.start;
      let dur = n.dur;
      let pitch = n.pitch;
      const sel = p.selection.has(n.id);
      if (d && sel) {
        if (d.kind === "move") {
          start = Math.max(0, start + d.dBeat);
          pitch = Math.min(MAX_P, Math.max(MIN_P, pitch + d.dPitch));
        } else if (d.kind === "resize") {
          dur = Math.max(p.grid, dur + d.dDur);
        }
      }
      if (p.playing && beat >= n.start && beat < n.start + n.dur) sounding.add(n.pitch);
      const x = bx(start);
      const w = Math.max(3, dur * v.ppb - 1);
      if (x + w < KEY_W || x > W) continue;
      const y = py(pitch);
      if (y + v.rowH < RULER_H || y > H) continue;
      const a = 0.45 + 0.55 * Math.min(1, n.vel / 110);
      ctx.fillStyle =
        guitar && n.string
          ? rgba(STRING_COLORS[n.string - 1], a)
          : guitar
            ? `rgba(148,163,184,${a})`
            : n.hand === "R"
              ? `rgba(90,180,255,${a})`
              : `rgba(255,181,71,${a})`;
      roundRect(ctx, x, y + 1, w, v.rowH - 2, 2.5);
      ctx.fill();
      if (sel) {
        ctx.strokeStyle = "#ffffff";
        ctx.lineWidth = 1.5;
        roundRect(ctx, x, y + 1, w, v.rowH - 2, 2.5);
        ctx.stroke();
      }
      if (guitar && n.string && w >= 10 && v.rowH >= 9) {
        ctx.fillStyle = n.stringLocked ? "#fde68a" : "#0b0d15";
        ctx.fillText(String(n.fret), x + 3, y + v.rowH / 2 + 0.5);
      } else if (!guitar && n.finger && showFinger(n) && w >= 10 && v.rowH >= 9) {
        ctx.fillStyle = n.fingerLocked ? "#fde68a" : "#0b0d15";
        ctx.fillText(String(n.finger), x + 3, y + v.rowH / 2 + 0.5);
      }
      if (n.ornament === "trill" && w > 22) {
        ctx.fillStyle = "#0b0d15";
        ctx.fillText("tr", x + w - 13, y + v.rowH / 2 + 0.5);
      }
    }

    // 再生位置
    const phx = bx(beat);
    if (phx >= KEY_W && phx <= W) {
      ctx.fillStyle = "#f0abfc";
      ctx.fillRect(phx - 0.5, RULER_H, 1.5, H - RULER_H);
    }

    // 範囲選択の枠
    if (d && d.kind === "marquee") {
      ctx.fillStyle = "rgba(129,140,248,0.12)";
      ctx.strokeStyle = "rgba(165,180,252,0.8)";
      ctx.lineWidth = 1;
      const rx = Math.min(d.x0, d.x1);
      const ry = Math.min(d.y0, d.y1);
      ctx.fillRect(rx, ry, Math.abs(d.x1 - d.x0), Math.abs(d.y1 - d.y0));
      ctx.strokeRect(rx + 0.5, ry + 0.5, Math.abs(d.x1 - d.x0), Math.abs(d.y1 - d.y0));
    }

    // ルーラー
    ctx.fillStyle = "#10131e";
    ctx.fillRect(0, 0, W, RULER_H);
    ctx.fillStyle = "rgba(255,255,255,0.08)";
    ctx.fillRect(0, RULER_H - 1, W, 1);
    ctx.fillRect(0, 18, W, 1);
    ctx.font = "600 10px Inter, system-ui, sans-serif";
    for (let m = meas0; m <= meas1; m++) {
      const x = bx(mStart(m));
      if (x < KEY_W - 40) continue;
      const mLen = mStart(m + 1) - mStart(m);
      ctx.fillStyle = "rgba(255,255,255,0.35)";
      ctx.fillRect(x, 0, 1, 18);
      ctx.fillStyle = "#c7cce4";
      ctx.fillText(String(m + 1), x + 4, 9);
      if (p.difficult.has(m)) {
        ctx.fillStyle = "#fb7185";
        ctx.fillRect(x + 1, 15, mLen * v.ppb - 2, 3);
      }
      if (Math.abs(mLen - (s.measureBeats ?? mLen)) > 1e-6) {
        // 長さの違う小節（2/4 など）には拍子を小さく出す
        ctx.fillStyle = "#fbbf24";
        ctx.fillText(`${Math.round(mLen)}/4`, x + 18, 9);
      }
    }
    const loopShow = d && d.kind === "loop" ? { start: mStart(Math.min(d.m0, d.m1)), end: mStart(Math.max(d.m0, d.m1) + 1) } : p.loop;
    if (loopShow) {
      ctx.fillStyle = "rgba(129,140,248,0.55)";
      ctx.fillRect(bx(loopShow.start), 0, (loopShow.end - loopShow.start) * v.ppb, 4);
    }
    ctx.font = "700 11px Inter, system-ui, sans-serif";
    for (const c of s.chords) {
      const x = bx(c.start);
      if (x > W || bx(c.end) < KEY_W) continue;
      ctx.fillStyle = "rgba(255,255,255,0.06)";
      ctx.fillRect(Math.max(KEY_W, x), 19, Math.max(0, Math.min(W, bx(c.end)) - Math.max(KEY_W, x)) - 2, RULER_H - 21);
      ctx.fillStyle = "#e9d5ff";
      ctx.fillText(prettyChord(c.symbol), Math.max(KEY_W + 2, x + 4), 30);
    }
    if (phx >= KEY_W && phx <= W) {
      ctx.fillStyle = "#f0abfc";
      ctx.beginPath();
      ctx.moveTo(phx - 5, 0);
      ctx.lineTo(phx + 5, 0);
      ctx.lineTo(phx, 7);
      ctx.fill();
    }

    // 鍵盤
    ctx.fillStyle = "#0f1220";
    ctx.fillRect(0, RULER_H, KEY_W, H - RULER_H);
    for (let pitch = pBot; pitch <= pTop; pitch++) {
      const y = py(pitch);
      const black = isBlack(pitch);
      const on = sounding.has(pitch);
      ctx.fillStyle = on ? "#f0abfc" : black ? "#1a1d2b" : "#e9ebf5";
      ctx.fillRect(black ? 0 : 0, y, black ? KEY_W * 0.62 : KEY_W - 1, v.rowH - (black ? 0 : 1));
      if (pitch % 12 === 0 && v.rowH >= 8) {
        ctx.fillStyle = on ? "#0b0d15" : "#4b5375";
        ctx.font = "600 9px Inter, system-ui, sans-serif";
        ctx.fillText(noteName(pitch), KEY_W - 26, y + v.rowH / 2);
      }
    }
    ctx.fillStyle = "#10131e";
    ctx.fillRect(0, 0, KEY_W, RULER_H);
  }

  // ------------------------------------------------------------------ 入力
  function hitNote(x: number, y: number): Note | null {
    const s = propsRef.current.score;
    const v = view.current;
    for (let i = s.notes.length - 1; i >= 0; i--) {
      const n = s.notes[i];
      const nx = bx(n.start);
      const nw = Math.max(3, n.dur * v.ppb - 1);
      const ny = py(n.pitch);
      if (x >= nx && x <= nx + nw && y >= ny && y <= ny + v.rowH) return n;
    }
    return null;
  }

  function local(e: React.PointerEvent | React.MouseEvent | React.WheelEvent) {
    const r = canvasRef.current!.getBoundingClientRect();
    return { x: e.clientX - r.left, y: e.clientY - r.top };
  }

  function onPointerDown(e: React.PointerEvent) {
    const p = propsRef.current;
    const { x, y } = local(e);
    canvasRef.current?.focus();
    if (x < KEY_W && y > RULER_H) {
      void previewNote(yp(y), 0.7, p.score.instrument === "guitar");
      return;
    }
    (e.target as HTMLElement).setPointerCapture(e.pointerId);
    if (y < RULER_H) {
      const b = Math.max(0, xb(x));
      if (e.shiftKey) {
        const m = measureIndexAt(b, p.score);
        drag.current = { kind: "loop", m0: m, m1: m };
      } else {
        drag.current = { kind: "seek" };
        p.onSeek(snap(b, p.grid));
      }
      dirty.current = true;
      return;
    }
    const hit = hitNote(x, y);
    if (hit) {
      let sel = new Set(p.selection);
      if (e.shiftKey || e.ctrlKey || e.metaKey) {
        if (sel.has(hit.id)) {
          sel.delete(hit.id);
          p.setSelection(sel);
          return;
        }
        sel.add(hit.id);
      } else if (!sel.has(hit.id)) {
        sel = new Set([hit.id]);
      }
      p.setSelection(sel);
      void previewNote(hit.pitch, hit.vel / 127, p.score.instrument === "guitar", hit.string);
      const right = bx(hit.start) + hit.dur * view.current.ppb;
      if (x > right - 7 && hit.dur * view.current.ppb > 12) {
        drag.current = { kind: "resize", x0: x, anchor: hit, ids: sel, dDur: 0 };
      } else {
        drag.current = { kind: "move", x0: x, y0: y, anchor: hit, ids: sel, dBeat: 0, dPitch: 0 };
      }
    } else {
      const base = e.shiftKey ? new Set(p.selection) : new Set<string>();
      if (!e.shiftKey) p.setSelection(base);
      drag.current = { kind: "marquee", x0: x, y0: y, x1: x, y1: y, base };
    }
    dirty.current = true;
  }

  function onPointerMove(e: React.PointerEvent) {
    const p = propsRef.current;
    const { x, y } = local(e);
    const d = drag.current;
    const canvas = canvasRef.current!;
    if (!d) {
      const hit = y > RULER_H && x > KEY_W ? hitNote(x, y) : null;
      if (hit) {
        const right = bx(hit.start) + hit.dur * view.current.ppb;
        canvas.style.cursor = x > right - 7 && hit.dur * view.current.ppb > 12 ? "ew-resize" : "grab";
      } else canvas.style.cursor = y < RULER_H ? "pointer" : x < KEY_W ? "pointer" : "crosshair";
      return;
    }
    const v = view.current;
    if (d.kind === "move") {
      const target = snap(d.anchor.start + (x - d.x0) / v.ppb, p.grid);
      const newDBeat = Math.max(-d.anchor.start, target - d.anchor.start);
      const newDPitch = Math.round(-(y - d.y0) / v.rowH);
      if (newDPitch !== d.dPitch) void previewNote(Math.min(MAX_P, Math.max(MIN_P, d.anchor.pitch + newDPitch)), 0.5, p.score.instrument === "guitar");
      d.dBeat = newDBeat;
      d.dPitch = newDPitch;
    } else if (d.kind === "resize") {
      const end = snap(d.anchor.start + d.anchor.dur + (x - d.x0) / v.ppb, p.grid);
      d.dDur = Math.max(p.grid, end - d.anchor.start) - d.anchor.dur;
    } else if (d.kind === "marquee") {
      d.x1 = x;
      d.y1 = y;
      const bA = xb(Math.min(d.x0, d.x1));
      const bB = xb(Math.max(d.x0, d.x1));
      const pA = yp(Math.max(d.y0, d.y1));
      const pB = yp(Math.min(d.y0, d.y1));
      const sel = new Set(d.base);
      for (const n of p.score.notes) {
        if (n.start < bB && n.start + n.dur > bA && n.pitch >= pA && n.pitch <= pB) sel.add(n.id);
      }
      p.setSelection(sel);
    } else if (d.kind === "loop") {
      d.m1 = measureIndexAt(Math.max(0, xb(x)), p.score);
    } else if (d.kind === "seek") {
      p.onSeek(snap(Math.max(0, xb(x)), p.grid));
    }
    dirty.current = true;
  }

  function onPointerUp() {
    const p = propsRef.current;
    const d = drag.current;
    drag.current = null;
    dirty.current = true;
    if (!d) return;
    if (d.kind === "move" && (Math.abs(d.dBeat) > 1e-9 || d.dPitch !== 0)) {
      const { dBeat, dPitch, ids } = d;
      p.onEdit(
        (notes) =>
          notes.map((n) =>
            ids.has(n.id)
              ? { ...n, start: Math.max(0, +(n.start + dBeat).toFixed(6)), pitch: Math.min(MAX_P, Math.max(MIN_P, n.pitch + dPitch)) }
              : n,
          ),
        dPitch ? "音の高さを変更" : "音の位置を変更",
      );
    } else if (d.kind === "resize" && Math.abs(d.dDur) > 1e-9) {
      const { dDur, ids } = d;
      p.onEdit((notes) => notes.map((n) => (ids.has(n.id) ? { ...n, dur: Math.max(p.grid, +(n.dur + dDur).toFixed(6)) } : n)), "長さを変更");
    } else if (d.kind === "loop") {
      p.onLoopSet(measureStart(Math.min(d.m0, d.m1), p.score), measureStart(Math.max(d.m0, d.m1) + 1, p.score));
    }
  }

  function onDoubleClick(e: React.MouseEvent) {
    const p = propsRef.current;
    const { x, y } = local(e);
    if (x < KEY_W || y < RULER_H || hitNote(x, y)) return;
    const pitch = yp(y);
    const start = Math.max(0, snap(xb(x) - p.grid / 2, p.grid));
    const id = nextNoteId(p.score.notes);
    const note: Note = { id, pitch, start, dur: Math.max(p.grid, 0.5), vel: 80, hand: pitch >= 60 ? "R" : "L" };
    p.onEdit((notes) => [...notes, note].sort((a, b) => a.start - b.start || a.pitch - b.pitch), "音符を追加");
    p.setSelection(new Set([id]));
    void previewNote(pitch, 0.7, p.score.instrument === "guitar");
  }

  function onWheel(e: React.WheelEvent) {
    const v = view.current;
    if (e.ctrlKey || e.metaKey) {
      const { x } = local(e);
      const anchor = xb(x);
      const next = Math.min(400, Math.max(10, v.ppb * (e.deltaY < 0 ? 1.15 : 1 / 1.15)));
      v.ppb = next;
      v.scrollX = anchor * next - (x - KEY_W);
      setPpb(next);
    } else if (e.shiftKey || Math.abs(e.deltaX) > Math.abs(e.deltaY)) {
      v.scrollX += e.shiftKey ? e.deltaY : e.deltaX;
    } else {
      v.scrollY += e.deltaY;
    }
    clampScroll();
    dirty.current = true;
  }

  useEffect(() => {
    // ブラウザのページ拡大（Ctrl+ホイール）を止める
    const c = canvasRef.current;
    if (!c) return;
    const stop = (e: WheelEvent) => e.preventDefault();
    c.addEventListener("wheel", stop, { passive: false });
    return () => c.removeEventListener("wheel", stop);
  }, []);

  return (
    <div className="flex h-full min-h-0 flex-col">
      <div ref={wrapRef} className="relative min-h-0 flex-1 overflow-hidden rounded-xl border border-white/8">
        <canvas
          ref={canvasRef}
          tabIndex={0}
          className="block outline-none"
          onPointerDown={onPointerDown}
          onPointerMove={onPointerMove}
          onPointerUp={onPointerUp}
          onPointerCancel={onPointerUp}
          onDoubleClick={onDoubleClick}
          onWheel={onWheel}
          onContextMenu={(e) => e.preventDefault()}
        />
      </div>
      <div className="mt-1.5 flex items-center gap-3 px-1 text-[11px] text-ink-400">
        <input
          type="range"
          className="range flex-1"
          min={0}
          max={Math.max(1, scrollInfo.max)}
          value={scrollInfo.x}
          style={{ ["--fill" as string]: `${(scrollInfo.x / Math.max(1, scrollInfo.max)) * 100}%` }}
          onChange={(e) => {
            view.current.scrollX = Number(e.target.value);
            clampScroll();
            dirty.current = true;
          }}
        />
        <span className="hidden whitespace-nowrap md:inline">
          ダブルクリックで追加 ・ ドラッグで移動 ・ 右端で長さ ・ Shift+ルーラーでループ ・ Ctrl+ホイールで拡大
        </span>
        <span className="tabular-nums">×{(ppb / 56).toFixed(1)}</span>
      </div>
    </div>
  );
}

function roundRect(ctx: CanvasRenderingContext2D, x: number, y: number, w: number, h: number, r: number) {
  const rr = Math.min(r, w / 2, h / 2);
  ctx.beginPath();
  ctx.moveTo(x + rr, y);
  ctx.arcTo(x + w, y, x + w, y + h, rr);
  ctx.arcTo(x + w, y + h, x, y + h, rr);
  ctx.arcTo(x, y + h, x, y, rr);
  ctx.arcTo(x, y, x + w, y, rr);
  ctx.closePath();
}
