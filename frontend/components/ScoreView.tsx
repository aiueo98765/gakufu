"use client";

import { useEffect, useRef, useState } from "react";
import { LoaderCircle } from "lucide-react";
import type { OpenSheetMusicDisplay } from "opensheetmusicdisplay";

interface Box {
  x: number;
  y: number;
  w: number;
  h: number;
  /** 小節内の拍位置 → x 座標（再生カーソル用） */
  entries: [number, number][];
}

export interface ScoreViewProps {
  xml: string | null;
  zoom: number;
  /** 小節の開始位置（4分音符単位）の列。最後は曲の終わり */
  bars: number[];
  getBeat: () => number;
  playing: boolean;
  loop: { start: number; end: number } | null;
  difficult: Set<number>;
  onMeasureClick: (measure: number, shift: boolean) => void;
  focusMeasure?: number | null;
}

/** 長い題名（動画タイトルは長い）が紙からはみ出さないよう、文字数で題名の大きさを決める */
export function fitTitle(osmd: OpenSheetMusicDisplay, xml: string, widthPx?: number, zoom = 1) {
  const m = /<work-title>([\s\S]*?)<\/work-title>/.exec(xml);
  const text = m ? m[1] : "";
  const len = text.length;
  let h = len > 70 ? 1.8 : len > 50 ? 2.2 : len > 32 ? 2.8 : 4.0;
  if (widthPx && len) {
    // 全角は文字の高さとほぼ同じ幅、半角は半分くらい。紙の幅の 9 割に収める
    const wide = [...text].filter((ch) => /[^\u0000-\u00ff]/.test(ch)).length;
    const units = wide + 0.55 * (len - wide);
    h = Math.min(h, (widthPx * 0.9) / (units * 10 * zoom));
  }
  osmd.EngravingRules.SheetTitleHeight = Math.max(1.0, h);
  osmd.EngravingRules.SheetSubtitleHeight = Math.min(2.0, Math.max(1.0, h * 0.75));
}

export function ScoreView(props: ScoreViewProps) {
  const scrollRef = useRef<HTMLDivElement>(null);
  const hostRef = useRef<HTMLDivElement>(null);
  const osmdRef = useRef<OpenSheetMusicDisplay | null>(null);
  const boxesRef = useRef<Box[]>([]);
  const cursorRef = useRef<HTMLDivElement>(null);
  const measureHiRef = useRef<HTMLDivElement>(null);
  const propsRef = useRef(props);
  propsRef.current = props;
  const [boxes, setBoxes] = useState<Box[]>([]);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const renderSeq = useRef(0);

  function measureBoxes(osmd: OpenSheetMusicDisplay): Box[] {
    const unit = 10 * osmd.zoom;
    const list = osmd.GraphicSheet?.MeasureList ?? [];
    const out: Box[] = [];
    for (const staves of list) {
      const ms = (staves || []).filter(Boolean);
      if (!ms.length) {
        out.push({ x: 0, y: 0, w: 0, h: 0, entries: [] });
        continue;
      }
      const top = ms[0].PositionAndShape;
      const bot = ms[ms.length - 1].PositionAndShape;
      const x = top.AbsolutePosition.x * unit;
      const y = (top.AbsolutePosition.y - 2.5) * unit;
      const yb = (bot.AbsolutePosition.y + bot.Size.height + 2.5) * unit;
      const ent: [number, number][] = [];
      for (const m of ms) {
        for (const se of m.staffEntries ?? []) {
          try {
            const ts = se.relInMeasureTimestamp.RealValue * 4;
            const ex = (m.PositionAndShape.AbsolutePosition.x + se.PositionAndShape.RelativePosition.x) * unit;
            ent.push([ts, ex]);
          } catch {
            /* 位置情報の無い要素は飛ばす */
          }
        }
      }
      ent.sort((a, b) => a[0] - b[0]);
      const dedup: [number, number][] = [];
      for (const e of ent) {
        if (dedup.length && Math.abs(dedup[dedup.length - 1][0] - e[0]) < 1e-6) {
          dedup[dedup.length - 1][1] = Math.min(dedup[dedup.length - 1][1], e[1]);
        } else dedup.push([e[0], e[1]]);
      }
      out.push({ x, y, w: top.Size.width * unit, h: yb - y, entries: dedup });
    }
    return out;
  }

  async function ensureOsmd() {
    if (osmdRef.current) return osmdRef.current;
    const { OpenSheetMusicDisplay } = await import("opensheetmusicdisplay");
    const osmd = new OpenSheetMusicDisplay(hostRef.current!, {
      backend: "svg",
      autoResize: false,
      drawTitle: true,
      drawSubtitle: true,
      drawComposer: true,
      drawPartNames: false,
      drawMeasureNumbers: true,
      drawFingerings: true,
      fingeringPosition: "auto",
      followCursor: false,
      pageBackgroundColor: "#00000000",
      defaultColorMusic: "#111111",
    });
    osmd.EngravingRules.RenderMultipleRestMeasures = false;
    osmd.EngravingRules.PageLeftMargin = 3;
    osmd.EngravingRules.PageRightMargin = 3;
    osmdRef.current = osmd;
    return osmd;
  }

  async function render(reload: boolean) {
    const p = propsRef.current;
    if (!p.xml || !hostRef.current) return;
    const seq = ++renderSeq.current;
    setBusy(true);
    try {
      const osmd = await ensureOsmd();
      fitTitle(osmd, p.xml, hostRef.current?.clientWidth, p.zoom);
      if (reload) await osmd.load(p.xml);
      if (seq !== renderSeq.current) return;
      osmd.zoom = p.zoom;
      osmd.render();
      const bx = measureBoxes(osmd);
      boxesRef.current = bx;
      setBoxes(bx);
      setErr(null);
    } catch (e) {
      setErr((e as Error).message || String(e));
    } finally {
      if (seq === renderSeq.current) setBusy(false);
    }
  }

  useEffect(() => {
    void render(true);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [props.xml]);

  useEffect(() => {
    if (osmdRef.current) void render(false);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [props.zoom]);

  // 幅が変わったら組み直す
  useEffect(() => {
    const el = scrollRef.current;
    if (!el) return;
    let w = el.clientWidth;
    let t: ReturnType<typeof setTimeout>;
    const ro = new ResizeObserver(() => {
      if (Math.abs(el.clientWidth - w) < 8) return;
      w = el.clientWidth;
      clearTimeout(t);
      t = setTimeout(() => osmdRef.current && render(false), 250);
    });
    ro.observe(el);
    return () => {
      ro.disconnect();
      clearTimeout(t);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // 指定小節へスクロール
  useEffect(() => {
    const m = props.focusMeasure;
    const b = m != null ? boxesRef.current[m] : null;
    if (b && scrollRef.current) scrollRef.current.scrollTo({ top: Math.max(0, b.y - 60), behavior: "smooth" });
  }, [props.focusMeasure, boxes]);

  // 再生カーソル
  useEffect(() => {
    let raf = 0;
    let lastM = -1;
    const loop = () => {
      const p = propsRef.current;
      const beat = p.getBeat();
      const bl = p.bars;
      let m = 0;
      while (m + 1 < bl.length - 1 && bl[m + 1] <= beat + 1e-9) m++;
      const mLen = (bl[m + 1] ?? bl[m] + 4) - bl[m];
      const b = boxesRef.current[m];
      const cur = cursorRef.current;
      const hi = measureHiRef.current;
      if (b && cur && hi && b.w > 0) {
        const local = beat - bl[m];
        const ent = b.entries;
        let x = b.x;
        if (ent.length) {
          let i = 0;
          while (i + 1 < ent.length && ent[i + 1][0] <= local) i++;
          const [t0, x0] = ent[i][0] <= local ? ent[i] : [0, b.x + 4];
          const [t1, x1] = i + 1 < ent.length && ent[i][0] <= local ? ent[i + 1] : [mLen, b.x + b.w - 4];
          x = t1 > t0 ? x0 + ((local - t0) / (t1 - t0)) * (x1 - x0) : x0;
        }
        cur.style.transform = `translate(${x}px, ${b.y}px)`;
        cur.style.height = `${b.h}px`;
        cur.style.opacity = beat > 0 || p.playing ? "1" : "0";
        hi.style.transform = `translate(${b.x}px, ${b.y}px)`;
        hi.style.width = `${b.w}px`;
        hi.style.height = `${b.h}px`;
        hi.style.opacity = beat > 0 || p.playing ? "1" : "0";
        if (p.playing && m !== lastM && scrollRef.current) {
          const sc = scrollRef.current;
          if (b.y < sc.scrollTop + 10 || b.y + b.h > sc.scrollTop + sc.clientHeight - 10) {
            sc.scrollTo({ top: Math.max(0, b.y - sc.clientHeight * 0.25), behavior: "smooth" });
          }
        }
        lastM = m;
      }
      raf = requestAnimationFrame(loop);
    };
    raf = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(raf);
  }, []);

  function onClick(e: React.MouseEvent) {
    const host = hostRef.current;
    if (!host) return;
    const r = host.getBoundingClientRect();
    const x = e.clientX - r.left;
    const y = e.clientY - r.top;
    const bs = boxesRef.current;
    let best = -1;
    let bestD = Infinity;
    for (let i = 0; i < bs.length; i++) {
      const b = bs[i];
      if (!b.w) continue;
      if (y < b.y - 20 || y > b.y + b.h + 20) continue;
      const dx = x < b.x ? b.x - x : x > b.x + b.w ? x - (b.x + b.w) : 0;
      if (dx < bestD) {
        bestD = dx;
        best = i;
      }
    }
    if (best >= 0 && bestD < 40) props.onMeasureClick(best, e.shiftKey);
  }

  const mOf = (beat: number) => {
    const bl = props.bars;
    let m = 0;
    while (m + 1 < bl.length - 1 && bl[m + 1] <= beat + 1e-9) m++;
    return m;
  };
  const loopM = props.loop ? { a: mOf(props.loop.start), b: mOf(props.loop.end - 1e-3) } : null;

  return (
    <div ref={scrollRef} className="scroll-thin relative h-full overflow-auto rounded-xl">
      <div className="sheet-paper relative mx-auto min-h-full cursor-pointer" onClick={onClick}>
        <div ref={hostRef} className="relative" />
        <div className="pointer-events-none absolute inset-0">
          {boxes.map((b, i) =>
            b.w && (props.difficult.has(i) || (loopM && i >= loopM.a && i <= loopM.b)) ? (
              <div
                key={i}
                className="absolute rounded-md"
                style={{
                  left: b.x,
                  top: b.y,
                  width: b.w,
                  height: b.h,
                  background:
                    loopM && i >= loopM.a && i <= loopM.b
                      ? "rgba(99,102,241,0.13)"
                      : "rgba(244,63,94,0.07)",
                  boxShadow: props.difficult.has(i) ? "inset 0 -3px 0 rgba(244,63,94,0.55)" : undefined,
                }}
              />
            ) : null,
          )}
          <div ref={measureHiRef} className="absolute left-0 top-0 rounded-md bg-fuchsia-400/10 opacity-0 transition-opacity" />
          <div ref={cursorRef} className="absolute left-0 top-0 w-[2px] rounded bg-fuchsia-500 opacity-0 shadow-[0_0_8px_rgba(217,70,239,0.8)]" />
        </div>
      </div>
      {(busy || !props.xml) && (
        <div className="absolute right-3 top-3 flex items-center gap-2 rounded-full bg-black/60 px-3 py-1 text-xs text-white">
          <LoaderCircle className="size-3.5 animate-spin" /> 五線譜を浄書中…
        </div>
      )}
      {err && (
        <div className="absolute inset-x-3 top-3 rounded-lg bg-rose-600/90 px-3 py-2 text-xs text-white">楽譜の描画に失敗しました: {err}</div>
      )}
    </div>
  );
}
