import type { Note, Score } from "./types";

export const MAX_FRET = 22;

/** 弦（1=1弦）の開放弦の実音（カポ込み） */
export function openPitch(score: Score, string: number): number {
  const g = score.guitar;
  const open = g?.open ?? [40, 45, 50, 55, 59, 64];
  return open[6 - string] + (g?.capo ?? 0);
}

export function fretOn(score: Score, pitch: number, string: number): number | null {
  const f = pitch - openPitch(score, string);
  return f >= 0 && f <= MAX_FRET ? f : null;
}

/**
 * 編集で音の高さが変わったギターの音の弦・フレットを合わせ直す。
 * 今の弦で届けばその弦のまま、届かなければ同じ時刻の他の音と弦が重ならない中で
 * 直前のポジションに一番近い弦へ移す（細かい最適化は「再計算」でバックエンドが行う）。
 */
export function refitFrets(score: Score, notes: Note[]): Note[] {
  if (score.instrument !== "guitar") return notes;
  const byStart = new Map<number, Note[]>();
  for (const n of notes) {
    const k = Math.round(n.start * 1e6);
    byStart.set(k, [...(byStart.get(k) ?? []), n]);
  }
  let lastPos = 3;
  const out: Note[] = [];
  for (const n of [...notes].sort((a, b) => a.start - b.start || a.pitch - b.pitch)) {
    let m = n;
    const cur = n.string ? fretOn(score, n.pitch, n.string) : null;
    if (n.string && cur !== null) {
      if (cur !== n.fret) m = { ...n, fret: cur };
    } else {
      const used = new Set((byStart.get(Math.round(n.start * 1e6)) ?? []).filter((x) => x.id !== n.id && x.string).map((x) => x.string));
      let best: { s: number; f: number; d: number } | null = null;
      for (let s = 1; s <= 6; s++) {
        if (used.has(s)) continue;
        const f = fretOn(score, n.pitch, s);
        if (f === null) continue;
        const d = Math.abs((f || lastPos) - lastPos) + (f > 12 ? 2 : 0);
        if (!best || d < best.d) best = { s, f, d };
      }
      m = best ? { ...n, string: best.s, fret: best.f } : { ...n, string: undefined, fret: undefined };
    }
    if (m.fret && m.fret > 0) lastPos = m.fret;
    out.push(m);
  }
  const order = new Map(notes.map((n, i) => [n.id, i]));
  return out.sort((a, b) => (order.get(a.id) ?? 0) - (order.get(b.id) ?? 0));
}

export const STRING_NAMES = ["1弦", "2弦", "3弦", "4弦", "5弦", "6弦"];
