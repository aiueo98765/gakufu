import type { Note, Score } from "./types";

const FIFTHS_STEPS = "FCGDAEB";
const BLACK = new Set([1, 3, 6, 8, 10]);

export const isBlack = (midi: number) => BLACK.has(((midi % 12) + 12) % 12);

function posToSpelling(pos: number): [string, number] {
  const step = FIFTHS_STEPS[(((pos + 1) % 7) + 7) % 7];
  const alter = Math.floor((pos + 1) / 7);
  return [step, alter];
}

/** 調に合わせた音名（バックエンドの theory.spell_pc と同じ規則） */
export function spell(midi: number, fifths = 0, minor = false): { step: string; alter: number; octave: number } {
  const pc = ((midi % 12) + 12) % 12;
  const lo = fifths - (minor ? 3 : 4);
  for (let pos = lo; pos < lo + 12; pos++) {
    if ((((pos * 7) % 12) + 12) % 12 === pc) {
      const [step, alter] = posToSpelling(pos);
      return { step, alter, octave: Math.floor((midi - alter) / 12) - 1 };
    }
  }
  return { step: "C", alter: 0, octave: 4 };
}

const ACC = { "-2": "𝄫", "-1": "♭", "0": "", "1": "♯", "2": "𝄪" } as Record<string, string>;

export function noteName(midi: number, fifths = 0, minor = false): string {
  const s = spell(midi, fifths, minor);
  return `${s.step}${ACC[String(s.alter)]}${s.octave}`;
}

const JA = { C: "ド", D: "レ", E: "ミ", F: "ファ", G: "ソ", A: "ラ", B: "シ" } as Record<string, string>;
export function noteNameJa(midi: number, fifths = 0, minor = false): string {
  const s = spell(midi, fifths, minor);
  return `${JA[s.step]}${ACC[String(s.alter)]}`;
}

export function prettyChord(sym: string): string {
  return sym.replace(/([A-G])b/g, "$1♭").replace(/([A-G])#/g, "$1♯").replace(/\(b/g, "(♭").replace(/\(#/g, "(♯");
}

/** 1小節の長さ（4分音符単位） */
export const beatsPerMeasure = (s: Score) => s.measureBeats ?? (s.timeSignature[0] * 4) / (s.timeSignature[1] || 4);
/** 1拍の長さ（4分音符単位）。複合拍子は付点4分=1.5 */
export const beatLen = (s: Score) => s.beatLen ?? 1;

/** 小節の開始位置の列（最後は終わり）。古いデータは一定長として作る。音が終わりを越えたら延ばす */
export function barStarts(s: Score): number[] {
  const mb = beatsPerMeasure(s);
  let end = s.totalBeats || mb;
  for (const n of s.notes) end = Math.max(end, n.start + n.dur);
  let b: number[];
  if (s.bars && s.bars.length >= 2) b = [...s.bars];
  else {
    b = [0];
  }
  while (b[b.length - 1] < end - 1e-6) b.push(+(b[b.length - 1] + mb).toFixed(6));
  if (b.length < 2) b.push(mb);
  return b;
}

const barCache = new WeakMap<Score, number[]>();
export function bars(s: Score): number[] {
  let b = barCache.get(s);
  if (!b) {
    b = barStarts(s);
    barCache.set(s, b);
  }
  return b;
}

export function measureCount(s: Score): number {
  return Math.max(1, bars(s).length - 1);
}

/** 拍の位置 → 小節番号（0 から） */
export function measureIndexAt(beat: number, s: Score): number {
  const b = bars(s);
  let lo = 0;
  let hi = b.length - 1;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (b[mid] <= beat + 1e-9) lo = mid;
    else hi = mid;
  }
  return Math.min(lo, b.length - 2);
}

/** 小節番号（0 から）→ 開始位置。末尾を越えたら一定長で外挿 */
export function measureStart(m: number, s: Score): number {
  const b = bars(s);
  if (m < b.length) return b[Math.max(0, m)];
  return b[b.length - 1] + (m - (b.length - 1)) * beatsPerMeasure(s);
}

export function meterLabel(s: Score): string {
  return `${s.timeSignature[0]}/${s.timeSignature[1] || 4}`;
}

/** テンポ表示。複合拍子は付点4分音符で */
export function tempoLabel(s: Score, rate = 1): string {
  const bl = beatLen(s);
  return bl === 1.5 ? `♩.=${Math.round((s.tempo * rate) / 1.5)}` : `♩=${Math.round(s.tempo * rate)}`;
}

export function measureOf(beat: number, s: Score) {
  return Math.floor(beat / beatsPerMeasure(s) + 1e-9);
}

export function formatPos(beat: number, s: Score): string {
  const m = measureIndexAt(beat, s);
  const b = beat - measureStart(m, s);
  return `${m + 1}小節 ${(Math.floor(b / beatLen(s) + 1e-6) + 1).toString()}拍`;
}

export function formatSec(sec: number): string {
  if (!isFinite(sec) || sec < 0) sec = 0;
  const m = Math.floor(sec / 60);
  const s = Math.floor(sec % 60);
  return `${m}:${s.toString().padStart(2, "0")}`;
}

export function parseTime(text: string): number | null {
  const t = text.trim();
  if (!t) return null;
  if (/^\d+(\.\d+)?$/.test(t)) return parseFloat(t);
  const m = /^(\d+):(\d{1,2})(?:\.(\d+))?$/.exec(t);
  if (!m) return null;
  return parseInt(m[1], 10) * 60 + parseInt(m[2], 10) + (m[3] ? parseFloat(`0.${m[3]}`) : 0);
}

/** 楽譜上の拍 → 元音声の秒（テンポ地図で補間） */
export function beatToSec(beat: number, s: Score): number {
  const bt = s.audio?.beatTimes;
  if (!bt || bt.length < 2) return (beat * 60) / s.tempo;
  if (beat <= 0) return bt[0] + beat * (bt[1] - bt[0]);
  const n = bt.length - 1;
  if (beat >= n) return bt[n] + (beat - n) * (bt[n] - bt[n - 1]);
  const i = Math.floor(beat);
  return bt[i] + (beat - i) * (bt[i + 1] - bt[i]);
}

export function secToBeat(sec: number, s: Score): number {
  const bt = s.audio?.beatTimes;
  if (!bt || bt.length < 2) return (sec * s.tempo) / 60;
  if (sec <= bt[0]) return (sec - bt[0]) / (bt[1] - bt[0]);
  const n = bt.length - 1;
  if (sec >= bt[n]) return n + (sec - bt[n]) / (bt[n] - bt[n - 1]);
  let lo = 0;
  let hi = n;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (bt[mid] <= sec) lo = mid;
    else hi = mid;
  }
  return lo + (sec - bt[lo]) / (bt[lo + 1] - bt[lo]);
}

export function scoreEnd(s: Score): number {
  let end = s.totalBeats;
  for (const n of s.notes) end = Math.max(end, n.start + n.dur);
  return end;
}

export function nextNoteId(notes: Note[]): string {
  let max = 0;
  for (const n of notes) {
    const v = parseInt(n.id.replace(/\D/g, ""), 10);
    if (!isNaN(v) && v > max) max = v;
  }
  return `n${max + 1}`;
}

export const GRID_OPTIONS = [
  { value: 1, label: "4分" },
  { value: 0.5, label: "8分" },
  { value: 1 / 3, label: "3連8分" },
  { value: 0.25, label: "16分" },
  { value: 1 / 6, label: "6連16分" },
  { value: 0.125, label: "32分" },
];

export function snap(beat: number, grid: number): number {
  const b = Math.floor(beat + 1e-9);
  const f = beat - b;
  return b + Math.round(f / grid) * grid;
}
