export type Mode = "easy" | "standard" | "virtuoso";
export type Instrument = "piano" | "guitar";
export type Hand = "R" | "L";

export interface Note {
  id: string;
  pitch: number;
  /** 4分音符=1 の拍単位 */
  start: number;
  dur: number;
  vel: number;
  hand: Hand;
  finger?: number;
  fingerLocked?: boolean;
  ornament?: "trill";
  arpeggiate?: boolean;
  role?: string;
  /** ギター: 弦（1=1弦=いちばん高い 〜 6）とフレット */
  string?: number;
  fret?: number;
  stringLocked?: boolean;
}

export interface GuitarInfo {
  tuning: string;
  tuningLabel: string;
  open: number[];
  capo: number;
  options: { tuning?: string; capo?: number | "auto" };
}

export interface Chord {
  start: number;
  end: number;
  symbol: string;
  root: number;
  suffix: string;
  kind: string;
  degrees: [number, number][];
  bass: number | null;
}

export interface MeasureDifficulty {
  index: number;
  score: number;
  reasons: string[];
  hands: Record<Hand, null | {
    nps: number;
    maxChord: number;
    maxSpan: number;
    leapSpeed: number;
    fingerCost: number;
    black: number;
    tuplets: number;
  }>;
}

export interface DifficultySpan {
  startMeasure: number;
  endMeasure: number;
  peak: number;
  reasons: string[];
}

export interface Difficulty {
  overall: number;
  label: string;
  threshold: number;
  measures: MeasureDifficulty[];
  spans: DifficultySpan[];
  difficultMeasures: number[];
}

export interface ArrangeOptions {
  tempoScale?: number | "auto";
  /** 自動で拍を倍にしたとき "x2" */
  autoTempo?: string | null;
  bpm?: number | null;
  timeSig?: "auto" | "4/4" | "3/4" | "2/4" | "6/8" | "12/8";
  transpose?: number;
  easyKey?: boolean;
  leftHand?: string;
}

export interface Score {
  version: number;
  mode: Mode;
  modeLabel: string;
  title: string;
  artist: string;
  tempo: number;
  timeSignature: [number, number];
  /** 1小節の長さ（4分音符単位）。6/8 なら 3、12/8 なら 6 */
  measureBeats?: number;
  /** 1拍の長さ（4分音符単位）。単純拍子 1、複合拍子 1.5 */
  beatLen?: number;
  /** 小節の開始位置（4分音符単位）の列。最後は曲の終わり。途中に 2/4 などの短い小節があってよい */
  bars?: number[];
  key: { fifths: number; minor: boolean; en: string; ja: string };
  totalBeats: number;
  notes: Note[];
  chords: Chord[];
  pedals: { start: number; end: number }[];
  dynamics: { beat: number; mark: string }[];
  audio: { beatTimes: number[]; offsetBeats: number };
  arrange: ArrangeOptions;
  source: { kind: string; engine: string; songLike: boolean };
  difficulty: Difficulty;
  edited?: boolean;
  instrument?: Instrument;
  guitar?: GuitarInfo;
}

export type StepState = "pending" | "running" | "done" | "skipped" | "error";

export interface Job {
  id: string;
  created: number;
  updated: number;
  status: "queued" | "running" | "done" | "error" | "canceled";
  stage: string | null;
  progress: number;
  message: string;
  steps: Record<string, { label: string; state: StepState; detail: string }>;
  stepOrder: string[];
  request: {
    kind: "url" | "upload";
    url?: string;
    filename?: string;
    platform: string;
    mode: Mode;
    sourceType: string;
    separation: string;
    start?: number | null;
    end?: number | null;
  };
  meta: {
    title?: string;
    uploader?: string;
    duration?: number;
    thumbnail?: string;
    webpage_url?: string;
    platform?: string;
    analyzedDuration?: number;
  };
  result: null | {
    defaultMode: Mode;
    sourceKind: string;
    sourceKindLabel: string;
    engine: string;
    engineId?: string;
    separation: { quality: string; engine: string | null };
    target: string;
    hasTarget: boolean;
    bpm: number;
    key: { ja: string; en: string };
  };
  error: string | null;
  log: { t: number; text: string }[];
}

export interface JobSummary {
  id: string;
  created: number;
  status: Job["status"];
  progress: number;
  title: string;
  platform: string;
  thumbnail: string;
  mode: Mode;
  error: string | null;
}

export interface SystemInfo {
  version: string;
  /** ブラウザ版エンジン（インストール不要）で動いているとき true */
  browser?: boolean;
  device: string;
  gpu: string | null;
  ffmpeg: boolean;
  ytdlp: string | null;
  engines: { bytedance: { available: boolean; weights: boolean }; demucs: boolean; builtin: boolean; basicpitch?: boolean };
  limits: { maxAnalyzeSec: number; maxSourceSec: number; maxUploadMB: number };
  leftHand: { id: string; label: string }[];
  tunings: { id: string; label: string; open: number[] }[];
}

export interface ProbeResult {
  title: string;
  uploader: string;
  duration: number;
  thumbnail: string;
  platform: string;
  webpage_url: string;
}

export interface DisplayOptions {
  fingering: "difficult" | "all" | "none";
  chords: boolean;
  pedal: boolean;
  dynamics: boolean;
}
