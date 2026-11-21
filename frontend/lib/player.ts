"use client";

import type { Hand, Score } from "./types";
import { bars as barsOf, beatLen, beatToSec, secToBeat } from "./music";

/**
 * 2種類の再生エンジンを同じ操作で扱う。
 *  - ScorePlayer : 楽譜をピアノ音源（Salamander Grand のサンプル）で鳴らす。Tone.js の Transport を使い、
 *                  テンポ変更・小節ループ・メトロノーム・片手ミュートができる。
 *  - AudioPlayer : 元の音声（または分離したピアノ音）をピッチを保ったままスロー再生する。
 */
export interface Engine {
  ready(): Promise<void>;
  setScore(score: Score): void;
  play(fromBeat: number): Promise<void>;
  pause(): void;
  stop(): void;
  seek(beat: number): void;
  isPlaying(): boolean;
  getBeat(): number;
  setRate(rate: number): void;
  setLoop(loop: { start: number; end: number } | null): void;
  setMetronome(on: boolean): void;
  setMuted(hand: Hand, muted: boolean): void;
  dispose(): void;
  onEnd?: () => void;
}

type ToneNS = typeof import("tone");
let tonePromise: Promise<ToneNS> | null = null;
export function loadTone(): Promise<ToneNS> {
  if (!tonePromise) tonePromise = import("tone");
  return tonePromise;
}

const SALAMANDER: Record<string, string> = {};
for (const oct of [0, 1, 2, 3, 4, 5, 6, 7]) {
  for (const [n, f] of [["A", "A"], ["C", "C"], ["D#", "Ds"], ["F#", "Fs"]] as const) {
    if (oct === 0 && n !== "A") continue;
    SALAMANDER[`${n}${oct}`] = `${f}${oct}.mp3`;
  }
}
SALAMANDER["C8"] = "C8.mp3";

let sharedInstrument: Promise<{ play: (midi: number, durSec: number, time: number, vel: number) => void; kind: string }> | null = null;

/** ピアノ音源を1度だけ読み込む。失敗したら軽いシンセで代用する。 */
export function getInstrument() {
  if (sharedInstrument) return sharedInstrument;
  sharedInstrument = (async () => {
    const Tone = await loadTone();
    const reverb = new Tone.Reverb({ decay: 2.2, wet: 0.16 }).toDestination();
    try {
      const sampler = await new Promise<import("tone").Sampler>((resolve, reject) => {
        const timer = setTimeout(() => reject(new Error("timeout")), 15000);
        const s = new Tone.Sampler({
          urls: SALAMANDER,
          baseUrl: "https://tonejs.github.io/audio/salamander/",
          release: 1.2,
          onload: () => {
            clearTimeout(timer);
            resolve(s);
          },
          onerror: (e) => {
            clearTimeout(timer);
            reject(e);
          },
        });
      });
      sampler.connect(reverb);
      sampler.volume.value = -4;
      return {
        kind: "Salamander Grand Piano",
        play: (midi: number, durSec: number, time: number, vel: number) =>
          sampler.triggerAttackRelease(Tone.Frequency(midi, "midi").toFrequency(), Math.max(0.05, durSec), time, vel),
      };
    } catch {
      const synth = new Tone.PolySynth(Tone.Synth, {
        oscillator: { type: "triangle8" },
        envelope: { attack: 0.004, decay: 0.9, sustain: 0.25, release: 0.9 },
      }).connect(reverb);
      synth.maxPolyphony = 64;
      synth.volume.value = -10;
      return {
        kind: "シンセ（ピアノ音源を読めませんでした）",
        play: (midi: number, durSec: number, time: number, vel: number) =>
          synth.triggerAttackRelease(Tone.Frequency(midi, "midi").toFrequency(), Math.max(0.05, durSec), time, vel),
      };
    }
  })();
  return sharedInstrument;
}

type Voice = { play: (midi: number, durSec: number, time: number, vel: number, string?: number) => void; kind: string };
export type GuitarTone = "nylon" | "steel";
let sharedGuitar: Promise<Voice> | null = null;
let guitarTone: GuitarTone = "nylon";

/**
 * ギターの生音サンプル（Tone.js 用楽器サンプル集 nbrosowsky/tonejs-instruments、CC BY 3.0）。
 * ナイロン弦（クラシック）とスチール弦（アコースティック）。読み込めなければ撥弦合成で代用する。
 */
const GUITAR_SETS: Record<GuitarTone, { base: string; files: string[] }> = {
  nylon: {
    base: "https://nbrosowsky.github.io/tonejs-instruments/samples/guitar-nylon/",
    files: ["B1", "D2", "E2", "Fs2", "Gs2", "A2", "B2", "Cs3", "D3", "E3", "Fs3", "G3", "A3", "B3", "Cs4", "Ds4", "E4", "Fs4", "Gs4", "A4", "B4", "Cs5", "D5", "E5", "Fs5", "G5", "Gs5", "A5", "As5"],
  },
  steel: {
    base: "https://nbrosowsky.github.io/tonejs-instruments/samples/guitar-acoustic/",
    files: ["D2", "Ds2", "E2", "F2", "Fs2", "G2", "Gs2", "A2", "As2", "B2", "C3", "Cs3", "D3", "Ds3", "E3", "F3", "Fs3", "G3", "Gs3", "A3", "As3", "B3", "C4", "Cs4", "D4", "Ds4", "E4", "F4", "Fs4", "G4", "Gs4", "A4", "As4", "B4", "C5", "Cs5", "D5"],
  },
};

/** ギターの音色を切り替える（次に鳴らすときから） */
export function setGuitarTone(tone: GuitarTone) {
  if (tone !== guitarTone) {
    guitarTone = tone;
    sharedGuitar = null;
  }
}

export function getGuitarTone(): GuitarTone {
  return guitarTone;
}

/** ギターの音：本物のギターの録音。同じ弦で次の音を弾いたら前の音は止まる（実物と同じ） */
export function getGuitar(): Promise<Voice> {
  if (sharedGuitar) return sharedGuitar;
  const tone = guitarTone;
  sharedGuitar = (async () => {
    const Tone = await loadTone();
    // 胴鳴りを少し足して、硬すぎる高域を丸める
    const eq = new Tone.EQ3({ low: 1.5, mid: 0, high: tone === "steel" ? -1 : -3, lowFrequency: 220, highFrequency: 3200 });
    const reverb = new Tone.Reverb({ decay: 1.8, preDelay: 0.01, wet: 0.18 }).toDestination();
    eq.connect(reverb);
    try {
      const set = GUITAR_SETS[tone];
      const urls: Record<string, string> = {};
      for (const f of set.files) urls[f.replace("s", "#")] = `${f}.mp3`;
      const sampler = await new Promise<import("tone").Sampler>((resolve, reject) => {
        const timer = setTimeout(() => reject(new Error("timeout")), 15000);
        const s = new Tone.Sampler({
          urls,
          baseUrl: set.base,
          release: 0.9,
          curve: "exponential",
          onload: () => {
            clearTimeout(timer);
            resolve(s);
          },
          onerror: (e) => {
            clearTimeout(timer);
            reject(e);
          },
        });
      });
      sampler.connect(eq);
      sampler.volume.value = tone === "steel" ? -3 : 0;
      const lastOnString = new Map<number, number>();
      return {
        kind: tone === "steel" ? "アコースティックギター（スチール弦）" : "クラシックギター（ナイロン弦）",
        play: (midi: number, durSec: number, time: number, vel: number, string?: number) => {
          const freq = Tone.Frequency(midi, "midi").toFrequency();
          if (string) {
            const prev = lastOnString.get(string);
            if (prev !== undefined && prev !== freq) sampler.triggerRelease(prev, time);
            lastOnString.set(string, freq);
          }
          // ギターは指を離しても少し響くので、短い音も最低限の長さで鳴らす
          sampler.triggerAttackRelease(freq, Math.max(0.18, durSec * 1.05), time, Math.min(1, 0.25 + 0.75 * vel));
        },
      };
    } catch {
      const body = new Tone.Filter({ frequency: 3200, type: "lowpass", rolloff: -12 });
      body.connect(eq);
      const make = () => {
        const p = new Tone.PluckSynth({ attackNoise: 1.2, dampening: 3600, resonance: 0.965, release: 0.6 });
        p.volume.value = -2;
        p.connect(body);
        return p;
      };
      const strings = Array.from({ length: 6 }, make);
      const spare = Array.from({ length: 4 }, make);
      let rr = 0;
      return {
        kind: "ギター（合成音：生音サンプルを読めませんでした）",
        play: (midi: number, durSec: number, time: number, vel: number, string?: number) => {
          const v = string && string >= 1 && string <= 6 ? strings[string - 1] : spare[rr++ % spare.length];
          v.volume.setValueAtTime(-14 + 14 * Math.min(1, vel), time);
          v.triggerAttackRelease(Tone.Frequency(midi, "midi").toFrequency(), Math.max(0.08, durSec), time);
        },
      };
    }
  })();
  return sharedGuitar;
}
export async function previewNote(midi: number, vel = 0.7, guitar = false, string?: number) {
  const Tone = await loadTone();
  await Tone.start();
  const inst = guitar ? await getGuitar() : await getInstrument();
  inst.play(midi, 0.6, Tone.now(), vel, string);
}

// ---------------------------------------------------------------------------
export class ScorePlayer implements Engine {
  private score: Score | null = null;
  private part: import("tone").Part | null = null;
  private clickId: number | null = null;
  private click: import("tone").MembraneSynth | null = null;
  private muted: Record<Hand, boolean> = { R: false, L: false };
  private metronome = false;
  private rate = 1;
  private loop: { start: number; end: number } | null = null;
  private playing = false;
  private endId: number | null = null;
  onEnd?: () => void;
  private readonly PPQ = 480;

  async ready() {
    const Tone = await loadTone();
    this.toneSync = Tone;
    await (this.score?.instrument === "guitar" ? getGuitar() : getInstrument());
    const tr = Tone.getTransport();
    tr.PPQ = this.PPQ;
    if (!this.click) {
      this.click = new Tone.MembraneSynth({ pitchDecay: 0.008, octaves: 2, envelope: { attack: 0.001, decay: 0.08, sustain: 0 } }).toDestination();
      this.click.volume.value = -8;
    }
  }

  setScore(score: Score) {
    this.score = score;
    void this.rebuild();
  }

  private tick(beat: number) {
    return `${Math.max(0, Math.round(beat * this.PPQ))}i`;
  }

  private async rebuild() {
    const Tone = await loadTone();
    const isGuitar = this.score?.instrument === "guitar";
    const inst: Voice = isGuitar ? await getGuitar() : await getInstrument();
    const tr = Tone.getTransport();
    if (this.part) {
      this.part.dispose();
      this.part = null;
    }
    if (this.endId !== null) {
      tr.clear(this.endId);
      this.endId = null;
    }
    const s = this.score;
    if (!s) return;
    tr.bpm.value = s.tempo * this.rate;
    // ギターの和音は低い弦から高い弦へ少しずつずらして鳴らす（ストローク／アルペジオ感）
    const strum = new Map<string, number>();
    if (isGuitar) {
      const groups = new Map<number, typeof s.notes>();
      for (const n of s.notes) {
        const k = Math.round(n.start * 1000);
        groups.set(k, [...(groups.get(k) ?? []), n]);
      }
      for (const g of groups.values()) {
        if (g.length < 2) continue;
        const step = g.length >= 4 ? 0.014 : 0.008;
        [...g].sort((a, b) => a.pitch - b.pitch).forEach((n, i) => strum.set(n.id, i * step));
      }
    }
    const events = s.notes.map((n) => ({
      time: this.tick(n.start),
      pitch: n.pitch,
      dur: n.dur,
      vel: n.vel,
      hand: n.hand,
      string: n.string,
      strum: strum.get(n.id) ?? 0,
    }));
    this.part = new Tone.Part((time, ev: { pitch: number; dur: number; vel: number; hand: Hand; string?: number; strum: number }) => {
      if (this.muted[ev.hand]) return;
      const secPerBeat = 60 / tr.bpm.value;
      inst.play(ev.pitch, ev.dur * secPerBeat, time + ev.strum, Math.min(1, Math.max(0.05, ev.vel / 110)), ev.string);
    }, events);
    this.part.start(0);
    let end = s.totalBeats;
    for (const n of s.notes) end = Math.max(end, n.start + n.dur);
    this.endId = tr.schedule(() => {
      if (!this.loop) {
        Tone.getDraw().schedule(() => {
          this.stop();
          this.onEnd?.();
        }, Tone.now());
      }
    }, this.tick(end + 0.5));
    this.applyLoop();
    this.applyMetronome();
  }

  private applyLoop() {
    void loadTone().then((Tone) => {
      const tr = Tone.getTransport();
      if (this.loop && this.loop.end > this.loop.start) {
        tr.loopStart = this.tick(this.loop.start);
        tr.loopEnd = this.tick(this.loop.end);
        tr.loop = true;
      } else {
        tr.loop = false;
      }
    });
  }

  private applyMetronome() {
    void loadTone().then((Tone) => {
      const tr = Tone.getTransport();
      if (this.clickId !== null) {
        tr.clear(this.clickId);
        this.clickId = null;
      }
      if (!this.metronome || !this.score) return;
      // 拍ごとに鳴らす（複合拍子は付点4分ごと）。小節頭だけ高い音
      const bl = beatLen(this.score);
      const step = Math.round(this.PPQ * bl);
      // 小節頭（小節マップ。2/4 などの短い小節も含む）だけ高い音
      const heads = new Set(barsOf(this.score).map((b) => Math.round((b * this.PPQ) / step)));
      this.clickId = tr.scheduleRepeat(
        (time) => {
          const beat = Math.round(tr.getTicksAtTime(time) / step);
          const head = heads.has(beat);
          this.click?.triggerAttackRelease(head ? "C6" : "G5", 0.03, time, head ? 0.9 : 0.5);
        },
        `${step}i`,
        0,
      );
    });
  }

  async play(fromBeat: number) {
    const Tone = await loadTone();
    await Tone.start();
    await this.ready();
    if (!this.part) await this.rebuild();
    const tr = Tone.getTransport();
    let start = fromBeat;
    if (this.loop && (start < this.loop.start || start >= this.loop.end)) start = this.loop.start;
    tr.stop();
    tr.ticks = Math.round(start * this.PPQ);
    tr.start("+0.05");
    this.playing = true;
  }

  pause() {
    void loadTone().then((Tone) => Tone.getTransport().pause());
    this.playing = false;
  }

  stop() {
    void loadTone().then((Tone) => {
      const tr = Tone.getTransport();
      tr.stop();
      tr.ticks = 0;
    });
    this.playing = false;
  }

  seek(beat: number) {
    void loadTone().then((Tone) => {
      Tone.getTransport().ticks = Math.round(Math.max(0, beat) * this.PPQ);
    });
  }

  isPlaying() {
    return this.playing;
  }

  private toneSync: ToneNS | null = null;
  getBeat() {
    if (!this.toneSync) {
      void loadTone().then((T) => (this.toneSync = T));
      return 0;
    }
    return this.toneSync.getTransport().ticks / this.PPQ;
  }

  setRate(rate: number) {
    this.rate = rate;
    void loadTone().then((Tone) => {
      if (this.score) Tone.getTransport().bpm.value = this.score.tempo * rate;
    });
  }

  setLoop(loop: { start: number; end: number } | null) {
    this.loop = loop;
    this.applyLoop();
  }

  setMetronome(on: boolean) {
    this.metronome = on;
    this.applyMetronome();
  }

  setMuted(hand: Hand, muted: boolean) {
    this.muted[hand] = muted;
  }

  dispose() {
    this.stop();
    this.part?.dispose();
    this.part = null;
    void loadTone().then((Tone) => {
      const tr = Tone.getTransport();
      if (this.clickId !== null) tr.clear(this.clickId);
      if (this.endId !== null) tr.clear(this.endId);
      tr.loop = false;
    });
  }
}

// ---------------------------------------------------------------------------
export class AudioPlayer implements Engine {
  private el: HTMLAudioElement;
  private score: Score | null = null;
  private loop: { start: number; end: number } | null = null;
  private raf = 0;
  private clickOn = false;
  private lastClickBeat = -1;
  /** 楽譜の 0 秒が、音声ファイルの何秒目か（ブラウザ版は元のファイルをそのまま再生するので、切り出した分ずれる） */
  private offset = 0;
  private pending: Promise<void> | null = null;
  onEnd?: () => void;

  constructor(src: string | Promise<{ url: string; offset: number }>) {
    this.el = new Audio();
    this.el.preload = "auto";
    this.el.crossOrigin = "anonymous";
    if (typeof src === "string") this.el.src = src;
    else
      this.pending = src
        .then(({ url, offset }) => {
          this.offset = offset || 0;
          this.el.src = url;
        })
        .catch(() => undefined);
    (this.el as HTMLAudioElement & { preservesPitch: boolean }).preservesPitch = true;
    this.el.addEventListener("ended", () => {
      cancelAnimationFrame(this.raf);
      this.onEnd?.();
    });
  }

  private get time() {
    return this.el.currentTime - this.offset;
  }

  private set time(sec: number) {
    this.el.currentTime = Math.max(0, sec + this.offset);
  }

  async ready() {
    if (this.pending) await this.pending;
    return new Promise<void>((resolve) => {
      if (this.el.readyState >= 2) return resolve();
      const done = () => {
        this.el.removeEventListener("canplay", done);
        resolve();
      };
      this.el.addEventListener("canplay", done);
      this.el.load();
      setTimeout(resolve, 8000);
    });
  }

  setScore(score: Score) {
    this.score = score;
  }

  private watch = () => {
    if (this.score) {
      const beat = this.getBeat();
      if (this.loop && beat >= this.loop.end) {
        this.time = Math.max(0, beatToSec(this.loop.start, this.score));
      }
      if (this.clickOn) {
        const bl = beatLen(this.score);
        const b = Math.floor(beat / bl);
        if (b !== this.lastClickBeat && beat / bl - b < 0.12) {
          this.lastClickBeat = b;
          void this.clickAt(b);
        }
      }
    }
    this.raf = requestAnimationFrame(this.watch);
  };

  private async clickAt(beat: number) {
    if (!this.score) return;
    const Tone = await loadTone();
    const bl = beatLen(this.score);
    const accent = barsOf(this.score).some((b) => Math.abs(b / bl - beat) < 1e-6);
    const s = new Tone.MembraneSynth({ envelope: { attack: 0.001, decay: 0.06, sustain: 0 } }).toDestination();
    s.volume.value = -10;
    s.triggerAttackRelease(accent ? "C6" : "G5", 0.03, Tone.now(), accent ? 0.9 : 0.5);
    setTimeout(() => s.dispose(), 400);
  }

  async play(fromBeat: number) {
    if (!this.score) return;
    let start = fromBeat;
    if (this.loop && (start < this.loop.start || start >= this.loop.end)) start = this.loop.start;
    if (this.pending) await this.pending;
    this.time = Math.max(0, beatToSec(start, this.score));
    await this.el.play();
    cancelAnimationFrame(this.raf);
    this.raf = requestAnimationFrame(this.watch);
  }

  pause() {
    this.el.pause();
    cancelAnimationFrame(this.raf);
  }

  stop() {
    this.el.pause();
    this.time = 0;
    cancelAnimationFrame(this.raf);
  }

  seek(beat: number) {
    if (this.score) this.time = Math.max(0, beatToSec(beat, this.score));
  }

  isPlaying() {
    return !this.el.paused;
  }

  getBeat() {
    return this.score ? secToBeat(this.time, this.score) : 0;
  }

  setRate(rate: number) {
    this.el.playbackRate = rate;
  }

  setLoop(loop: { start: number; end: number } | null) {
    this.loop = loop;
  }

  setMetronome(on: boolean) {
    this.clickOn = on;
  }

  setMuted() {
    /* 元音声は片手だけ消せない */
  }

  dispose() {
    this.stop();
    this.el.removeAttribute("src");
    this.el.load();
  }
}
