"""音名の綴り・調の推定・コード認識。"""
from __future__ import annotations

import math

import numpy as np

FIFTHS_STEPS = "FCGDAEB"
BLACK = {1, 3, 6, 8, 10}

# Krumhansl-Kessler の調性プロファイル
KK_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
KK_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

JA_STEP = {"C": "ハ", "D": "ニ", "E": "ホ", "F": "ヘ", "G": "ト", "A": "イ", "B": "ロ"}


def is_black(midi: int) -> bool:
    return midi % 12 in BLACK


def pos_to_spelling(pos: int) -> tuple[str, int]:
    """五度圏上の位置（C=0, G=1, F=-1 …）→ (音名, 変化記号)。"""
    return FIFTHS_STEPS[(pos + 1) % 7], (pos + 1) // 7


def spell_pc(pc: int, key_fifths: int, minor: bool = False) -> tuple[str, int]:
    """調に合わせて半音階の音を綴る。長調は [k-4, k+7]、短調は [k-3, k+8] の窓から選ぶ。"""
    lo = key_fifths - (3 if minor else 4)
    for pos in range(lo, lo + 12):
        if (pos * 7) % 12 == pc % 12:
            return pos_to_spelling(pos)
    return "C", 0  # 到達しない


def spell_midi(midi: int, key_fifths: int, minor: bool = False) -> tuple[str, int, int]:
    step, alter = spell_pc(midi % 12, key_fifths, minor)
    octave = (midi - alter) // 12 - 1
    return step, alter, octave


def alter_symbol(alter: int, ascii_only: bool = False) -> str:
    if ascii_only:
        return {-2: "bb", -1: "b", 0: "", 1: "#", 2: "x"}[alter]
    return {-2: "𝄫", -1: "♭", 0: "", 1: "♯", 2: "𝄪"}[alter]


def note_name(midi: int, key_fifths: int = 0, minor: bool = False) -> str:
    step, alter, octave = spell_midi(midi, key_fifths, minor)
    return f"{step}{alter_symbol(alter, True)}{octave}"


def key_tonic_pos(fifths: int, minor: bool) -> int:
    return fifths + 3 if minor else fifths


def key_name(fifths: int, minor: bool) -> dict:
    step, alter = pos_to_spelling(key_tonic_pos(fifths, minor))
    en = f"{step}{alter_symbol(alter)} {'minor' if minor else 'major'}"
    ja_prefix = {1: "嬰", -1: "変"}.get(alter, "")
    ja = f"{ja_prefix}{JA_STEP[step]}{'短調' if minor else '長調'}"
    return {"en": en, "ja": ja}


def fifths_for_tonic(pc: int, minor: bool) -> int:
    """主音の音高クラスから調号（-6..+6）を決める。"""
    cands = []
    for pos in range(-10, 12):
        if (pos * 7) % 12 == pc % 12:
            f = pos - 3 if minor else pos
            if -7 <= f <= 7:
                cands.append(f)
    # 調号の少ない方。±6 の同点は ♭ 側（変ト長調・変ホ短調）
    cands.sort(key=lambda f: (abs(f), f > 0))
    return cands[0]


def estimate_key(notes: list[dict]) -> dict:
    """音符（pitch, 長さ, 強さ）から調を推定する。"""
    pcp = np.zeros(12)
    bass = np.zeros(12)
    for n in notes:
        w = max(n.get("end", n.get("off", 0)) - n.get("start", n.get("on", 0)), 0.05) * (n.get("vel", 80) / 127)
        pcp[n["pitch"] % 12] += w
        if n["pitch"] < 55:
            bass[n["pitch"] % 12] += w
    if pcp.sum() <= 0:
        return {"fifths": 0, "minor": False, "tonic": 0, "confidence": 0.0, **key_name(0, False)}
    prof = pcp / pcp.sum() + 0.35 * (bass / (bass.sum() + 1e-9))
    best = (-2.0, 0, False)
    scores = []
    for tonic in range(12):
        for minor, tpl in ((False, KK_MAJOR), (True, KK_MINOR)):
            r = float(np.corrcoef(prof, np.roll(tpl, tonic))[0, 1])
            scores.append(r)
            if r > best[0]:
                best = (r, tonic, minor)
    r, tonic, minor = best
    srt = sorted(scores, reverse=True)
    conf = float(max(0.0, srt[0] - srt[1]) * 4 + max(r, 0) * 0.5)
    fifths = fifths_for_tonic(tonic, minor)
    return {"fifths": fifths, "minor": minor, "tonic": tonic, "confidence": round(min(conf, 1.0), 3), **key_name(fifths, minor)}


def transpose_key(fifths: int, minor: bool, semitones: int) -> int:
    tonic_pc = (key_tonic_pos(fifths, minor) * 7) % 12
    return fifths_for_tonic((tonic_pc + semitones) % 12, minor)


# --------------------------------------------------------------------------
# コード認識
# --------------------------------------------------------------------------
# (表記, 構成音, レベル, MusicXML kind, 追加 degree [(度, 変化)])
# レベル 0=簡単（三和音） 1=標準（四和音・sus・6・add9） 2=超プロ（テンション）
CHORD_DEFS: list[tuple[str, tuple[int, ...], int, str, list[tuple[int, int]]]] = [
    ("", (0, 4, 7), 0, "major", []),
    ("m", (0, 3, 7), 0, "minor", []),
    ("dim", (0, 3, 6), 0, "diminished", []),
    ("aug", (0, 4, 8), 1, "augmented", []),
    ("sus4", (0, 5, 7), 1, "suspended-fourth", []),
    ("sus2", (0, 2, 7), 1, "suspended-second", []),
    ("7", (0, 4, 7, 10), 1, "dominant", []),
    ("maj7", (0, 4, 7, 11), 1, "major-seventh", []),
    ("m7", (0, 3, 7, 10), 1, "minor-seventh", []),
    ("m7(b5)", (0, 3, 6, 10), 1, "half-diminished", []),
    ("dim7", (0, 3, 6, 9), 1, "diminished-seventh", []),
    ("6", (0, 4, 7, 9), 1, "major-sixth", []),
    ("m6", (0, 3, 7, 9), 1, "minor-sixth", []),
    ("add9", (0, 2, 4, 7), 1, "major", [(9, 0)]),
    ("7sus4", (0, 5, 7, 10), 1, "suspended-fourth", [(7, -1)]),
    ("mM7", (0, 3, 7, 11), 2, "major-minor", []),
    ("9", (0, 2, 4, 7, 10), 2, "dominant-ninth", []),
    ("maj9", (0, 2, 4, 7, 11), 2, "major-ninth", []),
    ("m9", (0, 2, 3, 7, 10), 2, "minor-ninth", []),
    ("7(b9)", (0, 1, 4, 7, 10), 2, "dominant", [(9, -1)]),
    ("7(#9)", (0, 3, 4, 7, 10), 2, "dominant", [(9, 1)]),
    ("7(#11)", (0, 4, 6, 7, 10), 2, "dominant", [(11, 1)]),
    ("7(13)", (0, 4, 7, 9, 10), 2, "dominant-13th", []),
    ("m11", (0, 3, 5, 7, 10), 2, "minor-11th", []),
    ("maj7(#11)", (0, 4, 6, 7, 11), 2, "major-seventh", [(11, 1)]),
    ("m7(9)", (0, 2, 3, 7, 10), 2, "minor-ninth", []),
]


def _chord_templates(level: int):
    tpls = []
    for suffix, ivs, lv, kind, degrees in CHORD_DEFS:
        if lv > level:
            continue
        for root in range(12):
            v = np.zeros(12)
            for i, iv in enumerate(ivs):
                v[(root + iv) % 12] = 1.0 if i == 0 else (0.85 if iv in (3, 4, 7) else 0.7)
            v /= np.linalg.norm(v)
            tpls.append((root, suffix, ivs, kind, degrees, v))
    return tpls


LETTERS = "CDEFGAB"
NATURAL_PC = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
# ルートからの半音数 → 音名で何文字先か（3度は3度、5度は5度として綴る）
INTERVAL_STEPS = {0: 0, 1: 1, 2: 1, 3: 2, 4: 2, 5: 3, 6: 4, 7: 4, 8: 4, 9: 5, 10: 6, 11: 6}


def spell_from_root(root_step: str, root_alter: int, interval: int) -> tuple[str, int]:
    """コードの構成音を、ルートからの音程で綴る（G7/B が G7/C♭ にならないように）。"""
    interval %= 12
    step = LETTERS[(LETTERS.index(root_step) + INTERVAL_STEPS[interval]) % 7]
    target = (NATURAL_PC[root_step] + root_alter + interval) % 12
    alter = (target - NATURAL_PC[step] + 6) % 12 - 6
    if abs(alter) > 2:
        return spell_pc(target, 0, False)
    return step, alter


def chord_symbol(root: int, suffix: str, bass: int | None, key_fifths: int, minor: bool) -> str:
    step, alter = spell_pc(root, key_fifths, minor)
    s = f"{step}{alter_symbol(alter, True)}{suffix}"
    if bass is not None and bass != root:
        bstep, balter = spell_from_root(step, alter, bass - root)
        s += f"/{bstep}{alter_symbol(balter, True)}"
    return s


def recognize_chords(
    notes: list[dict],
    total_beats: int,
    beats_per_measure: int,
    level: int,
    key_fifths: int,
    minor: bool,
    offset_beats: int = 0,
    beat_len: float = 1.0,
    bar_starts: set[float] | None = None,
) -> list[dict]:
    """拍ごとのピッチクラス分布にテンプレートを当て、ビタビで滑らかにつなぐ。

    notes の start/end と total_beats・beats_per_measure は4分音符単位。
    beat_len（複合拍子は 1.5）ごとの区切りで判定し、戻り値の start/end も4分音符単位。
    """
    if total_beats <= 0:
        return []
    if beat_len != 1.0:
        scaled = [{**n, "start": n["start"] / beat_len, "end": n["end"] / beat_len} for n in notes]
        res = recognize_chords(
            scaled,
            int(math.ceil(total_beats / beat_len - 1e-9)),
            int(round(beats_per_measure / beat_len)),
            level,
            key_fifths,
            minor,
            offset_beats,
        )
        for ch in res:
            ch["start"] *= beat_len
            ch["end"] = min(ch["end"] * beat_len, float(total_beats))
        return res
    chroma = np.zeros((total_beats, 12))
    bass = np.zeros((total_beats, 12))
    low = np.full(total_beats, 200.0)
    for n in notes:
        s, e = n["start"], n["end"]
        if e <= s:
            continue
        w_vel = (n.get("vel", 80) / 127) ** 0.7
        # 高すぎる音（倍音の聴き違いが多い）は軽く
        w_reg = 0.35 if n["pitch"] >= 88 else (0.7 if n["pitch"] >= 81 else 1.0)
        for b in range(max(0, int(math.floor(s))), min(total_beats, int(math.ceil(e)))):
            ov = min(e, b + 1) - max(s, b)
            if ov <= 0:
                continue
            # 鳴り始めの拍を重く（減衰する楽器なので）
            w = ov * w_vel * w_reg * (1.4 if s >= b - 1e-6 else 0.8)
            chroma[b, n["pitch"] % 12] += w
            if n["pitch"] < 60:
                bass[b, n["pitch"] % 12] += w * (1.0 + (60 - n["pitch"]) / 24)
                low[b] = min(low[b], n["pitch"])
    tpls = _chord_templates(level)
    n_t = len(tpls)
    energy = chroma.sum(axis=1)
    # 事前確率：よく使うコードの種類、調の音階に収まるコードを少し優遇する
    tonic = (key_tonic_pos(key_fifths, minor) * 7) % 12
    scale = {(tonic + i) % 12 for i in ((0, 2, 3, 5, 7, 8, 10, 11) if minor else (0, 2, 4, 5, 7, 9, 11))}
    type_prior = {"": 0.05, "m": 0.045, "7": 0.02, "m7": 0.015, "maj7": 0.01}
    prior = np.array(
        [
            type_prior.get(suffix, -0.035 if suffix in ("sus4", "sus2", "aug", "dim", "m7(b5)", "dim7") else -0.01)
            + (0.035 if all((root + iv) % 12 in scale for iv in ivs) else -0.01)
            for (root, suffix, ivs, _k, _d, _v) in tpls
        ]
    )
    emis = np.zeros((total_beats, n_t + 1))  # 最後の列 = コードなし (N.C.)
    for b in range(total_beats):
        if energy[b] < 1e-6:
            emis[b, :n_t] = -1.0
            emis[b, n_t] = 0.6
            continue
        c = chroma[b] / (np.linalg.norm(chroma[b]) + 1e-9)
        bv = bass[b] / (bass[b].sum() + 1e-9) if bass[b].sum() > 0 else None
        for i, (root, suffix, ivs, _k, _d, v) in enumerate(tpls):
            sc = float(c @ v)
            if bv is not None:
                sc += 0.18 * bv[root]
            sc -= 0.025 * max(0, len(ivs) - 3)
            emis[b, i] = sc + prior[i]
        emis[b, n_t] = 0.15

    # ビタビ：小節頭での変化は安く、拍の途中ほど高く
    n_s = n_t + 1
    score = emis[0].copy()
    back = np.zeros((total_beats, n_s), dtype=np.int32)
    for b in range(1, total_beats):
        if bar_starts is not None:
            # 小節マップがあるとき：直近の小節頭からの位置で判定
            prev = max((s for s in bar_starts if s <= b + 1e-6), default=0.0)
            nxt = min((s for s in bar_starts if s > b + 1e-6), default=prev + beats_per_measure)
            pos = int(round(b - prev))
            span = int(round(nxt - prev))
        else:
            pos = (b - offset_beats) % beats_per_measure if beats_per_measure else 0
            span = beats_per_measure
        if pos == 0:
            pen = 0.06
        elif span in (4, 6) and pos == span // 2:
            pen = 0.12
        else:
            pen = 0.30
        best_prev = int(np.argmax(score))
        stay = score
        switch = score[best_prev] - pen
        new = np.where(stay >= switch, stay, switch)
        back[b] = np.where(stay >= switch, np.arange(n_s), best_prev)
        score = new + emis[b]
    path = np.zeros(total_beats, dtype=np.int32)
    path[-1] = int(np.argmax(score))
    for b in range(total_beats - 1, 0, -1):
        path[b - 1] = back[b, path[b]]

    chords: list[dict] = []
    b = 0
    while b < total_beats:
        st = path[b]
        e = b
        while e + 1 < total_beats and path[e + 1] == st:
            e += 1
        if st < n_t:
            root, suffix, ivs, kind, degrees, _ = tpls[st]
            bsum = bass[b : e + 1].sum(axis=0)
            bass_pc = int(np.argmax(bsum)) if bsum.sum() > 0 else None
            chord_pcs = {(root + iv) % 12 for iv in ivs}
            if bass_pc is not None and (bass_pc == root or bass_pc not in chord_pcs):
                bass_pc = None
            if level == 0:
                bass_pc = None  # 簡単モードは分数コードを出さない
            chords.append(
                {
                    "start": float(b),
                    "end": float(e + 1),
                    "root": root,
                    "suffix": suffix,
                    "kind": kind,
                    "degrees": degrees,
                    "intervals": list(ivs),
                    "bass": bass_pc,
                    "symbol": chord_symbol(root, suffix, bass_pc, key_fifths, minor),
                }
            )
        b = e + 1
    return chords


def chord_at(chords: list[dict], beat: float) -> dict | None:
    for c in chords:
        if c["start"] <= beat + 1e-6 < c["end"]:
            return c
    return None
