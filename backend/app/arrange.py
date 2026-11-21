"""難易度別アレンジ。

採譜した生データ（秒単位の音符）は1回だけ作り、モード切替のたびにここで
数百ミリ秒で作り直す。

- easy（簡単）    : 右手=単旋律メロディ（8分音符まで・弾きやすい音域へ移動）
                    左手=コードのルート / 5度 / 3和音（白玉）
- standard（標準）: 右手=メロディ+和音2音まで、左手=ベース+和音2音まで。
                    トリルは tr 記号に、前打音は省略、速すぎる走句は16分に間引く。
                    歌ものは左手を分散和音などの伴奏型で作る。
- virtuoso（超プロ）: 一切削らない。32分音符・6連符・3連符まで拍ごとに判定し、
                    両手10度超えの和音はアルペジオ記号、ペダル・強弱も付ける。
"""
from __future__ import annotations

import math
from copy import deepcopy

import numpy as np

from . import barmap
from . import bars as BARS
from . import denoise
from . import hands as H
from . import quantize as Q
from . import theory as T
from .difficulty import analyze as analyze_difficulty
from .fingering import assign_fingering
from .rhythm import TempoMap, detect_meter, rescale_beats, uniform_beats

MODES = ("easy", "standard", "virtuoso")
# 計算方式の版。上げると、編集されていない保存済みの楽譜は次に開いたとき作り直される。
# 3: 音符に合わせた拍・雑音除去・テンポに応じた量子化・複合拍子・ギター
ARRANGE_VERSION = 6  # 5: 小節マップ・Beat This!  6: 左手が空ならコードで補う
MODE_LABEL = {"easy": "簡単", "standard": "標準", "virtuoso": "超プロ (Virtuoso)"}

PARAMS = {
    # subdiv: 単純拍子（拍=4分）で許す拍の分割 / subdiv_c: 複合拍子（拍=付点4分）
    "easy": {"min_vel": 30, "min_sec": 0.08, "subdiv": [1, 2], "default_sub": 2, "subdiv_c": [1, 3], "default_sub_c": 3, "chord_level": 0},
    "standard": {"min_vel": 22, "min_sec": 0.05, "subdiv": [1, 2, 4, 3], "default_sub": 4, "subdiv_c": [1, 3, 6], "default_sub_c": 6, "chord_level": 1},
    "virtuoso": {"min_vel": 6, "min_sec": 0.02, "subdiv": [1, 2, 4, 8, 3, 6], "default_sub": 4, "subdiv_c": [1, 3, 6, 12, 2, 4], "default_sub_c": 6, "chord_level": 2},
}
TIME_SIGS = ("auto", "4/4", "3/4", "2/4", "6/8", "12/8")

DEFAULT_OPTIONS = {
    "tempoScale": "auto",  # auto / 1 / 2 / 0.5（テンポの倍・半分の取り違えを直す。auto は細かすぎれば×2）
    "bpm": None,  # 指定すると一定テンポのグリッドで量子化
    "timeSig": "auto",  # auto / 4/4 / 3/4 / 2/4 / 6/8 / 12/8
    "transpose": 0,  # 半音
    "easyKey": False,  # ハ長調/イ短調に移調（主に簡単モード）
    "leftHand": "auto",  # auto / original / root / fifth / triad / block / broken / alberti
}

LEFT_HAND_LABEL = {
    "auto": "おまかせ",
    "original": "原曲どおり",
    "root": "ルート音のみ",
    "fifth": "ルート＋5度",
    "triad": "3和音",
    "block": "ベース＋和音",
    "broken": "分散和音（8分）",
    "alberti": "アルベルティ・バス",
}


# --------------------------------------------------------------------------
# 入口
# --------------------------------------------------------------------------
def arrange(raw: dict, analysis: dict, mode: str, options: dict | None = None, meta: dict | None = None) -> dict:
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode}")
    P = PARAMS[mode]
    opts = {**DEFAULT_OPTIONS, **{k: v for k, v in (options or {}).items() if k in DEFAULT_OPTIONS}}
    meta = meta or {}

    raw_notes = [dict(n) for n in raw["notes"]]
    # 雑音（実際には弾かれていない音）を除く。超プロでも雑音は残さない
    raw_notes, noise_stats = denoise.clean(raw_notes, mode)
    kind = raw.get("sourceKind", "piano")
    song_like = kind == "song" and any(n.get("role") == "melody" for n in raw_notes)

    # ---- テンポ地図と拍子 ----
    beats = list(analysis["beats"])
    forced_ts = None if opts["timeSig"] == "auto" else str(opts["timeSig"])
    auto_tempo = None
    if opts.get("bpm"):
        beats = uniform_beats(float(opts["bpm"]), analysis["duration"], analysis.get("firstOnset", 0.0))
    elif str(opts.get("tempoScale")) in ("2", "0.5", "2.0"):
        beats = rescale_beats(beats, float(opts["tempoScale"]))
    elif str(opts.get("tempoScale", "auto")) == "auto" and forced_ts not in ("6/8", "12/8") and _too_fine(beats, raw_notes):
        # 拍が遅く取れていて、音の多くが32分・6連の細かさになる → 拍を倍にして16分で書く
        beats = rescale_beats(beats, 2)
        auto_tempo = "x2"
    meter = detect_meter(beats, raw_notes, forced_ts)
    beat_len = float(meter["beatLen"])  # 1拍 = 4分音符いくつ分（複合拍子は 1.5）
    per_bar = int(meter["perBar"])  # 1小節の拍数
    bpm_m = int(round(per_bar * beat_len))  # 1小節の長さ（4分音符単位）
    ts_num, ts_den = (int(x) for x in meter["ts"].split("/"))
    offset = (per_bar - meter["phase"]) % per_bar
    tm = TempoMap(meter["beats"], offset)

    notes = []
    for n in raw_notes:
        if n.get("vel", 80) < P["min_vel"] or n["off"] - n["on"] < P["min_sec"]:
            continue
        m = dict(n)
        m["start"] = float(tm.sec_to_beat(n["on"]))
        m["end"] = float(tm.sec_to_beat(n["off"]))
        notes.append(m)
    if notes:
        # 最初の音が1小節目に入るよう、小節単位で全体をずらす（前に空の小節を作らない）
        lowest = min(n["start"] for n in notes)
        shift = -int(math.floor((lowest + 0.25) / per_bar)) * per_bar
        if shift:
            offset += shift
            tm = TempoMap(meter["beats"], offset)
            for n in notes:
                n["start"] += shift
                n["end"] += shift
        for n in notes:
            # ここから先はすべて4分音符=1 の単位で扱う
            n["start"] = max(0.0, n["start"]) * beat_len
            n["end"] = max(0.0, n["end"]) * beat_len
    key = dict(analysis["key"])
    minor = bool(key["minor"])
    fifths = int(key["fifths"])

    total_beats = _total_beats(notes, bpm_m)
    chords = T.recognize_chords(
        [n for n in notes if n.get("role") != "melody" or not song_like],
        total_beats,
        bpm_m,
        P["chord_level"],
        fifths,
        minor,
        0,
        beat_len=beat_len,
    )
    # コードの変わり目は小節頭に来やすい。変わり目が小節頭にそろう位置へ小節線をずらす。
    best_k = _best_bar_phase(chords, per_bar, beat_len, bpm_m)
    if best_k:
        add = (per_bar - best_k) * beat_len
        offset += per_bar - best_k
        for n in notes:
            n["start"] += add
            n["end"] += add
        lowest = min((n["start"] for n in notes), default=0.0)
        if lowest >= bpm_m - 0.25:  # 先頭が空の小節になったら1小節詰める
            k = int(math.floor((lowest + 0.25) / bpm_m))
            offset -= k * per_bar
            for n in notes:
                n["start"] -= k * bpm_m
                n["end"] -= k * bpm_m
        tm = TempoMap(meter["beats"], offset)
        total_beats = _total_beats(notes, bpm_m)
        chords = T.recognize_chords(
            [n for n in notes if n.get("role") != "melody" or not song_like],
            total_beats,
            bpm_m,
            P["chord_level"],
            fifths,
            minor,
            0,
            beat_len=beat_len,
        )
    compound = beat_len != 1.0

    # ---- 小節マップ：途中に 2/4・3/4 の短い小節が挟まる曲でも小節線がずれ続けないように ----
    bars = BARS.regular(total_beats, bpm_m)
    if not compound and per_bar in (3, 4) and notes:
        tb = np.asarray(meter["beats"], dtype=float)
        dnn_db = np.asarray(analysis.get("dnnDownbeats") or [], dtype=float)
        ev = barmap.downbeat_evidence(tb, raw_notes, dnn_db if dnn_db.size else None)
        idx = barmap.bar_starts(ev, per_bar, (-offset) % per_bar)
        starts = [(i + offset) * beat_len for i in idx if (i + offset) * beat_len > 0.5]
        cand = BARS.normalize(starts, total_beats, bpm_m)
        # 先頭が音の無い小節になったら、最初の音を含む小節が1小節目になるよう詰める
        lowest = min(n["start"] for n in notes)
        k = BARS.index(lowest + 0.25, cand)
        if k > 0:
            d = cand[k]
            offset -= int(round(d / beat_len))
            for n in notes:
                n["start"] -= d
                n["end"] -= d
            cand = [round(b - d, 6) for b in cand[k:]]
            tm = TempoMap(meter["beats"], offset)
        total_beats = max(_total_beats(notes, bpm_m), int(math.ceil(cand[-1])))
        bars = BARS.normalize(cand[1:-1], total_beats, bpm_m)
        chords = T.recognize_chords(
            [n for n in notes if n.get("role") != "melody" or not song_like],
            int(math.ceil(bars[-1])),
            bpm_m,
            P["chord_level"],
            fifths,
            minor,
            0,
            beat_len=beat_len,
            bar_starts=set(bars[:-1]),
        )

    ctx = {
        "bars": bars,
        "mode": mode,
        "P": P,
        "opts": opts,
        "bpm_m": bpm_m,
        "total_beats": total_beats,
        "chords": chords,
        "song_like": song_like,
        "kind": kind,
        "fifths": fifths,
        "minor": minor,
        "beat_len": beat_len,
        "compound": compound,
        "subdiv": P["subdiv_c"] if compound else P["subdiv"],
        "beat_sec": float(np.median(np.diff(np.asarray(meter["beats"], dtype=float)))),
        "default_sub": P["default_sub_c"] if compound else P["default_sub"],
    }
    if mode == "virtuoso":
        out = _arrange_full(notes, ctx)
    elif mode == "standard":
        out = _arrange_standard(notes, ctx)
    else:
        out = _arrange_easy(notes, ctx)

    # ---- 左手がほぼ空（アカペラ・ギターの高音だけ等）なら、コード進行から左手を補う ----
    lh = sum(1 for n in out if n.get("hand") == "L")
    if out and lh < 0.15 * len(out) and chords and opts.get("leftHand") != "original":
        pattern = opts.get("leftHand") if opts.get("leftHand") not in (None, "auto", "original") else (
            "root" if mode == "easy" else "fifth" if mode == "standard" else "block"
        )
        out = [n for n in out if n.get("hand") != "L"] + _left_from_chords(ctx, pattern)

    # ---- 移調 ----
    shift = int(opts.get("transpose") or 0)
    if opts.get("easyKey"):
        tonic = (T.key_tonic_pos(fifths, minor) * 7) % 12
        target = 9 if minor else 0
        d = (target - tonic) % 12
        shift += d - 12 if d > 6 else d
    if shift:
        for n in out:
            n["pitch"] += shift
        new_fifths = T.transpose_key(fifths, minor, shift)
        for c in chords:
            c["root"] = (c["root"] + shift) % 12
            if c.get("bass") is not None:
                c["bass"] = (c["bass"] + shift) % 12
            c["symbol"] = T.chord_symbol(c["root"], c["suffix"], c.get("bass"), new_fifths, minor)
        fifths = new_fifths
    for n in out:
        while n["pitch"] < 21:
            n["pitch"] += 12
        while n["pitch"] > 108:
            n["pitch"] -= 12

    out = Q.cleanup(out)
    _tidy_durations(out, mode, bpm_m)
    end = max((n["end"] for n in out), default=0.0)
    bars = BARS.normalize(bars[1:-1], max(end, bars[-1] if len(bars) > 1 else 0), bpm_m)
    ctx["bars"] = bars
    total_beats = int(math.ceil(bars[-1] - 1e-9))

    # ---- ペダル・強弱 ----
    pedals = _pedals(raw.get("pedals", []), tm, ctx) if mode != "easy" else []
    dynamics = _dynamics(out, bars) if mode != "easy" else []

    tempo = round(tm.bpm * beat_len)  # 4分音符のテンポ
    score_notes = []
    for i, n in enumerate(sorted(out, key=lambda n: (n["start"], n["pitch"]))):
        sn = {
            "id": f"n{i + 1}",
            "pitch": int(n["pitch"]),
            "start": round(float(n["start"]), 6),
            "dur": round(float(n["end"] - n["start"]), 6),
            "vel": int(np.clip(n.get("vel", 80), 1, 127)),
            "hand": n.get("hand", "R"),
        }
        for k in ("ornament", "arpeggiate", "role"):
            if n.get(k):
                sn[k] = n[k]
        score_notes.append(sn)

    score = {
        "version": 1,
        "engineVersion": ARRANGE_VERSION,
        "mode": mode,
        "modeLabel": MODE_LABEL[mode],
        "title": meta.get("title") or "無題",
        "artist": meta.get("uploader") or "",
        "tempo": tempo,
        "timeSignature": [ts_num, ts_den],
        "measureBeats": bpm_m,
        "bars": [round(b, 6) for b in bars],
        "beatLen": beat_len,
        "key": {"fifths": fifths, "minor": minor, **T.key_name(fifths, minor)},
        "totalBeats": total_beats,
        "notes": score_notes,
        "chords": [
            {
                "start": c["start"],
                "end": c["end"],
                "symbol": c["symbol"],
                "root": c["root"],
                "suffix": c["suffix"],
                "kind": c["kind"],
                "degrees": c["degrees"],
                "intervals": list(c.get("intervals") or []),
                "bass": c.get("bass"),
            }
            for c in chords
        ],
        "pedals": pedals,
        "dynamics": dynamics,
        "audio": {"beatTimes": tm.quarter_times(total_beats, beat_len), "offsetBeats": offset},
        "arrange": {**opts, "autoTempo": auto_tempo},
        "source": {
            "kind": kind,
            "engine": raw.get("engine"),
            "songLike": song_like,
            "noiseRemoved": noise_stats.get("removed", 0),
            "beatMethod": analysis.get("beatMethod"),
            "beatAlignment": analysis.get("beatAlignment"),
        },
    }
    assign_fingering(score)
    score["difficulty"] = analyze_difficulty(score)
    return score


TIDY = {
    # (埋める隙間の上限[拍], 長い音の終わりを揃える格子[拍], 次の音に食い込んだら切る上限[拍])
    "easy": (1.0, 1.0, 1.0),
    "standard": (1.0, 0.5, 0.5),
    "virtuoso": (0.5, 0.25, 0.25),
}


def _tidy_durations(notes: list[dict], mode: str, bpm_m: int) -> None:
    """読みやすさのための音価の整え。

    録音から取った離鍵は早めに出がちで、そのまま書くと白玉が「2分音符+16分休符…」に割れる。
    - 次の打鍵までの短い隙間は前の音を伸ばして埋める（休符の吸収）
    - 1拍以上の長い音は終わりを粗い格子に揃える
    - 次の打鍵に少しだけ食い込む音は切る（余計な2声部を作らない）
    """
    max_gap, long_grid, max_overlap = TIDY[mode]
    for hand in ("R", "L"):
        hn = [n for n in notes if n.get("hand") == hand]
        starts = sorted({n["start"] for n in hn})
        nxt_of = {s: (starts[i + 1] if i + 1 < len(starts) else None) for i, s in enumerate(starts)}
        for n in hn:
            nxt = nxt_of[n["start"]]
            d = n["end"] - n["start"]
            if long_grid and d >= 1 - 1e-9:
                e = round(n["end"] / long_grid) * long_grid
                if e > n["start"]:
                    n["end"] = e
            if nxt is None:
                continue
            gap = nxt - n["end"]
            cur = n["end"] - n["start"]
            # 標準・簡単では長い音ほど大きな隙間まで吸収する（白玉の後ろの休符を消す）
            allowed = max_gap if mode == "virtuoso" else min(2.0, max(max_gap, 0.75 * cur))
            if 0 < gap <= allowed + 1e-9 and gap <= cur * 1.01:
                n["end"] = nxt
            elif gap < 0 and -gap <= max_overlap + 1e-9 and (n["end"] - n["start"]) > -gap * 2:
                n["end"] = nxt


def _too_fine(beats: list[float], raw_notes: list[dict]) -> bool:
    """拍の取り方が遅すぎるか（倍のテンポで書く方が自然か）。

    - 倍にしても速すぎない（元が 105 未満）
    - 32分音符の位置（拍の 1/8, 3/8, …）に来る音が 6% 以上あり、かつ毎秒 9 音以上の密な曲
      （例: 166 BPM の J-POP を 83 BPM で取ると、16分が全部32分になる）
    """
    b = np.asarray(beats, dtype=float)
    # 倍にした結果が 180 BPM を超えるなら倍にしない（Instagram のピアノ曲が ♩=186 になった誤り）
    if b.size < 8 or 60.0 / float(np.median(np.diff(b))) * 2 > 180:
        return False
    pts = [n for n in raw_notes if n.get("role") != "melody"]
    if len(pts) < 40:
        return False
    tm = TempoMap(beats, 0)
    pos = np.array([float(tm.sec_to_beat(n["on"])) for n in pts])
    f = pos - np.floor(pos)
    k = np.round(f * 8)
    on32 = np.abs(f * 8 - k) < 0.25
    odd = float((on32 & (k % 2 == 1)).mean())
    span = max(1.0, max(n["on"] for n in pts) - min(n["on"] for n in pts))
    nps = len(pts) / span
    return odd >= 0.06 and nps >= 9


def _best_bar_phase(chords: list[dict], per_bar: int, beat_len: float, bpm_m: int) -> int:
    """コードの変わり目が一番多く小節頭に来る「拍のずらし量」を返す（0 ならそのまま）。"""
    changes = [c["start"] for c in chords[1:]]
    if len(changes) < 4 or per_bar < 2:
        return 0
    counts = []
    for k in range(per_bar):
        sh = k * beat_len
        counts.append(sum(1 for t in changes if abs((t - sh) / bpm_m - round((t - sh) / bpm_m)) < 1e-6))
    best = max(range(per_bar), key=lambda k: counts[k])
    if best and counts[best] >= max(3, 1.5 * counts[0] + 1):
        return best
    return 0


def _total_beats(notes: list[dict], bpm_m: int) -> int:
    end = max((n["end"] for n in notes), default=bpm_m)
    return max(bpm_m, int(math.ceil((end - 1e-6) / bpm_m)) * bpm_m)


# --------------------------------------------------------------------------
# 超プロ：完全再現
# --------------------------------------------------------------------------
def _arrange_full(notes: list[dict], ctx: dict) -> list[dict]:
    P = ctx["P"]
    if ctx["song_like"]:
        notes = _prepare_song_layers(notes, keep_harmony=True)
    H.split_hands(notes, key="on")
    out: list[dict] = []
    for hand in ("R", "L"):
        hn = [n for n in notes if n["hand"] == hand]
        out += Q.quantize(hn, ctx["subdiv"], ctx["default_sub"], beat_len=ctx["beat_len"], beat_sec=ctx["beat_sec"])
    out = Q.cleanup(out)
    _mark_big_chords(out)
    return out


def _prepare_song_layers(notes: list[dict], keep_harmony: bool) -> list[dict]:
    """歌ものの層（メロディ・ハーモニー・ベース）をピアノ用に整える。"""
    mel = [n for n in notes if n.get("role") == "melody"]
    bass = [n for n in notes if n.get("role") == "bass"]
    harm = [n for n in notes if n.get("role") not in ("melody", "bass")]
    _place_melody_octave(mel, 64, 77)
    for b in bass:
        while b["pitch"] > 55:
            b["pitch"] -= 12
        while b["pitch"] < 28:
            b["pitch"] += 12
    kept_harm = []
    if keep_harmony:
        mel_sorted = sorted(mel, key=lambda n: n["start"])
        starts = np.array([m["start"] for m in mel_sorted]) if mel_sorted else np.zeros(0)
        for h in harm:
            # メロディより上で鳴るハーモニーは主旋律を隠すので落とす
            i = int(np.searchsorted(starts, h["start"], side="right")) - 1
            cur = mel_sorted[i] if 0 <= i < len(mel_sorted) and mel_sorted[i]["end"] > h["start"] else None
            if cur and h["pitch"] >= cur["pitch"] - 1:
                continue
            if h["pitch"] > 80:
                continue
            kept_harm.append(h)
    return mel + kept_harm + bass


def _place_melody_octave(mel: list[dict], lo: int, hi: int) -> None:
    if not mel:
        return
    med = float(np.median([m["pitch"] for m in mel]))
    target = (lo + hi) / 2
    k = round((target - med) / 12)
    for m in mel:
        m["pitch"] += 12 * k


def _mark_big_chords(notes: list[dict]) -> None:
    """片手で届かない和音（10度超）にアルペジオ記号を付ける。"""
    groups: dict[tuple[str, float], list[dict]] = {}
    for n in notes:
        groups.setdefault((n["hand"], n["start"]), []).append(n)
    for g in groups.values():
        if len(g) < 2:
            continue
        ps = sorted(x["pitch"] for x in g)
        if ps[-1] - ps[0] > 16:
            for x in g:
                x["arpeggiate"] = True


# --------------------------------------------------------------------------
# 標準
# --------------------------------------------------------------------------
def _arrange_standard(notes: list[dict], ctx: dict) -> list[dict]:
    P = ctx["P"]
    lh_mode = ctx["opts"]["leftHand"]
    if ctx["song_like"]:
        mel = [n for n in notes if n.get("role") == "melody"]
        mel = _melody_line(mel, ctx, 64, 77)
        for m in mel:
            m["hand"] = "R"
        support = _rh_support(mel, ctx)
        pattern = "broken" if lh_mode in ("auto", "original") else lh_mode
        left = _left_from_chords(ctx, pattern)
        return mel + support + left

    notes = _remove_grace_notes(notes)
    H.split_hands(notes, key="on")
    out: list[dict] = []
    for hand in ("R", "L"):
        hn = [n for n in notes if n["hand"] == hand]
        qn = Q.quantize(hn, ctx["subdiv"], ctx["default_sub"], beat_len=ctx["beat_len"], beat_sec=ctx["beat_sec"])
        qn = Q.thin_collisions(qn)
        out += qn
    out = Q.cleanup(out)
    out = _merge_trills(out)
    out = [n for n in out if not (n.get("vel", 80) < 32 and n["end"] - n["start"] < 0.5)]
    out = _reduce_chords(out, max_notes=3, rh_span=12, lh_span=12)
    out = _merge_fast_repeats(out, 0.25)
    if lh_mode not in ("auto", "original"):
        out = [n for n in out if n["hand"] == "R"] + _left_from_chords(ctx, lh_mode)
    return out


def _remove_grace_notes(notes: list[dict]) -> list[dict]:
    """装飾音（短く弱い前打音）を落とす。量子化前の拍位置で判定。"""
    order = sorted(notes, key=lambda n: n["start"])
    drop = set()
    for i, n in enumerate(order):
        d = n["end"] - n["start"]
        if d > 0.14:
            continue
        for m in order[i + 1 : i + 6]:
            if m["start"] - n["start"] > 0.16:
                break
            if 0 < abs(m["pitch"] - n["pitch"]) <= 2 and (m["end"] - m["start"]) >= 3 * d:
                drop.add(id(n))
                break
    return [n for n in order if id(n) not in drop]


def _merge_trills(notes: list[dict]) -> list[dict]:
    """2音の速い交互反復（トリル）を主音1つ + tr 記号にまとめる。"""
    out = []
    for hand in ("R", "L"):
        hn = sorted([n for n in notes if n["hand"] == hand], key=lambda n: (n["start"], -n["pitch"]))
        # 同時に鳴る和音の中の音は対象外にするため、単音の並びだけで探す
        starts: dict[float, int] = {}
        for n in hn:
            starts[n["start"]] = starts.get(n["start"], 0) + 1
        singles = [n for n in hn if starts[n["start"]] == 1]
        used = set()
        i = 0
        while i < len(singles):
            j = i
            a = singles[i]["pitch"]
            b = None
            while j + 1 < len(singles):
                nx = singles[j + 1]
                if nx["start"] - singles[j]["start"] > 0.26:
                    break
                p = nx["pitch"]
                if b is None and 0 < abs(p - a) <= 2:
                    b = p
                expect = a if (j + 1 - i) % 2 == 0 else b
                if p != expect:
                    break
                j += 1
            if b is not None and j - i + 1 >= 5:
                main = dict(singles[i])
                main["pitch"] = min(a, b)
                main["end"] = singles[j]["end"]
                main["ornament"] = "trill"
                out.append(main)
                for k in range(i, j + 1):
                    used.add(id(singles[k]))
                i = j + 1
            else:
                i += 1
        out += [n for n in hn if id(n) not in used]
    return out


def _reduce_chords(notes: list[dict], max_notes: int, rh_span: int, lh_span: int) -> list[dict]:
    groups: dict[tuple[str, float], list[dict]] = {}
    for n in notes:
        groups.setdefault((n["hand"], n["start"]), []).append(n)
    out = []
    for (hand, _s), g in groups.items():
        if len(g) == 1:
            out += g
            continue
        if hand == "R":
            g.sort(key=lambda n: -n["pitch"])
            anchor = g[0]
            keep = [anchor]
            pcs = {anchor["pitch"] % 12}
            for n in g[1:]:
                if len(keep) >= max_notes:
                    break
                if anchor["pitch"] - n["pitch"] <= rh_span and n["pitch"] % 12 not in pcs:
                    keep.append(n)
                    pcs.add(n["pitch"] % 12)
        else:
            g.sort(key=lambda n: n["pitch"])
            anchor = g[0]
            keep = [anchor]
            pcs = {anchor["pitch"] % 12}
            for n in g[1:]:
                if len(keep) >= max_notes:
                    break
                if n["pitch"] - anchor["pitch"] <= lh_span and n["pitch"] % 12 not in pcs:
                    keep.append(n)
                    pcs.add(n["pitch"] % 12)
        out += keep
    return out


def _merge_fast_repeats(notes: list[dict], min_gap: float) -> list[dict]:
    """同じ音の速すぎる連打（トレモロ・同音連打）を1音にまとめる。"""
    groups: dict[tuple[str, int], list[dict]] = {}
    for n in notes:
        groups.setdefault((n["hand"], n["pitch"]), []).append(n)
    out = []
    for g in groups.values():
        g.sort(key=lambda n: n["start"])
        cur = None
        for n in g:
            if cur is not None and n["start"] - cur["start"] < min_gap - 1e-9 and n["start"] <= cur["end"] + 1e-9:
                cur["end"] = max(cur["end"], n["end"])
                continue
            cur = n
            out.append(n)
    return out


# --------------------------------------------------------------------------
# 簡単
# --------------------------------------------------------------------------
def _arrange_easy(notes: list[dict], ctx: dict) -> list[dict]:
    P = ctx["P"]
    if ctx["song_like"]:
        src = [n for n in notes if n.get("role") == "melody"]
    else:
        H.split_hands(notes, key="on")
        src = [n for n in notes if n["hand"] == "R"]
        if not src:
            src = notes
    mel = _melody_line(src, ctx, 60, 79, min_dur=0.5, min_vel=34, per_phrase=True)
    for m in mel:
        m["hand"] = "R"
    pattern = ctx["opts"]["leftHand"]
    if pattern in ("auto", "original", "broken", "alberti", "block"):
        pattern = "root" if pattern == "auto" or pattern == "original" else pattern
    left = _left_from_chords(ctx, pattern)
    return mel + left


def _melody_line(
    src: list[dict],
    ctx: dict,
    lo: int,
    hi: int,
    min_dur: float = 0.25,
    min_vel: int = 0,
    per_phrase: bool = False,
) -> list[dict]:
    """一番上の音をたどる単旋律（スカイライン）。重なりは次の音の頭で切る。"""
    src = [dict(n, hand="R") for n in src if n.get("vel", 80) >= min_vel]
    beat_len = ctx["beat_len"]
    q = Q.quantize(src, ctx["subdiv"], ctx["default_sub"], beat_len=beat_len, beat_sec=ctx["beat_sec"])
    by_start: dict[float, dict] = {}
    for n in q:
        cur = by_start.get(n["start"])
        if cur is None or n["pitch"] > cur["pitch"]:
            by_start[n["start"]] = n
    line = [by_start[s] for s in sorted(by_start)]
    # 直前の長い音の下で鳴る短い内声は旋律ではない
    filtered: list[dict] = []
    for n in line:
        if filtered:
            p = filtered[-1]
            if p["end"] > n["start"] + 1e-6 and n["pitch"] < p["pitch"] - 2 and n["end"] <= p["end"] + 1e-6:
                continue
        filtered.append(n)
    line = filtered
    for a, b in zip(line, line[1:]):
        if a["end"] > b["start"]:
            a["end"] = b["start"]
    line = [n for n in line if n["end"] - n["start"] >= min_dur - 1e-9 or (n["start"] % beat_len) < 1e-6]
    for n in line:
        if n["end"] - n["start"] < min_dur:
            n["end"] = n["start"] + min_dur
    for a, b in zip(line, line[1:]):
        if a["end"] > b["start"]:
            a["end"] = b["start"]
    line = [n for n in line if n["end"] - n["start"] > 1e-6]
    if per_phrase:
        _fit_phrases(line, lo, hi)
    else:
        _place_melody_octave(line, lo, hi)
    return line


def _fit_phrases(line: list[dict], lo: int, hi: int) -> None:
    """フレーズ（1拍以上の休みで区切る）ごとにオクターブを動かして弾きやすい音域に収める。"""
    if not line:
        return
    phrases: list[list[dict]] = [[line[0]]]
    for a, b in zip(line, line[1:]):
        if b["start"] - a["end"] >= 1.0 - 1e-6:
            phrases.append([b])
        else:
            phrases[-1].append(b)
    center = (lo + hi) / 2
    for ph in phrases:
        ps = np.array([n["pitch"] for n in ph])
        best = None
        for k in range(-4, 5):
            sh = ps + 12 * k
            outside = int(((sh < lo) | (sh > hi)).sum())
            cost = outside * 10 + abs(float(np.median(sh)) - center)
            if best is None or cost < best[0]:
                best = (cost, k)
        for n in ph:
            n["pitch"] += 12 * best[1]
            while n["pitch"] < lo:
                n["pitch"] += 12
            while n["pitch"] > hi:
                n["pitch"] -= 12


# --------------------------------------------------------------------------
# 伴奏型（コード → 左手）
# --------------------------------------------------------------------------
def _chord_segments(ctx: dict) -> list[tuple[float, float, dict]]:
    """コードの区間を小節線で切る（各小節で弾き直す）。"""
    bpm_m = ctx["bpm_m"]
    segs = []
    for c in ctx["chords"]:
        s, e = c["start"], c["end"]
        cur = s
        while cur < e - 1e-9:
            bars = ctx.get("bars") or BARS.regular(max(e, cur) + bpm_m, bpm_m)
            bar_end = bars[BARS.index(cur, bars) + 1] if BARS.index(cur, bars) + 1 < len(bars) else cur + bpm_m
            if bar_end <= cur + 1e-9:
                bar_end = cur + bpm_m
            nxt = min(e, bar_end)
            segs.append((cur, nxt, c))
            cur = nxt
    return segs


def _nearest_in_range(pc: int, lo: int, hi: int, prev: int | None) -> int:
    cands = [p for p in range(lo, hi + 1) if p % 12 == pc]
    if not cands:
        cands = [lo + ((pc - lo) % 12)]
    if prev is None:
        return cands[len(cands) // 2] if len(cands) > 1 else cands[0]
    return min(cands, key=lambda p: abs(p - prev))


def _third(c: dict) -> int:
    ivs = c["intervals"]
    for iv in (4, 3, 5, 2):
        if iv in ivs:
            return iv
    return 4


def _fifth(c: dict) -> int:
    ivs = c["intervals"]
    for iv in (7, 6, 8):
        if iv in ivs:
            return iv
    return 7


def _left_from_chords(ctx: dict, pattern: str) -> list[dict]:
    mode = ctx["mode"]
    bpm_m = ctx["bpm_m"]
    out: list[dict] = []
    prev_bass: int | None = None
    lo, hi = (48, 59) if mode == "easy" else (40, 52)
    for s, e, c in _chord_segments(ctx):
        L = e - s
        if L < 0.5 - 1e-9:
            continue
        root = c["root"]
        bass_pc = c["bass"] if c.get("bass") is not None else root
        bass = _nearest_in_range(bass_pc, lo, hi, prev_bass)
        prev_bass = bass
        third = (root + _third(c)) % 12
        fifth = (root + _fifth(c)) % 12
        vel = 70

        def add(p, st, en, v=vel):
            if en - st > 1e-6:
                out.append({"pitch": int(p), "start": st, "end": en, "vel": v, "hand": "L", "role": "accomp"})

        def above(pc, base, gap=1):
            p = base + ((pc - base) % 12)
            if p - base < gap:
                p += 12
            return p

        if pattern == "root" or L < 1 - 1e-9:
            add(bass, s, e)
        elif pattern == "fifth":
            add(bass, s, e)
            add(above(fifth, bass, 3), s, e)
        elif pattern == "triad":
            r = _nearest_in_range(root, 48, 59, None)
            add(r, s, e)
            add(above(third, r), s, e)
            add(above(fifth, r), s, e)
        elif pattern == "block":
            add(bass, s, e, 74)
            top_base = bass + 12 if bass + 12 <= 60 else bass
            for pc in (root, third, fifth):
                p = above(pc, top_base - 1)
                if p != bass:
                    add(p, s, min(e, s + max(1.0, min(2.0, L))), 62)
        elif pattern in ("broken", "alberti"):
            if pattern == "broken":
                seq = [bass, above(fifth, bass, 3), above(root, bass, 12), above(third, above(root, bass, 12))]
                if ctx.get("compound"):
                    # 複合拍子は 8分3つで1拍： ルート-5度-8度 / 10度-8度-5度
                    seq = [seq[0], seq[1], seq[2], seq[3], seq[2], seq[1]]
                elif L <= 2:
                    seq = [seq[0], seq[1], seq[2], seq[1]]
                else:
                    seq = [seq[0], seq[1], seq[2], seq[3], seq[2], seq[1], seq[2], seq[1]]
            else:
                r = _nearest_in_range(root, 48, 57, None)
                seq = [r, above(fifth, r), above(third, r), above(fifth, r)]
            n8 = int(round(L * 2))
            for i in range(n8):
                st = s + i * 0.5
                en = min(e, st + 0.5)
                add(seq[i % len(seq)], st, en, 72 if i == 0 else 58)
        else:
            add(bass, s, e)
    return out


def _rh_support(mel: list[dict], ctx: dict) -> list[dict]:
    """標準モードの歌もの：強拍の長いメロディ音の下にコードの音を1〜2音足す。"""
    bpm_m = ctx["bpm_m"]
    out = []
    for m in mel:
        if m["end"] - m["start"] < 1 - 1e-9:
            continue
        # 小節頭と、4拍以上の小節の真ん中（4/4 の3拍目、12/8 の3拍目）
        strong = BARS.strong(m["start"], ctx.get("bars") or BARS.regular(m["start"] + bpm_m, bpm_m))
        if not strong:
            continue
        c = T.chord_at(ctx["chords"], m["start"])
        if not c:
            continue
        pcs = {(c["root"] + iv) % 12 for iv in c["intervals"]}
        cands = [p for p in range(m["pitch"] - 12, m["pitch"] - 2) if p % 12 in pcs and p % 12 != m["pitch"] % 12]
        cands.sort(key=lambda p: m["pitch"] - p)
        picked: list[int] = []
        for p in cands:
            if all(abs(p - q) >= 3 for q in picked):
                picked.append(p)
            if len(picked) == 2:
                break
        for p in picked:
            out.append({"pitch": p, "start": m["start"], "end": m["end"], "vel": max(40, m.get("vel", 80) - 18), "hand": "R", "role": "accomp"})
    return out


# --------------------------------------------------------------------------
# ペダル・強弱
# --------------------------------------------------------------------------
def _pedals(raw_pedals: list[dict], tm: TempoMap, ctx: dict) -> list[dict]:
    out: list[dict] = []
    if ctx["song_like"]:
        for s, e, _c in _chord_segments(ctx):
            if e - s >= 1:
                out.append({"start": s, "end": e})
        return _merge_pedals(out)
    grid = 0.25 if ctx["mode"] == "virtuoso" else 0.5
    for p in raw_pedals:
        s = round(float(tm.sec_to_beat(p["on"])) * ctx["beat_len"] / grid) * grid
        e = round(float(tm.sec_to_beat(p["off"])) * ctx["beat_len"] / grid) * grid
        if e - s >= 0.5 and s >= 0:
            out.append({"start": s, "end": e})
    return _merge_pedals(out)


def _merge_pedals(ps: list[dict]) -> list[dict]:
    ps.sort(key=lambda p: p["start"])
    out: list[dict] = []
    for p in ps:
        if out and p["start"] < out[-1]["end"] - 1e-9:
            out[-1]["end"] = max(out[-1]["end"], p["end"])
        else:
            out.append(dict(p))
    return out


DYN_LEVELS = [(38, "pp"), (52, "p"), (66, "mp"), (80, "mf"), (96, "f"), (200, "ff")]


def _dynamics(notes: list[dict], bars: list[float]) -> list[dict]:
    n_meas = BARS.count(bars)
    acc = np.zeros(n_meas)
    cnt = np.zeros(n_meas)
    for n in notes:
        m = BARS.index(n["start"], bars)
        if 0 <= m < n_meas:
            acc[m] += n.get("vel", 80)
            cnt[m] += 1
    vel = np.where(cnt > 0, acc / np.maximum(cnt, 1), np.nan)
    marks = []
    for m in range(n_meas):
        win = vel[max(0, m - 1) : m + 2]
        win = win[~np.isnan(win)]
        if win.size == 0:
            marks.append(None)
            continue
        v = float(win.mean())
        marks.append(next(lbl for lim, lbl in DYN_LEVELS if v <= lim))
    out = []
    last = None
    m = 0
    while m < n_meas:
        mk = marks[m]
        if mk and mk != last:
            # 2小節以上続く変化だけ書く（ちらつき防止）
            run = 1
            while m + run < n_meas and marks[m + run] in (mk, None):
                run += 1
            if run >= 2 or last is None:
                out.append({"beat": float(bars[m]), "mark": mk})
                last = mk
        m += 1
    return out


def rearrange_from_score(score: dict) -> dict:
    """編集済みスコアの指番号・難所分析を作り直す（アレンジはしない）。"""
    s = deepcopy(score)
    assign_fingering(s)
    s["difficulty"] = analyze_difficulty(s)
    return s
