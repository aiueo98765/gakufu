"""楽譜データ（拍単位の音符列）→ MusicXML 4.0 の浄書。

- 大譜表（右手=ト音、左手=ヘ音）。音域に応じて小節単位で音部記号を切り替え、
  高すぎる右手は 8va にする。
- 1つの譜表につき最大2声部。長さの違う同時音・持続音の下で動く内声を声部に分ける。
- 拍ごとに 2/4/8 分割 or 3/6 連符を判定し、拍をまたぐ音はタイで分割する。
- 連桁は拍単位、臨時記号は小節内の状態を追跡して必要な所にだけ付ける。
- コードネーム・ペダル・強弱・テンポ・指番号・トリル・アルペジオも書き出す。
"""
from __future__ import annotations

import datetime as _dt
import math
from xml.sax.saxutils import escape

from . import bars as BARS
from . import theory as T

DIV = 48  # 4分音符あたりの divisions

TYPE_TABLE = {
    288: ("whole", 1),
    192: ("whole", 0),
    144: ("half", 1),
    96: ("half", 0),
    72: ("quarter", 1),
    48: ("quarter", 0),
    36: ("eighth", 1),
    24: ("eighth", 0),
    18: ("16th", 1),
    12: ("16th", 0),
    9: ("32nd", 1),
    6: ("32nd", 0),
    3: ("64th", 0),
}
# 連符の音価（拍内の tick → 見た目の音符）
TUPLET_TYPES = {
    # 単純拍子（拍=48tick）
    3: {16: ("eighth", 0), 32: ("quarter", 0)},
    6: {8: ("16th", 0), 16: ("eighth", 0), 24: ("eighth", 1), 32: ("quarter", 0)},
    # 複合拍子（拍=72tick）：2連符・4連符
    2: {36: ("eighth", 0)},
    4: {18: ("16th", 0), 36: ("eighth", 0), 54: ("eighth", 1)},
}
TUPLET_RATIO = {3: (3, 2, "eighth"), 6: (6, 4, "16th"), 2: (2, 3, "eighth"), 4: (4, 6, "16th")}
SUBDIV_SIMPLE = [1, 2, 4, 8, 3, 6]
SUBDIV_COMPOUND = [1, 3, 6, 12, 2, 4]
BEAMABLE = {"eighth", "16th", "32nd", "64th"}
ACC_NAME = {-2: "flat-flat", -1: "flat", 0: "natural", 1: "sharp", 2: "double-sharp"}
SHARP_ORDER = "FCGDAEB"
FLAT_ORDER = "BEADGCF"

DEFAULT_OPTS = {
    "fingering": "difficult",  # all / difficult / none
    "chords": True,
    "pedal": True,
    "dynamics": True,
    "ottava": True,
    "tempoMark": True,
    "subtitle": None,
}


# --------------------------------------------------------------------------
# 位置の正規化
# --------------------------------------------------------------------------
def _snap_tick(t: int) -> int:
    if t % 6 == 0 or t % 8 == 0:
        return t
    return int(round(t / 6) * 6)


def _to_ticks(notes: list[dict]) -> list[dict]:
    out = []
    for n in notes:
        s = _snap_tick(int(round(n["start"] * DIV)))
        e = _snap_tick(int(round((n["start"] + n["dur"]) * DIV)))
        if e <= s:
            e = s + 6
        out.append({**n, "s": max(0, s), "e": e})
    return out


def _normalize_beats(notes: list[dict]) -> None:
    """同じ譜表・同じ拍に 16分系(6の倍数)と3連系(8の倍数)が混ざったら少数派を寄せる。"""
    bounds: dict[int, list[tuple[dict, str]]] = {}
    for n in notes:
        for k in ("s", "e"):
            r = n[k] % DIV
            if r:
                bounds.setdefault(n[k] // DIV, []).append((n, k))
    for _b, items in bounds.items():
        rs = [it[0][it[1]] % DIV for it in items]
        if all(r % 6 == 0 for r in rs) or all(r % 8 == 0 for r in rs):
            continue
        bin_votes = sum(1 for r in rs if r % 6 == 0)
        tup_votes = sum(1 for r in rs if r % 8 == 0)
        grid = 6 if bin_votes >= tup_votes else 8
        for n, k in items:
            base = n[k] - n[k] % DIV
            n[k] = base + int(round((n[k] % DIV) / grid) * grid)
    for n in notes:
        if n["e"] <= n["s"]:
            n["e"] = n["s"] + 6


# --------------------------------------------------------------------------
# 声部分け
# --------------------------------------------------------------------------
def _separate_voices(notes: list[dict]) -> list[list[dict]]:
    """[(声部0のイベント列), (声部1のイベント列)] を返す。イベント = 同時に始まり同時に終わる音の束。"""
    groups: dict[tuple[int, int], list[dict]] = {}
    for n in notes:
        groups.setdefault((n["s"], n["e"]), []).append(n)
    subs = sorted(groups.items(), key=lambda kv: (kv[0][0], -max(x["pitch"] for x in kv[1])))
    voices: list[list[dict]] = [[], []]

    def v_end(v):
        return voices[v][-1]["e"] if voices[v] else -1

    for (s, e), ns in subs:
        ev = {"s": s, "e": e, "notes": sorted(ns, key=lambda x: x["pitch"])}
        placed = False
        for v in (0, 1):
            last = voices[v][-1] if voices[v] else None
            if last is None or last["e"] <= s:
                if last is not None and last["s"] == s:
                    continue
                voices[v].append(ev)
                placed = True
                break
            # 少しだけ次の音に食い込むレガートは切って同じ声部に
            if last["s"] < s and (last["e"] - s) <= max(6, (last["e"] - last["s"]) // 4):
                last["e"] = s
                voices[v].append(ev)
                placed = True
                break
        if placed:
            continue
        # 3声目が要る：同じ開始の束があれば合流、無ければ一番早く終わる声部を切る
        merged = False
        for v in (0, 1):
            last = voices[v][-1] if voices[v] else None
            if last and last["s"] == s:
                last["notes"] = sorted(last["notes"] + ns, key=lambda x: x["pitch"])
                merged = True
                break
        if merged:
            continue
        v = min((0, 1), key=lambda k: v_end(k))
        last = voices[v][-1]
        if last["s"] < s:
            last["e"] = s
            voices[v].append(ev)
        else:
            last["notes"] = sorted(last["notes"] + ns, key=lambda x: x["pitch"])
    # 声部1に回したが、その区間の声部0が空いているイベントは声部0に戻す
    import bisect

    v0 = sorted(voices[0], key=lambda ev: ev["s"])
    starts = [ev["s"] for ev in v0]
    keep1 = []
    for ev in voices[1]:
        i = bisect.bisect_right(starts, ev["s"]) - 1
        before_ok = i < 0 or v0[i]["e"] <= ev["s"]
        after_ok = i + 1 >= len(v0) or v0[i + 1]["s"] >= ev["e"]
        if before_ok and after_ok:
            v0.insert(i + 1, ev)
            starts.insert(i + 1, ev["s"])
        else:
            keep1.append(ev)
    return [v0, keep1]


# --------------------------------------------------------------------------
# 音価の分割
# --------------------------------------------------------------------------
class Meter:
    """小節と拍の構造（tick 単位）。単純拍子は 拍=4分=48、複合拍子は 拍=付点4分=72。"""

    def __init__(self, num: int, den: int, measure_quarters: float):
        self.num, self.den = num, den
        self.compound = den == 8 and num % 3 == 0
        self.bt = 72 if self.compound else DIV
        self.ml = int(round(measure_quarters * DIV))
        self.nb = max(1, self.ml // self.bt)
        self.subdiv_order = SUBDIV_COMPOUND if self.compound else SUBDIV_SIMPLE
        self.tuplets = (2, 4) if self.compound else (3, 6)

    def allowed_chunks(self, beat: int) -> list[float]:
        """拍頭から始まる音が、拍をまたいで1つの音符で書ける長さ（拍数）。"""
        if self.compound:
            table = {2: {0: [2, 1], 1: [1]}, 4: {0: [4, 2, 1], 1: [1], 2: [2, 1], 3: [1]}}
        else:
            table = {
                4: {0: [4, 3, 2, 1.5, 1], 1: [1], 2: [2, 1.5, 1], 3: [1]},
                3: {0: [3, 2, 1.5, 1], 1: [2, 1.5, 1], 2: [1]},
                2: {0: [2, 1.5, 1], 1: [1]},
            }
        return table.get(self.nb, {}).get(beat, [1])


def _split_within_beat(cur: int, end: int, base: int, S: int, mt: Meter) -> list[tuple]:
    a, b = cur - base, end - base
    bt = mt.bt
    out = []
    if S in mt.tuplets and not (a == 0 and b == bt):
        unit = bt // S
        while a < b:
            rem = b - a
            if S == 3:  # 3連8分（1拍=48）
                d = 32 if (rem >= 32 and a in (0, 16)) else 16
            elif S == 6:  # 6連16分
                d = next(x for x in (32, 24, 16, 8) if x <= rem and (x != 24 or a % 24 == 0) and (x != 32 or a % 16 == 0)) if rem >= 8 else rem
            elif S == 2:  # 複合拍子の2連符（付点4分=72 に8分2つ）
                d = 36
            else:  # S == 4：4連16分
                d = next(x for x in (54, 36, 18) if x <= rem and (x != 54 or a == 0) and (x != 36 or a % 36 == 0)) if rem >= 18 else rem
            d = max(1, min(d, rem, unit * S))
            typ, dots = TUPLET_TYPES[S].get(d, _nearest_type(d))
            out.append((base + a, d, typ, dots, S))
            a += d
        return out
    while a < b:
        rem = b - a
        chosen = None
        if mt.compound:
            for d in (72, 48, 36, 24, 18, 12, 9, 6, 3):
                if d > rem:
                    continue
                ok = (
                    (d == 72 and a == 0)
                    or (d == 48 and a in (0, 24))
                    or (d == 36 and a == 0)
                    or (d == 24 and a % 24 == 0)
                    or (d == 18 and a % 6 == 0 and a % 24 + 18 <= 24)
                    or (d == 12 and a % 12 == 0)
                    or (d == 9 and a % 3 == 0 and a % 12 + 9 <= 12)
                    or (d in (6, 3) and a % d == 0)
                )
                if ok:
                    chosen = d
                    break
        else:
            for d in (48, 36, 24, 18, 12, 9, 6, 3):
                if d > rem:
                    continue
                if d in (48, 24, 12, 6, 3) and a % d == 0:
                    chosen = d
                    break
                if d in (36, 18, 9) and a % (d // 3) == 0 and a + d <= DIV:
                    chosen = d
                    break
        if chosen is None:
            chosen = rem
        typ, dots = TYPE_TABLE.get(chosen) or _nearest_type(chosen)
        out.append((base + a, chosen, typ, dots, None))
        a += chosen
    return out


def _nearest_type(ticks: int) -> tuple[str, int]:
    k = min(TYPE_TABLE, key=lambda x: abs(x - ticks))
    return TYPE_TABLE[k]


def _split_item(a: int, b: int, mt: Meter, sub: list[int]) -> list[tuple]:
    """小節内の区間 [a, b)（小節頭からの tick）を書ける音価の列に分ける。"""
    bt = mt.bt
    out = []
    cur = a
    while cur < b:
        beat, off = divmod(cur, bt)
        if off == 0:
            chunk = None
            for c in mt.allowed_chunks(beat):
                ticks = int(c * bt)
                if ticks > b - cur or beat + math.ceil(c) > mt.nb:
                    continue
                if c == 1.5 and (beat + 1 >= mt.nb or sub[beat + 1] not in (1, 2, 4, 8)):
                    continue
                if ticks not in TYPE_TABLE:
                    continue
                chunk = ticks
                break
            if chunk:
                typ, dots = TYPE_TABLE[chunk]
                out.append((cur, chunk, typ, dots, None))
                cur += chunk
                continue
        end = min(b, (beat + 1) * bt)
        S = sub[beat] if beat < len(sub) else (6 if mt.compound else 4)
        out += _split_within_beat(cur, end, beat * bt, S, mt)
        cur = end
    return out


def _beat_subdivisions(bounds: list[int], mt: Meter) -> list[int]:
    bt = mt.bt
    sub = []
    for beat in range(mt.nb):
        rs = [x - beat * bt for x in bounds if beat * bt < x < (beat + 1) * bt]
        chosen = 12 if mt.compound else 8
        for S in mt.subdiv_order:
            unit = bt // S
            if all(r % unit == 0 for r in rs):
                chosen = S
                break
        sub.append(chosen)
    return sub

# --------------------------------------------------------------------------
# 音部記号・8va
# --------------------------------------------------------------------------
def _clefs_and_ottava(staff_notes: list[list[dict]], n_meas: int, meas_of, use_ottava: bool):
    per_meas: list[list[list[dict]]] = [[[] for _ in range(n_meas)] for _ in range(2)]
    for st in (0, 1):
        for n in staff_notes[st]:
            m = meas_of(n["s"])
            if 0 <= m < n_meas:
                per_meas[st][m].append(n)
    clefs = [["G"] * n_meas, ["F"] * n_meas]
    for st, default in ((0, "G"), (1, "F")):
        cur = default
        for m in range(n_meas):
            ns = per_meas[st][m]
            if ns:
                w = sum((n["e"] - n["s"]) for n in ns) or 1
                mean = sum(n["pitch"] * (n["e"] - n["s"]) for n in ns) / w
                lo = min(n["pitch"] for n in ns)
                hi = max(n["pitch"] for n in ns)
                if cur == "G" and mean < 54 and hi < 62:
                    cur = "F"
                elif cur == "F" and mean > 65 and lo >= 57:
                    cur = "G"
                elif st == 0 and cur == "F" and mean >= 58:
                    cur = "G"
                elif st == 1 and cur == "G" and mean <= 57:
                    cur = "F"
            clefs[st][m] = cur
    ottava = [False] * n_meas
    if use_ottava:
        for m in range(n_meas):
            ns = per_meas[0][m]
            if ns and clefs[0][m] == "G":
                lo = min(n["pitch"] for n in ns)
                mean = sum(n["pitch"] for n in ns) / len(ns)
                ottava[m] = lo >= 79 and mean >= 86
    return clefs, ottava


# --------------------------------------------------------------------------
# 本体
# --------------------------------------------------------------------------
def score_to_musicxml(score: dict, options: dict | None = None) -> str:
    opts = {**DEFAULT_OPTS, **(options or {})}
    ts_num, ts_den = int(score["timeSignature"][0]), int(score["timeSignature"][1] if len(score["timeSignature"]) > 1 else 4)
    L = float(score.get("measureBeats") or (ts_num * 4 / ts_den))  # 1小節の長さ（4分音符単位）
    mt = Meter(ts_num, ts_den, L)
    fifths = int(score["key"]["fifths"])
    minor = bool(score["key"]["minor"])
    notes = _to_ticks(score.get("notes", []))
    # 小節マップ（小節ごとの開始 tick）。途中に 2/4 などの短い小節があってよい
    bar_q = BARS.of_score(score)
    MS = [int(round(b * DIV)) for b in bar_q]
    end_tick = max((n["e"] for n in notes), default=MS[-1])
    while MS[-1] < end_tick:
        MS.append(MS[-1] + mt.ml)
    n_meas = len(MS) - 1
    import bisect as _bis

    def meas_of(tick: int) -> int:
        return max(0, min(n_meas - 1, _bis.bisect_right(MS, tick) - 1))

    def meter_of(m: int) -> Meter:
        length = MS[m + 1] - MS[m]
        if length == mt.ml:
            return mt
        if mt.compound:
            eighths = max(1, round(length / 24))
            return Meter(eighths, 8, length / DIV)
        return Meter(max(1, round(length / DIV)), 4, length / DIV)

    mts = [meter_of(m) for m in range(n_meas)]

    guitar = score.get("instrument") == "guitar"
    if guitar:
        # ギター: 上=五線譜（ト音記号 8vb）、下=タブ譜。両方に同じ音を書く
        staff_notes = [[dict(n) for n in notes], [dict(n) for n in notes]]
    else:
        staff_notes = [[n for n in notes if n.get("hand", "R") == "R"], [n for n in notes if n.get("hand", "R") == "L"]]
    for sn in staff_notes:
        if mt.compound:
            for n in sn:  # 複合拍子は 32分(6tick) の格子に揃える
                n["s"] = int(round(n["s"] / 6) * 6)
                n["e"] = max(n["s"] + 6, int(round(n["e"] / 6) * 6))
        else:
            _normalize_beats(sn)
    voices = [_separate_voices(sn) for sn in staff_notes]
    if guitar:
        clefs, ottava = [["G8"] * n_meas, ["TAB"] * n_meas], [False] * n_meas
    else:
        clefs, ottava = _clefs_and_ottava(staff_notes, n_meas, meas_of, opts["ottava"])

    difficult = set((score.get("difficulty") or {}).get("difficultMeasures") or [])
    show_finger = opts["fingering"]

    chord_ticks: dict[int, dict] = {}
    if opts["chords"]:
        for c in score.get("chords", []):
            chord_ticks[int(round(c["start"] * DIV))] = c
    pedal_marks: list[tuple[int, str]] = []
    if opts["pedal"]:
        for p in score.get("pedals", []):
            pedal_marks.append((int(round(p["start"] * DIV)), "start"))
            pedal_marks.append((int(round(p["end"] * DIV)), "stop"))
        pedal_marks.sort(key=lambda x: (x[0], 0 if x[1] == "stop" else 1))
    dyn_by_meas: dict[int, str] = {}
    if opts["dynamics"]:
        for d in score.get("dynamics", []):
            dyn_by_meas[meas_of(int(round(d["beat"] * DIV)))] = d["mark"]

    # ---- 各声部をピースに分解 ----
    # pieces[staff][voice][measure] = list of piece dict
    pieces = [[[[] for _ in range(n_meas)] for _ in range(2)] for _ in range(2)]
    voice_present = [[[False] * n_meas for _ in range(2)] for _ in range(2)]
    for st in (0, 1):
        for vi in (0, 1):
            evs = voices[st][vi]
            if vi == 1 and not evs:
                continue
            for ev in evs:
                for m in range(meas_of(ev["s"]), meas_of(ev["e"] - 1) + 1):
                    voice_present[st][vi][m] = True
            if vi == 0:
                for m in range(n_meas):
                    voice_present[st][vi][m] = True
            # 休符で埋めたアイテム列
            items = []
            cur = 0
            for ev in evs:
                if ev["s"] > cur:
                    items.append({"s": cur, "e": ev["s"], "notes": None})
                items.append(ev)
                cur = max(cur, ev["e"])
            end_total = MS[-1]
            if cur < end_total:
                items.append({"s": cur, "e": end_total, "notes": None})
            extra_split = set(chord_ticks) if (st == 0 and vi == 0) else set()
            for it in items:
                cuts = sorted(
                    {it["s"], it["e"]}
                    | {MS[m] for m in range(meas_of(it["s"]) + 1, meas_of(it["e"] - 1) + 1)}
                    | {t for t in extra_split if it["s"] < t < it["e"]}
                )
                segs = list(zip(cuts, cuts[1:]))
                it["_segs"] = segs
            # 声部・小節ごとの拍分割を、その小節の境界から決める
            bounds_by_meas: dict[int, list[int]] = {}
            for it in items:
                for a, b in it["_segs"]:
                    m = meas_of(a)
                    bounds_by_meas.setdefault(m, []).extend([a - MS[m], b - MS[m]])
            subs = {m: _beat_subdivisions(bs, mts[m]) for m, bs in bounds_by_meas.items()}
            for it in items:
                ev_pieces = []
                for a, b in it["_segs"]:
                    m = meas_of(a)
                    if a >= MS[-1]:
                        continue
                    if it["notes"] is None and not voice_present[st][vi][m]:
                        continue
                    if it["notes"] is None and a == MS[m] and b == MS[m + 1]:
                        p = {"m": m, "s": a, "dur": MS[m + 1] - MS[m], "type": None, "dots": 0, "tup": None, "rest": True, "measure_rest": True}
                        ev_pieces.append(p)
                        continue
                    default_sub = [6 if mts[m].compound else 4] * mts[m].nb
                    for (ps, pd, typ, dots, tup) in _split_item(a - MS[m], b - MS[m], mts[m], subs.get(m, default_sub)):
                        ev_pieces.append(
                            {
                                "m": m,
                                "s": MS[m] + ps,
                                "dur": pd,
                                "type": typ,
                                "dots": dots,
                                "tup": tup,
                                "rest": it["notes"] is None,
                                "notes": it["notes"],
                            }
                        )
                if it["notes"] is not None and len(ev_pieces) > 1:
                    for i, p in enumerate(ev_pieces):
                        p["tie_start"] = i < len(ev_pieces) - 1
                        p["tie_stop"] = i > 0
                if ev_pieces and it["notes"] is not None:
                    ev_pieces[0]["first"] = True
                for p in ev_pieces:
                    p["hidden"] = p["rest"] and vi == 1
                    pieces[st][vi][p["m"]].append(p)

    # ---- 連符の括弧と連桁 ----
    for st in (0, 1):
        for vi in (0, 1):
            for m in range(n_meas):
                ps = pieces[st][vi][m]
                ps.sort(key=lambda p: p["s"])
                by_beat: dict[int, list[dict]] = {}
                for p in ps:
                    by_beat.setdefault((p["s"] - MS[m]) // mts[m].bt, []).append(p)
                for _b, group in by_beat.items():
                    tups = [p for p in group if p.get("tup")]
                    if tups:
                        tups[0]["tup_start"] = True
                        tups[-1]["tup_stop"] = True
                    run: list[dict] = []
                    for p in group + [None]:
                        ok = p is not None and not p["rest"] and p.get("type") in BEAMABLE and p["dur"] < mts[m].bt
                        if ok:
                            run.append(p)
                            continue
                        if len(run) >= 2:
                            run[0]["beam"] = "begin"
                            for q in run[1:-1]:
                                q["beam"] = "continue"
                            run[-1]["beam"] = "end"
                        run = []

    # ---- 臨時記号（譜表ごと・小節ごと・全声部の時間順）----
    key_alter = {s: 0 for s in "CDEFGAB"}
    if fifths > 0:
        for s in SHARP_ORDER[:fifths]:
            key_alter[s] = 1
    elif fifths < 0:
        for s in FLAT_ORDER[: -fifths]:
            key_alter[s] = -1
    for st in (0, 1):
        for m in range(n_meas):
            state: dict[tuple[str, int], int] = {}
            allp = sorted(
                [p for vi in (0, 1) for p in pieces[st][vi][m] if not p["rest"]],
                key=lambda p: p["s"],
            )
            for p in allp:
                p["spell"] = []
                for n in p["notes"]:
                    step, alter, octave = T.spell_midi(n["pitch"], fifths, minor)
                    acc = None
                    k = (step, octave)
                    cur = state.get(k, key_alter[step])
                    if not p.get("tie_stop"):
                        if alter != cur:
                            acc = ACC_NAME[alter]
                        state[k] = alter
                    p["spell"].append((step, alter, octave, acc))

    # ---- ペダル記号は左手声部0のピースの前に置く ----
    pedal_at: dict[int, list[str]] = {}
    lh_starts = sorted({p["s"] for m in range(n_meas) for p in pieces[1][0][m]})
    import bisect

    for t, kind in pedal_marks:
        i = bisect.bisect_left(lh_starts, t)
        if i < len(lh_starts):
            pedal_at.setdefault(lh_starts[i], []).append(kind)

    # ---- 書き出し ----
    title = escape(score.get("title") or "無題")
    subtitle = opts.get("subtitle")
    if subtitle is None:
        subtitle = f"{score.get('modeLabel', '')}"
    out: list[str] = [
        '<?xml version="1.0" encoding="UTF-8" standalone="no"?>',
        '<!DOCTYPE score-partwise PUBLIC "-//Recordare//DTD MusicXML 4.0 Partwise//EN" "http://www.musicxml.org/dtds/partwise.dtd">',
        '<score-partwise version="4.0">',
        f"<work><work-title>{title}</work-title></work>",
    ]
    if subtitle:
        out.append(f"<movement-title>{escape(subtitle)}</movement-title>")
    out.append("<identification>")
    if score.get("artist"):
        out.append(f'<creator type="composer">{escape(score["artist"])}</creator>')
    out.append(f'<creator type="arranger">AI採譜・{escape(score.get("modeLabel", ""))}</creator>')
    out.append(
        f"<encoding><software>gakufu</software><encoding-date>{_dt.date.today().isoformat()}</encoding-date></encoding>"
    )
    out.append("</identification>")
    inst_name, inst_abbr, program = ("Guitar", "Gtr.", 26) if guitar else ("Piano", "Pno.", 1)
    out.append(
        f'<part-list><score-part id="P1"><part-name>{inst_name}</part-name><part-abbreviation>{inst_abbr}</part-abbreviation>'
        f'<score-instrument id="P1-I1"><instrument-name>{inst_name}</instrument-name></score-instrument>'
        f'<midi-instrument id="P1-I1"><midi-channel>1</midi-channel><midi-program>{program}</midi-program></midi-instrument>'
        "</score-part></part-list>"
    )
    g = score.get("guitar") or {}
    out.append('<part id="P1">')
    ott_on = False
    for m in range(n_meas):
        out.append(f'<measure number="{m + 1}">')
        attrs = []
        if m == 0:
            attrs.append(f"<divisions>{DIV}</divisions>")
            attrs.append(f"<key><fifths>{fifths}</fifths><mode>{'minor' if minor else 'major'}</mode></key>")
            attrs.append(f"<time><beats>{mts[0].num}</beats><beat-type>{mts[0].den}</beat-type></time>")
            attrs.append("<staves>2</staves>")
        elif (mts[m].num, mts[m].den) != (mts[m - 1].num, mts[m - 1].den):
            # 小節の長さが変わる所（2/4 の小節など）で拍子記号を書く
            attrs.append(f"<time><beats>{mts[m].num}</beats><beat-type>{mts[m].den}</beat-type></time>")
        for st in (0, 1):
            if m == 0 or clefs[st][m] != clefs[st][m - 1]:
                cl = clefs[st][m]
                if cl == "G8":
                    attrs.append(f'<clef number="{st + 1}"><sign>G</sign><line>2</line><clef-octave-change>-1</clef-octave-change></clef>')
                elif cl == "TAB":
                    attrs.append(f'<clef number="{st + 1}"><sign>TAB</sign><line>5</line></clef>')
                else:
                    sign, line = ("G", 2) if cl == "G" else ("F", 4)
                    attrs.append(f'<clef number="{st + 1}"><sign>{sign}</sign><line>{line}</line></clef>')
        if m == 0 and guitar:
            tun = "".join(
                f'<staff-tuning line="{i + 1}"><tuning-step>{s}</tuning-step>'
                + (f"<tuning-alter>{a}</tuning-alter>" if a else "")
                + f"<tuning-octave>{o}</tuning-octave></staff-tuning>"
                for i, (s, a, o) in enumerate(T.spell_midi(p, 0, False) for p in (g.get("open") or [40, 45, 50, 55, 59, 64]))
            )
            capo_xml = f"<capo>{int(g.get('capo') or 0)}</capo>" if g.get("capo") else ""
            attrs.append(f'<staff-details number="2"><staff-lines>6</staff-lines>{tun}{capo_xml}</staff-details>')
        if attrs:
            out.append("<attributes>" + "".join(attrs) + "</attributes>")
        if m == 0 and opts["tempoMark"]:
            bpm = int(round(score.get("tempo", 100)))
            unit = "<beat-unit>quarter</beat-unit><beat-unit-dot/>" if mt.compound else "<beat-unit>quarter</beat-unit>"
            shown = int(round(bpm / 1.5)) if mt.compound else bpm
            out.append(
                '<direction placement="above"><direction-type><metronome parentheses="no">'
                f"{unit}<per-minute>{shown}</per-minute></metronome></direction-type>"
                f'<staff>1</staff><sound tempo="{bpm}"/></direction>'
            )
        if m == 0 and guitar and (g.get("capo") or g.get("tuning", "standard") != "standard"):
            parts_txt = []
            if g.get("tuning", "standard") != "standard":
                parts_txt.append("Tuning: " + "".join(T.note_name(p)[:-1] for p in (g.get("open") or [])))
            if g.get("capo"):
                parts_txt.append(f"Capo {int(g['capo'])}")
            out.append(
                '<direction placement="above"><direction-type><words font-weight="bold">'
                + escape("  ".join(parts_txt))
                + "</words></direction-type><staff>1</staff></direction>"
            )
        first_voice_written = False
        for st in (0, 1):
            for vi in (0, 1):
                if not voice_present[st][vi][m] or not pieces[st][vi][m]:
                    continue
                if first_voice_written:
                    out.append(f"<backup><duration>{MS[m + 1] - MS[m]}</duration></backup>")
                first_voice_written = True
                two_voices = voice_present[st][1][m] and bool(pieces[st][1][m])
                voice_no = (1 if vi == 0 else 2) if st == 0 else (5 if vi == 0 else 6)
                # 2声部の小節では、その小節で高い方の声部を符尾上向きにする
                upper = 0
                if two_voices:
                    def _mean(v):
                        ps = [n["pitch"] for p in pieces[st][v][m] if not p["rest"] for n in p["notes"]]
                        return sum(ps) / len(ps) if ps else None

                    m0, m1 = _mean(0), _mean(1)
                    if m0 is not None and m1 is not None and m1 > m0:
                        upper = 1
                if st == 0 and vi == 0:
                    if m in dyn_by_meas:
                        mk = dyn_by_meas[m]
                        out.append(
                            f'<direction placement="below"><direction-type><dynamics><{mk}/></dynamics></direction-type>'
                            f"<staff>1</staff></direction>"
                        )
                    if ottava[m] and not ott_on:
                        out.append(
                            '<direction placement="above"><direction-type><octave-shift type="down" size="8" number="1"/>'
                            "</direction-type><staff>1</staff></direction>"
                        )
                        ott_on = True
                for p in pieces[st][vi][m]:
                    if st == 0 and vi == 0 and p["s"] in chord_ticks:
                        out.append(_harmony_xml(chord_ticks[p["s"]], fifths, minor))
                    if st == 1 and vi == 0 and p["s"] in pedal_at:
                        for kind in pedal_at.pop(p["s"]):
                            out.append(
                                f'<direction placement="below"><direction-type><pedal type="{kind}" line="no" sign="yes"/>'
                                "</direction-type><staff>2</staff></direction>"
                            )
                    out.extend(
                        _piece_xml(
                            p,
                            st,
                            voice_no,
                            two_voices,
                            vi == upper,
                            show_finger == "all" or (show_finger == "difficult" and m in difficult),
                            tab=guitar and st == 1,
                        )
                    )
                if st == 0 and vi == 0 and ott_on and (m + 1 >= n_meas or not ottava[m + 1]):
                    out.append(
                        '<direction placement="above"><direction-type><octave-shift type="stop" size="8" number="1"/>'
                        "</direction-type><staff>1</staff></direction>"
                    )
                    ott_on = False
        if m == n_meas - 1:
            out.append('<barline location="right"><bar-style>light-heavy</bar-style></barline>')
        out.append("</measure>")
    out.append("</part></score-partwise>")
    return "\n".join(out)


def _harmony_xml(c: dict, fifths: int, minor: bool) -> str:
    step, alter = T.spell_pc(c["root"], fifths, minor)
    parts = ['<harmony print-frame="no" placement="above"><root>', f"<root-step>{step}</root-step>"]
    if alter:
        parts.append(f"<root-alter>{alter}</root-alter>")
    parts.append("</root>")
    parts.append(f'<kind text="{escape(c["suffix"])}">{c["kind"]}</kind>')
    if c.get("bass") is not None:
        bstep, balter = T.spell_from_root(step, alter, c["bass"] - c["root"])
        parts.append(f"<bass><bass-step>{bstep}</bass-step>")
        if balter:
            parts.append(f"<bass-alter>{balter}</bass-alter>")
        parts.append("</bass>")
    for deg, alt in c.get("degrees") or []:
        parts.append(
            f"<degree><degree-value>{deg}</degree-value><degree-alter>{alt}</degree-alter>"
            "<degree-type>add</degree-type></degree>"
        )
    parts.append("<staff>1</staff></harmony>")
    return "".join(parts)


def _piece_xml(
    p: dict, st: int, voice_no: int, two_voices: bool, stem_up: bool, show_finger: bool, tab: bool = False
) -> list[str]:
    lines: list[str] = []
    staff = st + 1
    if p["rest"]:
        attr = ' print-object="no"' if p.get("hidden") else ""
        x = [f"<note{attr}>"]
        if p.get("measure_rest"):
            x.append('<rest measure="yes"/>')
        else:
            x.append("<rest/>")
        x.append(f"<duration>{p['dur']}</duration><voice>{voice_no}</voice>")
        if not p.get("measure_rest"):
            x.append(f"<type>{p['type']}</type>" + "<dot/>" * p["dots"])
            if p.get("tup"):
                a, nn, nt = TUPLET_RATIO[p["tup"]]
                x.append(
                    f"<time-modification><actual-notes>{a}</actual-notes><normal-notes>{nn}</normal-notes>"
                    f"<normal-type>{nt}</normal-type></time-modification>"
                )
        x.append(f"<staff>{staff}</staff>")
        nots = _tuplet_notation(p)
        if nots:
            x.append("<notations>" + nots + "</notations>")
        x.append("</note>")
        lines.append("".join(x))
        return lines

    for idx, (n, (step, alter, octave, acc)) in enumerate(zip(p["notes"], p["spell"])):
        x = ["<note>"]
        if idx > 0:
            x.append("<chord/>")
        x.append(f"<pitch><step>{step}</step>")
        if alter:
            x.append(f"<alter>{alter}</alter>")
        x.append(f"<octave>{octave}</octave></pitch>")
        x.append(f"<duration>{p['dur']}</duration>")
        if p.get("tie_stop"):
            x.append('<tie type="stop"/>')
        if p.get("tie_start"):
            x.append('<tie type="start"/>')
        x.append(f"<voice>{voice_no}</voice>")
        x.append(f"<type>{p['type']}</type>" + "<dot/>" * p["dots"])
        if acc and not tab:
            x.append(f"<accidental>{acc}</accidental>")
        if p.get("tup"):
            a, nn, nt = TUPLET_RATIO[p["tup"]]
            x.append(
                f"<time-modification><actual-notes>{a}</actual-notes><normal-notes>{nn}</normal-notes>"
                f"<normal-type>{nt}</normal-type></time-modification>"
            )
        if two_voices:
            x.append(f"<stem>{'up' if stem_up else 'down'}</stem>")
        x.append(f"<staff>{staff}</staff>")
        if idx == 0 and p.get("beam"):
            x.append(f'<beam number="1">{p["beam"]}</beam>')
        nots = ""
        if p.get("tie_stop"):
            nots += '<tied type="stop"/>'
        if p.get("tie_start"):
            nots += '<tied type="start"/>'
        if idx == 0:
            nots += _tuplet_notation(p)
        if tab and n.get("string"):
            nots += f'<technical><string>{int(n["string"])}</string><fret>{int(n["fret"])}</fret></technical>'
        if p.get("first") and not tab:
            if n.get("ornament") == "trill" and idx == len(p["notes"]) - 1:
                nots += "<ornaments><trill-mark/></ornaments>"
            if show_finger and n.get("finger"):
                place = "above" if st == 0 else "below"
                nots += f'<technical><fingering placement="{place}">{int(n["finger"])}</fingering></technical>'
            if n.get("arpeggiate") and len(p["notes"]) > 1:
                nots += "<arpeggiate/>"
        if nots:
            x.append("<notations>" + nots + "</notations>")
        x.append("</note>")
        lines.append("".join(x))
    return lines


def _tuplet_notation(p: dict) -> str:
    s = ""
    if p.get("tup_start"):
        s += '<tuplet type="start" bracket="no"/>'
    if p.get("tup_stop"):
        s += '<tuplet type="stop"/>'
    return s
