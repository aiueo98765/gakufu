"""ギター編曲とタブ譜（弦・フレット）の割り当て。

ピアノ用に作った同じ難易度の楽譜（拍・雑音除去・コード認識は済み）から、
ギター1本で弾ける形に落とし、各音を「どの弦の何フレットで弾くか」を決める。

- easy（簡単）     : 単音のメロディ（ローポジション中心）＋コード名
- standard（標準） : メロディ＋ベース（＋強拍に内声1音）のソロギター
- virtuoso（超プロ）: メロディ・ベース・内声を、同時に押さえられる限りすべて

弦・フレットの割り当ては、同時に鳴らす音のかたまり（イベント）ごとに
「押さえ方の候補」を全部並べ、指の開き（フレット幅）・必要な指の数・ポジションの高さ・
次のイベントへのポジション移動・伸ばしている音の弦を切ってしまうこと をコストにして
曲全体でビタビ（動的計画法）で最小化する。
"""
from __future__ import annotations

import itertools
import math

import numpy as np

from . import theory as T

# 低い弦（6弦）→高い弦（1弦）の開放弦の音高
TUNINGS: dict[str, tuple[str, list[int]]] = {
    "standard": ("標準 (EADGBE)", [40, 45, 50, 55, 59, 64]),
    "halfdown": ("半音下げ (E♭A♭D♭G♭B♭E♭)", [39, 44, 49, 54, 58, 63]),
    "dropd": ("ドロップD (DADGBE)", [38, 45, 50, 55, 59, 64]),
    "dadgad": ("DADGAD", [38, 45, 50, 55, 57, 62]),
    "openg": ("オープンG (DGDGBD)", [38, 43, 50, 55, 59, 62]),
}
MAX_FRET = 19
MODE_MAX_NOTES = {"easy": 1, "standard": 3, "virtuoso": 6}
MODE_MAX_FRET = {"easy": 7, "standard": 15, "virtuoso": MAX_FRET}
FRIENDLY_MAJOR = {0: 0.0, 7: 0.0, 2: 0.0, 9: 0.2, 4: 0.3, 5: 0.8}  # C G D A E F
FRIENDLY_MINOR = {9: 0.0, 4: 0.0, 2: 0.2, 11: 0.8}  # Am Em Dm Bm


def suggest_capo(key: dict, tuning: list[int]) -> int:
    """開放弦を活かせる調の形になるカポ位置（0〜7）を選ぶ。"""
    tonic = (T.key_tonic_pos(int(key["fifths"]), bool(key["minor"])) * 7) % 12
    table = FRIENDLY_MINOR if key["minor"] else FRIENDLY_MAJOR
    tune_shift = (tuning[0] - 40) % 12 if tuning[0] != 38 else 0
    best = (99.0, 0)
    for capo in range(0, 8):
        shape = (tonic - capo - tune_shift) % 12
        cost = table.get(shape, 2.5) + 0.12 * capo
        if cost < best[0]:
            best = (cost, capo)
    return best[1]


# ---------------------------------------------------------------------------
# 1. ギター向けの音の選び方
# ---------------------------------------------------------------------------
def _events(notes: list[dict]) -> list[dict]:
    evs: dict[float, list[dict]] = {}
    for n in notes:
        evs.setdefault(round(n["start"], 6), []).append(n)
    return [{"start": s, "notes": sorted(v, key=lambda n: n["pitch"])} for s, v in sorted(evs.items())]


def _fold(p: int, lo: int, hi: int) -> int:
    while p < lo:
        p += 12
    while p > hi:
        p -= 12
    return p


def reduce_for_guitar(piano: dict, mode: str, open_strings: list[int], capo: int) -> list[dict]:
    """ピアノ譜の音から、ギターで弾く音を選ぶ（音域に収め、同時発音数を絞る）。"""
    lo = open_strings[0] + capo
    hi = open_strings[-1] + capo + MODE_MAX_FRET[mode]
    rh = [n for n in piano["notes"] if n.get("hand") == "R"]
    lh = [n for n in piano["notes"] if n.get("hand") == "L"]
    out: list[dict] = []

    # メロディ = 右手の各打点の最高音。フレーズ（1拍以上の休み）ごとにオクターブを選び、
    # 12フレットまでの弾きやすい高さ（実音 G4 前後が中心）に置く。右手の内声も同じだけ動かす。
    mel_hi = open_strings[-1] + capo + 12
    rh_events = _events(rh)
    phrases: list[list[dict]] = []
    last_end = -1e9
    for ev in rh_events:
        if ev["start"] - last_end >= 1.0 or not phrases:
            phrases.append([])
        phrases[-1].append(ev)
        last_end = max(last_end, max(n["start"] + n["dur"] for n in ev["notes"]))
    rh_shift: dict[float, int] = {}
    for ph in phrases:
        tops = np.array([ev["notes"][-1]["pitch"] for ev in ph])
        best = None
        for k in (-3, -2, -1, 0, 1):
            sh = tops + 12 * k
            outside = int(((sh < lo + 7) | (sh > mel_hi)).sum())
            cost = outside * 10 + abs(float(np.median(sh)) - 67)
            if best is None or cost < best[0]:
                best = (cost, k)
        for ev in ph:
            rh_shift[ev["start"]] = 12 * best[1]
    for ev in rh_events:
        top = ev["notes"][-1]
        p = _fold(top["pitch"] + rh_shift[ev["start"]], lo + 7, hi)
        out.append({**top, "pitch": p, "hand": "R", "role": "melody"})
    if mode == "easy":
        # 簡単は単旋律だけ。重なりは次の音の頭で切る
        out.sort(key=lambda n: n["start"])
        for a, b in zip(out, out[1:]):
            if a["start"] + a["dur"] > b["start"]:
                a["dur"] = max(0.25, b["start"] - a["start"])
        return out

    # ベース = 左手の各打点の最低音（標準は強拍と和音の変わり目だけ）
    from . import bars as BARS

    bar_list = BARS.of_score(piano)
    beat_len = float(piano.get("beatLen") or 1)
    chord_starts = {round(c["start"], 6) for c in piano.get("chords", [])}
    for ev in _events(lh):
        bot = ev["notes"][0]
        if mode == "standard":
            strong = BARS.strong(ev["start"], bar_list) or round(ev["start"], 6) in chord_starts
            if not strong:
                continue
        p = _fold(bot["pitch"], lo, lo + 16)
        out.append({**bot, "pitch": p, "hand": "L", "role": "bass"})
        if mode == "virtuoso":
            # 内声：左手の他の音（ベースより上、メロディより下）
            for n in ev["notes"][1:]:
                q = _fold(n["pitch"], lo + 5, mel_hi - 5)
                out.append({**n, "pitch": q, "hand": "L", "role": "inner"})
    if mode == "virtuoso":
        for ev in rh_events:
            top_p = _fold(ev["notes"][-1]["pitch"] + rh_shift[ev["start"]], lo + 7, hi)
            for n in ev["notes"][:-1]:
                q = _fold(n["pitch"] + rh_shift[ev["start"]], lo + 5, max(lo + 5, top_p - 1))
                if q < top_p:
                    out.append({**n, "pitch": q, "hand": "R", "role": "inner"})
    elif mode == "standard":
        # 強拍の長いメロディの下にコードの音を1つ
        for m in [n for n in out if n["role"] == "melody"]:
            if m["dur"] < beat_len - 1e-6 or not BARS.strong(m["start"], bar_list):
                continue
            c = T.chord_at(piano.get("chords", []), m["start"])
            if not c:
                continue
            ivs = c.get("intervals") or next((d[1] for d in T.CHORD_DEFS if d[0] == c.get("suffix")), (0, 4, 7))
            pcs = {(c["root"] + iv) % 12 for iv in ivs}
            for q in range(m["pitch"] - 3, m["pitch"] - 10, -1):
                if q % 12 in pcs and q % 12 != m["pitch"] % 12 and q >= lo + 5:
                    out.append({**m, "pitch": q, "role": "inner", "hand": "R", "vel": max(40, m["vel"] - 15)})
                    break
    # 同じ時刻・同じ高さの重複を消す
    seen = set()
    uniq = []
    for n in sorted(out, key=lambda n: (n["start"], n["pitch"], n["role"] != "melody")):
        k = (round(n["start"], 6), n["pitch"])
        if k in seen:
            continue
        seen.add(k)
        uniq.append(n)
    return uniq


# ---------------------------------------------------------------------------
# 2. 弦・フレットの割り当て
# ---------------------------------------------------------------------------
def _candidates(pitches: list[int], opens: list[int], max_fret: int, locked: list[int | None]) -> list[tuple]:
    """低い音から順に、低い弦→高い弦へ（弦は重複なし・音高順）割り当てる候補を全部作る。"""
    k = len(pitches)
    out = []
    for strings in itertools.combinations(range(6), k):
        frets = []
        ok = True
        for p, s, lk in zip(pitches, strings, locked):
            f = p - opens[s]
            if f < 0 or f > max_fret or (lk is not None and lk != s):
                ok = False
                break
            frets.append(f)
        if ok:
            out.append((strings, tuple(frets)))
    return out


def _static_cost(frets: tuple[int, ...]) -> float:
    fretted = [f for f in frets if f > 0]
    c = 0.0
    if fretted:
        span = max(fretted) - min(fretted)
        c += max(0, span - 3) * 2.5 + max(0, span - 4) * 8
        low = min(fretted)
        # 一番低いフレットに複数あるならセーハ（人差し指1本）とみなす
        fingers = len(fretted) - max(0, fretted.count(low) - 1)
        c += max(0, fingers - 4) * 10
        # ハイポジションは弾きにくく音も細いので、必要なときだけ使う
        c += 0.15 * max(0, low - 5) + (0.2 * (low - 12) if low > 12 else 0)
    c -= 0.15 * sum(1 for f in frets if f == 0) if (not fretted or min(fretted) <= 5) else 0
    return c


def _position(frets: tuple[int, ...], prev: float) -> float:
    fretted = [f for f in frets if f > 0]
    return float(min(fretted)) if fretted else prev


def _victim(ns: list[dict]) -> dict:
    """押さえきれないとき最初に外す音：内声（同じ音名の重複を優先）→ ベース。メロディは最後まで残す。"""
    inner = [n for n in ns if n.get("role") not in ("melody", "bass")]
    if inner:
        pcs = [n["pitch"] % 12 for n in ns]
        dup = [n for n in inner if pcs.count(n["pitch"] % 12) > 1]
        pool = dup or inner
        mid = (ns[0]["pitch"] + ns[-1]["pitch"]) / 2
        return min(pool, key=lambda n: abs(n["pitch"] - mid))
    bass = [n for n in ns if n.get("role") == "bass"]
    if bass and len(ns) > 1:
        return bass[0]
    return ns[0] if ns[0] is not ns[-1] else ns[-1]


def assign_frets(notes: list[dict], opens: list[int], max_fret: int, sec_per_beat: float) -> list[dict]:
    """notes に "string"（MusicXML の番号: 1=1弦=いちばん高い）と "fret" を書き込む。弾けない音は落とす。"""
    evs = _events(notes)
    cands: list[list[tuple]] = []
    kept_events: list[dict] = []
    for ev in evs:
        ns = list(ev["notes"])
        while len(ns) > 6:
            ns.remove(_victim(ns))
        while True:
            pitches = [n["pitch"] for n in ns]
            locked = [(6 - n["string"]) if n.get("stringLocked") and n.get("string") else None for n in ns]
            cs = _candidates(pitches, opens, max_fret, locked)
            best = min((_static_cost(c[1]) for c in cs), default=99.0)
            # 4フレット幅を超える・指が足りない＝押さえられない。1音ずつ外して押さえられる形にする
            if (cs and best < 12) or len(ns) == 1:
                cs = [c for c in cs if _static_cost(c[1]) < 12] or cs
                break
            ns.remove(_victim(ns))
        if not cs:
            continue
        ev = {"start": ev["start"], "notes": ns, "end": max(n["start"] + n["dur"] for n in ns)}
        kept_events.append(ev)
        cands.append(cs)
    if not kept_events:
        return []

    # ビタビ
    static = [np.array([_static_cost(c[1]) for c in cs]) for cs in cands]
    cost = static[0].copy()
    pos_prev = np.array([_position(c[1], 1.0) for c in cands[0]])
    back: list[np.ndarray] = [np.zeros(len(cands[0]), dtype=np.int32)]
    for i in range(1, len(kept_events)):
        a, b = kept_events[i - 1], kept_events[i]
        dt = (b["start"] - a["start"]) * sec_per_beat
        w = 0.6 if dt < 0.25 else (0.3 if dt < 0.6 else 0.12)
        pos_b = np.array([_position(c[1], np.nan) for c in cands[i]])
        # 開放弦だけのイベントは直前のポジションを保つ
        shift = np.abs(np.where(np.isnan(pos_b)[None, :], pos_prev[:, None], pos_b[None, :]) - pos_prev[:, None])
        trans = w * shift
        # 伸ばしている音の弦を、違う音で弾き直してしまう
        held = [
            {s for s, n in zip(ca[0], a["notes"]) if n["start"] + n["dur"] > b["start"] + 1e-6}
            for ca in cands[i - 1]
        ]
        if any(held):
            used = [set(cb[0]) for cb in cands[i]]
            same_pitch = {n["pitch"] for n in b["notes"]}
            for ia, hs in enumerate(held):
                if not hs:
                    continue
                held_p = {s: n["pitch"] for s, n in zip(cands[i - 1][ia][0], a["notes"])}
                for ib, us in enumerate(used):
                    clash = [s for s in hs & us if held_p[s] not in same_pitch]
                    if clash:
                        trans[ia, ib] += 1.5 * len(clash)
        tot = cost[:, None] + trans
        arg = np.argmin(tot, axis=0)
        cost = tot[arg, np.arange(tot.shape[1])] + static[i]
        back.append(arg.astype(np.int32))
        pos_prev = np.where(np.isnan(pos_b), pos_prev[arg], pos_b)
    k = int(np.argmin(cost))
    choice = [0] * len(kept_events)
    for i in range(len(kept_events) - 1, -1, -1):
        choice[i] = k
        k = int(back[i][k])

    out = []
    for ev, cs, ci in zip(kept_events, cands, choice):
        strings, frets = cs[ci]
        for n, s, f in zip(ev["notes"], strings, frets):
            m = dict(n)
            m["string"] = 6 - s
            m["fret"] = int(f)
            out.append(m)
    out.sort(key=lambda n: (n["start"], n["pitch"]))
    return out


# ---------------------------------------------------------------------------
# 3. 難しさ（ギター用）
# ---------------------------------------------------------------------------
def guitar_difficulty(score: dict, threshold: float = 0.45) -> dict:
    from . import bars as BARS
    from .difficulty import LEVELS

    bar_list = BARS.of_score(score)
    spb = 60.0 / max(20.0, float(score.get("tempo", 100)))
    n_meas = BARS.count(bar_list)
    by_m: dict[int, list[dict]] = {}
    for n in score["notes"]:
        by_m.setdefault(BARS.index(n["start"], bar_list), []).append(n)
    measures = []
    for m in range(n_meas):
        ns = sorted(by_m.get(m, []), key=lambda n: (n["start"], n["pitch"]))
        if not ns:
            measures.append({"index": m, "score": 0.0, "reasons": [], "hands": {"R": None, "L": None}})
            continue
        evs = _events(ns)
        bpm = BARS.length(m, bar_list)
        nps = len(evs) / (bpm * spb)
        spans, poss = [], []
        for ev in evs:
            fr = [n.get("fret", 0) for n in ev["notes"] if n.get("fret", 0) > 0]
            spans.append(max(fr) - min(fr) if fr else 0)
            poss.append(min(fr) if fr else None)
        shifts = 0.0
        last = None
        for p in poss:
            if p is None:
                continue
            if last is not None:
                shifts += abs(p - last)
            last = p
        chord = max(len(ev["notes"]) for ev in evs)
        high = max((n.get("fret", 0) for n in ns), default=0)
        s_speed = min(1.0, max(0.0, (nps - 3) / 9))
        s_span = min(1.0, max(0.0, (max(spans) - 2) / 4))
        s_shift = min(1.0, shifts / (bpm * spb) / 20)
        s_chord = min(1.0, max(0.0, (chord - 2) / 4))
        s_high = min(1.0, max(0.0, (high - 9) / 10))
        sc = 0.35 * s_speed + 0.2 * s_span + 0.2 * s_shift + 0.15 * s_chord + 0.1 * s_high
        reasons = []
        if s_speed >= 0.45:
            reasons.append(f"速いパッセージ（毎秒{nps:.0f}音）")
        if s_span >= 0.5:
            reasons.append(f"指を大きく開く（{max(spans)}フレット幅）")
        if s_shift >= 0.5:
            reasons.append("ポジション移動が多い")
        if s_chord >= 0.5:
            reasons.append(f"{chord}音の和音")
        if s_high >= 0.5:
            reasons.append(f"ハイポジション（{high}フレット）")
        measures.append({"index": m, "score": round(min(1.0, sc), 3), "reasons": reasons, "hands": {"R": None, "L": None}})
    arr = np.array([x["score"] for x in measures if x["reasons"] or x["score"] > 0] or [0.0])
    overall = round(1 + 9 * min(1.0, 0.65 * float(np.percentile(arr, 90)) + 0.35 * float(arr.mean())) ** 0.85, 1)
    label = next(lbl for lim, lbl in LEVELS if overall <= lim)
    spans_ = []
    for x in measures:
        if x["score"] >= threshold:
            if spans_ and spans_[-1]["endMeasure"] == x["index"] - 1:
                spans_[-1]["endMeasure"] = x["index"]
                spans_[-1]["peak"] = max(spans_[-1]["peak"], x["score"])
                spans_[-1]["reasons"] += [r for r in x["reasons"] if r not in spans_[-1]["reasons"]]
            else:
                spans_.append({"startMeasure": x["index"], "endMeasure": x["index"], "peak": x["score"], "reasons": list(x["reasons"])})
    return {
        "overall": overall,
        "label": label,
        "threshold": threshold,
        "measures": measures,
        "spans": spans_,
        "difficultMeasures": [x["index"] for x in measures if x["score"] >= threshold],
    }


# ---------------------------------------------------------------------------
# 4. 入口
# ---------------------------------------------------------------------------
DEFAULT_GUITAR = {"tuning": "standard", "capo": "auto"}


def build_guitar_score(piano: dict, options: dict | None = None) -> dict:
    opts = {**DEFAULT_GUITAR, **(options or {})}
    tuning_key = opts["tuning"] if opts["tuning"] in TUNINGS else "standard"
    tuning_label, opens = TUNINGS[tuning_key]
    mode = piano["mode"]
    if opts["capo"] in ("auto", None, ""):
        capo = suggest_capo(piano["key"], opens) if mode != "virtuoso" else 0
    else:
        capo = int(max(0, min(9, int(opts["capo"]))))
    eff = [o + capo for o in opens]
    picked = reduce_for_guitar(piano, mode, opens, capo)
    sec_per_beat = 60.0 / max(20.0, float(piano.get("tempo", 100)))
    notes = assign_frets(picked, eff, MODE_MAX_FRET[mode] if mode != "easy" else 12, sec_per_beat)
    out_notes = []
    for i, n in enumerate(notes):
        sn = {k: n[k] for k in ("pitch", "start", "dur", "vel", "hand", "string", "fret") if k in n}
        sn["id"] = f"g{i + 1}"
        if n.get("role"):
            sn["role"] = n["role"]
        out_notes.append(sn)
    score = {
        **{k: v for k, v in piano.items() if k not in ("notes", "difficulty", "edited", "pedals", "dynamics")},
        "instrument": "guitar",
        "modeLabel": piano.get("modeLabel", ""),
        "notes": out_notes,
        "pedals": [],
        "dynamics": piano.get("dynamics", []),
        "guitar": {"tuning": tuning_key, "tuningLabel": tuning_label, "open": opens, "capo": capo, "options": opts},
    }
    score["difficulty"] = guitar_difficulty(score)
    return score


def refret(score: dict) -> dict:
    """編集後のギター譜の弦・フレットを割り当て直す（手で固定した弦は保つ）。"""
    g = score.get("guitar") or {}
    opens = g.get("open") or TUNINGS["standard"][1]
    capo = int(g.get("capo") or 0)
    eff = [o + capo for o in opens]
    sec_per_beat = 60.0 / max(20.0, float(score.get("tempo", 100)))
    by_id = {n["id"]: n for n in score["notes"]}
    assigned = assign_frets([dict(n) for n in score["notes"]], eff, MAX_FRET, sec_per_beat)
    seen = set()
    for n in assigned:
        if n["id"] in by_id:
            by_id[n["id"]]["string"] = n["string"]
            by_id[n["id"]]["fret"] = n["fret"]
            seen.add(n["id"])
    for nid, n in by_id.items():
        if nid not in seen:
            n.pop("string", None)
            n.pop("fret", None)
    score["difficulty"] = guitar_difficulty(score)
    return score


def ascii_tab(score: dict, width: int = 96) -> str:
    """テキストのタブ譜（e|---0---| 形式）。16分音符を1文字として小節ごとに並べる。"""
    g = score.get("guitar") or {}
    opens = g.get("open") or TUNINGS["standard"][1]
    names = []
    for p in opens[::-1]:
        step, alter, _ = T.spell_midi(p, 0, False)
        names.append((step if alter == 0 else step + ("#" if alter > 0 else "b")).ljust(2))
    names[0] = names[0].lower() if names[0].strip() == "E" else names[0]
    from . import bars as BARS

    bar_list = BARS.of_score(score)
    # 小節ごとの16分音符の列の範囲（2/4 の小節は8列）
    cols = [int(round(b * 4)) for b in bar_list]
    total = cols[-1]
    cells = [["-"] * total for _ in range(6)]
    for n in score["notes"]:
        if "string" not in n:
            continue
        i = int(round(n["start"] * 4))
        if 0 <= i < total:
            cells[n["string"] - 1][i] = str(n["fret"])
    lines = [
        f"{score.get('title', '')}",
        f"調弦: {g.get('tuningLabel', '標準')}   カポ: {g.get('capo', 0)}   ♩={score.get('tempo')}   {score['timeSignature'][0]}/{score['timeSignature'][1]}",
        "",
    ]
    # 16分音符1つ = 1列。列の幅はその列の一番長いフレット番号（2桁なら2文字）＋区切りの「-」
    widths = [max(1, max(len(cells[s][i]) for s in range(6))) for i in range(total)]
    n_meas = len(cols) - 1
    meas_w = [sum(widths[cols[m] : cols[m + 1]]) + (cols[m + 1] - cols[m]) + 1 for m in range(n_meas)]
    m0 = 0
    while m0 < n_meas:
        ms = [m0]
        used = 4 + meas_w[m0]
        while ms[-1] + 1 < n_meas and used + meas_w[ms[-1] + 1] <= width:
            ms.append(ms[-1] + 1)
            used += meas_w[ms[-1]]
        lines.append(f"  {m0 + 1}")
        for s in range(6):
            row = names[s] + "|"
            for m in ms:
                for i in range(cols[m], cols[m + 1]):
                    c = cells[s][i]
                    row += c.ljust(widths[i], "-") + "-"
                row += "|"
            lines.append(row)
        lines.append("")
        m0 = ms[-1] + 1
    return "\n".join(lines)
