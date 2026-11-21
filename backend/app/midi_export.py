"""MIDI 書き出し（mido）。

- score_to_midi: 楽譜どおり（量子化済み）。右手/左手を別トラック・別チャンネル、
  テンポ・拍子・調号・ダンパーペダル(CC64)つき。
- raw_to_midi:   AI が聴き取ったままのタイミング（量子化前）。演奏のニュアンスが残る。
"""
from __future__ import annotations

import io

import mido

from . import theory as T

TPB = 480


def _text(s: str) -> str:
    """mido はメタ文字列を Latin-1 で書く。日本語の題名は UTF-8 のバイト列をそのまま入れる
    （多くの DAW・MuseScore は UTF-8 として読む）。"""
    return (s or "").encode("utf-8")[:120].decode("latin-1")


def _key_name_for_mido(fifths: int, minor: bool) -> str:
    step, alter = T.pos_to_spelling(T.key_tonic_pos(fifths, minor))
    name = step + {-1: "b", 1: "#", 0: ""}.get(alter, "")
    valid = {
        "C", "G", "D", "A", "E", "B", "F#", "C#", "F", "Bb", "Eb", "Ab", "Db", "Gb", "Cb",
    }
    if minor:
        name += "m"
        valid = {v + "m" for v in ["A", "E", "B", "F#", "C#", "G#", "D#", "A#", "D", "G", "C", "F", "Bb", "Eb", "Ab"]}
    return name if name in valid else ("Am" if minor else "C")


def _track_from_events(events: list[tuple[int, int, mido.Message]], name: str) -> mido.MidiTrack:
    tr = mido.MidiTrack()
    tr.append(mido.MetaMessage("track_name", name=_text(name), time=0))
    events.sort(key=lambda x: (x[0], x[1]))
    last = 0
    for tick, _order, msg in events:
        msg.time = max(0, tick - last)
        tr.append(msg)
        last = tick
    tr.append(mido.MetaMessage("end_of_track", time=0))
    return tr


def score_to_midi(score: dict, tempo_scale: float = 1.0) -> bytes:
    mid = mido.MidiFile(type=1, ticks_per_beat=TPB)
    tempo_bpm = max(20.0, float(score.get("tempo", 100)) * tempo_scale)
    num = int(score["timeSignature"][0])
    den = int(score["timeSignature"][1]) if len(score["timeSignature"]) > 1 else 4
    meta = mido.MidiTrack()
    meta.append(mido.MetaMessage("track_name", name=_text(score.get("title", "Piano")[:60]), time=0))
    meta.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(tempo_bpm), time=0))
    try:
        meta.append(
            mido.MetaMessage("key_signature", key=_key_name_for_mido(score["key"]["fifths"], score["key"]["minor"]), time=0)
        )
    except Exception:
        pass
    # 小節マップ：長さの違う小節（2/4 など）で拍子を切り替える
    from . import bars as BARS

    bar_list = BARS.of_score(score)
    nominal = float(score.get("measureBeats") or num * 4 / den)
    ts_events = []
    prev = None
    for i in range(len(bar_list) - 1):
        L = bar_list[i + 1] - bar_list[i]
        sig = (num, den) if abs(L - nominal) < 1e-6 else ((int(round(L * 2)), 8) if den == 8 else (max(1, int(round(L))), 4))
        if sig != prev:
            ts_events.append((int(round(bar_list[i] * TPB)), sig))
            prev = sig
    last = 0
    for tick, (nu, de) in ts_events:
        meta.append(mido.MetaMessage("time_signature", numerator=nu, denominator=de, time=tick - last))
        last = tick
    meta.append(mido.MetaMessage("end_of_track", time=0))
    mid.tracks.append(meta)

    guitar = score.get("instrument") == "guitar"
    program = 25 if guitar else 0  # 25 = Acoustic Guitar (steel)
    names = (("R", 0, "Melody"), ("L", 1, "Bass")) if guitar else (("R", 0, "Right Hand"), ("L", 1, "Left Hand"))
    for hand, ch, name in names:
        ev: list[tuple[int, int, mido.Message]] = [(0, 0, mido.Message("program_change", program=program, channel=ch))]
        for n in score["notes"]:
            if n.get("hand", "R") != hand:
                continue
            on = int(round(n["start"] * TPB))
            off = int(round((n["start"] + n["dur"]) * TPB))
            if off <= on:
                off = on + 1
            vel = int(max(1, min(127, n.get("vel", 80))))
            ev.append((on, 2, mido.Message("note_on", note=int(n["pitch"]), velocity=vel, channel=ch)))
            ev.append((off, 1, mido.Message("note_off", note=int(n["pitch"]), velocity=0, channel=ch)))
        if hand == "L":
            for p in score.get("pedals", []):
                ev.append((int(round(p["start"] * TPB)), 3, mido.Message("control_change", control=64, value=127, channel=ch)))
                ev.append((int(round(p["end"] * TPB)), 0, mido.Message("control_change", control=64, value=0, channel=ch)))
        mid.tracks.append(_track_from_events(ev, name))

    buf = io.BytesIO()
    mid.save(file=buf)
    return buf.getvalue()


def raw_to_midi(raw: dict, bpm: float = 120.0) -> bytes:
    """秒単位の生データをそのまま書く（拍 = 60/bpm 秒として変換）。"""
    mid = mido.MidiFile(type=1, ticks_per_beat=TPB)
    meta = mido.MidiTrack()
    meta.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(bpm), time=0))
    meta.append(mido.MetaMessage("end_of_track", time=0))
    mid.tracks.append(meta)
    tick_per_sec = TPB * bpm / 60.0
    roles = {}
    for n in raw.get("notes", []):
        roles.setdefault(n.get("role", "piano"), []).append(n)
    for ch, (role, ns) in enumerate(sorted(roles.items())):
        ev: list[tuple[int, int, mido.Message]] = [(0, 0, mido.Message("program_change", program=0, channel=ch % 16))]
        for n in ns:
            on = int(round(n["on"] * tick_per_sec))
            off = max(on + 1, int(round(n["off"] * tick_per_sec)))
            vel = int(max(1, min(127, n.get("vel", 80))))
            ev.append((on, 2, mido.Message("note_on", note=int(n["pitch"]), velocity=vel, channel=ch % 16)))
            ev.append((off, 1, mido.Message("note_off", note=int(n["pitch"]), velocity=0, channel=ch % 16)))
        if role in ("piano", "harmony"):
            for p in raw.get("pedals", []):
                ev.append((int(round(p["on"] * tick_per_sec)), 3, mido.Message("control_change", control=64, value=127, channel=ch % 16)))
                ev.append((int(round(p["off"] * tick_per_sec)), 0, mido.Message("control_change", control=64, value=0, channel=ch % 16)))
        mid.tracks.append(_track_from_events(ev, role))
    buf = io.BytesIO()
    mid.save(file=buf)
    return buf.getvalue()
