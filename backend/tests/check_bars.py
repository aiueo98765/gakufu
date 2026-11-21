"""小節マップの検算：MusicXML の各小節の長さが小節マップと一致するか、拍子記号の変化、
MIDI の拍子イベントを確かめる。

python tests/check_bars.py <job_id> [mode]
"""
import io
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8")

import mido  # noqa: E402

from app import bars as BARS  # noqa: E402
from app import pipeline  # noqa: E402
from app.guitar import ascii_tab, build_guitar_score  # noqa: E402
from app.midi_export import score_to_midi  # noqa: E402
from app.notation import score_to_musicxml  # noqa: E402


def check(score: dict, label: str) -> None:
    b = BARS.of_score(score)
    xml = score_to_musicxml(score, {"fingering": "all"})
    root = ET.fromstring(xml)
    div = int(root.find(".//divisions").text)
    bad = []
    times = []
    for i, me in enumerate(root.findall(".//measure")):
        pos = mx = 0
        for el in me:
            if el.tag == "note" and el.find("chord") is None:
                pos += int(el.find("duration").text)
            elif el.tag == "backup":
                pos -= int(el.find("duration").text)
            mx = max(mx, pos)
        want = round((b[i + 1] - b[i]) * div)
        if mx != want:
            bad.append((i + 1, mx, want))
        t = me.find("attributes/time")
        if t is not None:
            times.append((i + 1, f"{t.findtext('beats')}/{t.findtext('beat-type')}"))
    mid = mido.MidiFile(file=io.BytesIO(score_to_midi(score)))
    ts_events = sum(1 for msg in mid.tracks[0] if msg.type == "time_signature")
    print(f"{label:22s} measures={len(root.findall('.//measure'))} bars={len(b) - 1} bad_len={bad[:5]} time_sigs={times[:8]} midi_ts={ts_events}")


def main():
    job = sys.argv[1]
    mode = sys.argv[2] if len(sys.argv) > 2 else "virtuoso"
    s = pipeline.ensure_score(job, mode, force=True) if "--force" in sys.argv else pipeline.ensure_score(job, mode)
    check(s, f"piano {mode}")
    g = build_guitar_score(s, {})
    check(g, f"guitar {mode}")
    tab = ascii_tab(g)
    print("ascii tab lines:", len(tab.splitlines()))


if __name__ == "__main__":
    main()
