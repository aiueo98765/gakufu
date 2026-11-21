"""ギター編曲とタブ譜の検証。保存済みジョブのピアノ譜から3モードのギター譜を作り、
弦・フレットの妥当性（音高 = 開放弦 + カポ + フレット、同時に同じ弦を使わない、フレット幅）と
MusicXML の小節長を検算し、テキスト TAB の先頭を表示する。

python tests/test_guitar.py <job_id> [tuning] [capo]
"""
import collections
import json
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import bars as BARS  # noqa: E402
from app import pipeline  # noqa: E402
from app.guitar import ascii_tab, build_guitar_score  # noqa: E402
from app.notation import score_to_musicxml  # noqa: E402


def main():
    job = sys.argv[1]
    opts = {}
    if len(sys.argv) > 2:
        opts["tuning"] = sys.argv[2]
    if len(sys.argv) > 3:
        opts["capo"] = sys.argv[3]
    for mode in ("easy", "standard", "virtuoso"):
        piano = pipeline.ensure_score(job, mode)
        t = time.time()
        g = build_guitar_score(piano, opts)
        dt = time.time() - t
        opens = [o + g["guitar"]["capo"] for o in g["guitar"]["open"]]
        bad_pitch = sum(1 for n in g["notes"] if opens[6 - n["string"]] + n["fret"] != n["pitch"])
        groups = collections.defaultdict(list)
        for n in g["notes"]:
            groups[n["start"]].append(n)
        dup_string = sum(1 for v in groups.values() if len({n["string"] for n in v}) != len(v))
        spans = [max(f) - min(f) for v in groups.values() if len(f := [n["fret"] for n in v if n["fret"] > 0]) > 1]
        xml = score_to_musicxml(g, {})
        root = ET.fromstring(xml)
        div = int(root.find(".//divisions").text)
        bad_len = 0
        bl = BARS.of_score(g)
        for mi, me in enumerate(root.findall(".//measure")):
            pos = mx = 0
            for el in me:
                if el.tag == "note" and el.find("chord") is None:
                    pos += int(el.find("duration").text)
                elif el.tag == "backup":
                    pos -= int(el.find("duration").text)
                mx = max(mx, pos)
            bad_len += mx != round((bl[mi + 1] - bl[mi]) * div)
        frets = len(root.findall(".//fret"))
        maxpoly = max(len(v) for v in groups.values())
        print(
            f"{mode:9s} piano={len(piano['notes'])} guitar={len(g['notes'])} capo={g['guitar']['capo']} "
            f"poly<={maxpoly} max_span={max(spans) if spans else 0} bad_pitch={bad_pitch} dup_string={dup_string} "
            f"xml_frets={frets} bad_len={bad_len} diff={g['difficulty']['overall']}({g['difficulty']['label']}) {dt:.2f}s"
        )
        if mode == "standard":
            print("\n".join(ascii_tab(g).splitlines()[:12]))


if __name__ == "__main__":
    main()
