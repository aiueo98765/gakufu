"""保存済みジョブの生データから3モードを作り直し、MusicXML/MIDI まで通す。

python tests/rearrange_job.py <job_id> [timeSig]
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.arrange import arrange  # noqa: E402
from app.config import JOBS_DIR  # noqa: E402
from app.midi_export import score_to_midi  # noqa: E402
from app.notation import score_to_musicxml  # noqa: E402


def main() -> None:
    job = sys.argv[1]
    opts = {"timeSig": sys.argv[2]} if len(sys.argv) > 2 else {}
    d = JOBS_DIR / job
    raw = json.loads((d / "raw.json").read_text(encoding="utf-8"))
    an = json.loads((d / "analysis.json").read_text(encoding="utf-8"))
    meta = json.loads((d / "job.json").read_text(encoding="utf-8"))["meta"]
    out = d / "check"
    out.mkdir(exist_ok=True)
    for mode in ("easy", "standard", "virtuoso"):
        t = time.time()
        s = arrange(raw, an, mode, opts, meta)
        xml = score_to_musicxml(s, {"fingering": "all"})
        score_to_midi(s)
        (out / f"{mode}.musicxml").write_text(xml, encoding="utf-8")
        (out / f"{mode}.json").write_text(json.dumps(s, ensure_ascii=False), encoding="utf-8")
        d_ = s["difficulty"]
        print(
            f"{mode:9s} ts={s['timeSignature']} mb={s['measureBeats']} tempo={s['tempo']} notes={len(s['notes'])} "
            f"meas={s['totalBeats'] // s['measureBeats']} diff={d_['overall']} hard={d_['difficultMeasures'][:6]} "
            f"chords={' '.join(c['symbol'] for c in s['chords'][:8])} ({time.time() - t:.2f}s)"
        )


if __name__ == "__main__":
    main()
