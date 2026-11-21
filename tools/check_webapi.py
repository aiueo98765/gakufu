"""ブラウザ版エンジンの Python 側（app/webapi.py）を、PC 上で同じ条件（librosa・scipy なし）で通す。

    python tools/check_webapi.py [ジョブID]

既存ジョブの target.wav / mix.wav を使い、JS がやる部分（ONNX の推論）は onnxruntime で代わりに行い、
webapi の関数を JS と同じ順で呼ぶ。最後に PC 版の結果（raw.json の音符数など）と比べる。
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "tools"))
TMP = Path(tempfile.mkdtemp(prefix="gakufu-web-"))
os.environ["GAKUFU_DATA"] = str(TMP)

# ブラウザと同じく scipy・librosa は使えないことにする
import soundfile  # noqa: E402,F401  （音声の読み込みは検証用に使う）

from app import audio as A  # noqa: E402

for k in list(sys.modules):
    if k == "scipy" or k.startswith("scipy.") or k == "librosa" or k.startswith("librosa."):
        del sys.modules[k]
sys.modules["scipy"] = None
from app import minilibrosa  # noqa: E402

sys.modules["librosa"] = minilibrosa

from app import webapi as W  # noqa: E402
from export_onnx import bt_frames, bt_logmel, bt_peaks, ort_session  # noqa: E402

SRC = ROOT / "backend" / "data" / "jobs"
MODELS = ROOT / "frontend" / "public" / "engine" / "models"


def main() -> None:
    jid_src = sys.argv[1] if len(sys.argv) > 1 else "77f5f8cdea0b"
    sd = SRC / jid_src
    ref_raw = json.loads((sd / "raw.json").read_text(encoding="utf-8"))
    ref_an = json.loads((sd / "analysis.json").read_text(encoding="utf-8"))
    job = json.loads(W.job_create(json.dumps({"kind": "upload", "filename": "x.wav", "platform": "upload", "mode": "standard", "sourceType": "auto", "separation": "auto", "options": {}}), json.dumps({"title": "test"})))
    jid = job["id"]
    print("job", jid, "status", job["status"])
    # 分離した音のエネルギー（PC 版で分けた stem を使う）
    stems = {p.stem: p for p in (sd / "stems").glob("*.wav")} if (sd / "stems").exists() else {}
    energies = {k: float(np.mean(np.square(A.read(p)[0]))) for k, p in stems.items()} if stems else None
    plan = json.loads(W.plan(json.dumps(energies) if energies else "", "auto", "auto"))
    print("plan", {k: plan[k] for k in ("kind", "use", "engineId", "profile", "song")}, "PC版:", ref_raw["sourceKind"], ref_raw.get("engineId"))
    W.job_step(jid, "download", "done", "test")
    W.job_step(jid, "preprocess", "done", "test")
    W.job_step(jid, "separate", "done", "test")
    W.job_step(jid, "transcribe", "running")
    # 採譜（ONNX、JS と同じ区切り方）
    if plan["engineId"] == "bytedance":
        from app.transcription import bd_post

        y = A.load_mono(sd / "target.wav", 16000)
        seg = bd_post.SEGMENT_SAMPLES
        n = int(np.ceil(len(y) / seg)) * seg
        a = np.zeros(n, np.float32)
        a[: len(y)] = y
        s = ort_session(MODELS / "piano_transcription.onnx")
        names = [o.name for o in s.get_outputs()]
        outs = {k: [] for k in names}
        for st in range(0, n - seg + 1, seg // 2):
            for k, v in zip(names, s.run(None, {"audio": a[None, st : st + seg]})):
                outs[k].append(v if v.ndim == 2 else v[:, None])
        od = {k: bd_post.deframe(np.stack(v)) for k, v in outs.items()}
        T = od["frame_output"].shape[0]
        r = json.loads(W.bytedance_notes(T, *(od[k].reshape(-1) for k in names)))
        main_notes, pedals = r["notes"], r["pedals"]
    else:
        from app.transcription import bp_notes

        y = A.load_mono(sd / "target.wav", 22050)
        s = ort_session(MODELS / "basic_pitch.onnx")
        audio = np.concatenate([np.zeros(bp_notes.OVERLAP_LEN // 2, np.float32), y])
        notes_l, onsets_l = [], []
        for i in range(0, len(audio), bp_notes.HOP_SIZE):
            w = audio[i : i + bp_notes.AUDIO_N_SAMPLES]
            w = np.pad(w, (0, bp_notes.AUDIO_N_SAMPLES - len(w)))
            o = dict(zip([x.name for x in s.get_outputs()], s.run(None, {s.get_inputs()[0].name: w[None, :, None]})))
            notes_l.append(o["StatefulPartitionedCall:1"][0])
            onsets_l.append(o["StatefulPartitionedCall:2"][0])
        main_notes = json.loads(W.basicpitch_notes(len(notes_l), len(y), np.stack(notes_l).reshape(-1), np.stack(onsets_l).reshape(-1), plan["profile"]))
        pedals = []
    mel = bass = ""
    if plan["song"]:
        mel = W.track_part(A.load_mono(stems["vocals"], 16000), "vocals")
        bass = W.track_part(A.load_mono(stems["bass"], 16000), "bass") if plan["bass"] else "[]"
    asm = json.loads(W.assemble_notes(plan["engineId"], plan["kind"], json.dumps(main_notes), mel, bass))
    detail = W.save_raw(jid, json.dumps(asm["notes"]), json.dumps(pedals), asm["engine"], plan["engineId"], plan["kind"], plan["label"])
    W.job_step(jid, "transcribe", "done", detail)
    print("transcribe:", detail)
    print("  PC版:", len(ref_raw["notes"]), "音", ref_raw["engine"])
    # 解析（Beat This! も ONNX で）
    W.job_step(jid, "analyze", "running")
    y22 = A.load_mono(sd / "mix.wav", 22050)
    bt = ort_session(MODELS / "beat_this.onnx")
    b, d = bt_peaks(*bt_frames(bt, bt_logmel(y22)))
    r = json.loads(W.analyze(jid, y22, plan["kind"], json.dumps(plan["ratios"]), json.dumps({"beats": b.tolist(), "downbeats": d.tolist()})))
    W.job_step(jid, "analyze", "done", r["detail"])
    print("analyze:", r["detail"])
    print("  PC版: bpm", ref_an["bpm"], ref_an.get("beatMethod"), ref_an["key"]["ja"])
    W.job_step(jid, "arrange", "running")
    r = json.loads(W.arrange_first(jid))
    W.job_step(jid, "arrange", "done", r["detail"])
    W.job_finish(jid, json.dumps({"quality": "high", "engine": "test"}), plan["kind"], asm["engine"], plan["engineId"], plan["label"])
    print("arrange:", r["detail"])
    # API
    for method, path, q, body in (
        ("GET", "/api/jobs", "", ""),
        ("GET", f"/api/jobs/{jid}", "", ""),
        ("GET", f"/api/jobs/{jid}/score", "mode=standard&instrument=piano", ""),
        ("GET", f"/api/jobs/{jid}/score", "mode=virtuoso&instrument=guitar", ""),
        ("POST", f"/api/jobs/{jid}/arrange", "", json.dumps({"mode": "easy", "options": {"leftHand": "root"}})),
        ("GET", f"/api/jobs/{jid}/export/raw-midi", "", ""),
        ("GET", "/api/static", "", ""),
        ("GET", "/api/jobs/000000000000", "", ""),
    ):
        res = json.loads(W.handle(method, path, q, body))
        brief = {k: (v if k in ("status", "filename", "type") else (len(v) if isinstance(v, (str, list)) else (sorted(v)[:5] if isinstance(v, dict) else v))) for k, v in res.items()}
        print(method, path, q, "→", brief)
    score = json.loads(W.handle("GET", f"/api/jobs/{jid}/score", "mode=standard&instrument=piano", ""))["json"]
    for path in ("/api/notation/musicxml", "/api/export/midi", "/api/export/musicxml", "/api/notation/fingering"):
        res = json.loads(W.handle("POST", path, "", json.dumps({"score": score, "options": {}})))
        print("POST", path, "→", res["status"], res.get("filename") or res.get("type") or "", len(res.get("text") or res.get("b64") or json.dumps(res.get("json"))))
    print("data dir:", TMP)


if __name__ == "__main__":
    main()
