"""動画URL（またはファイル）→ 楽譜 までの全工程。"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import audio as A
from . import beatfit, rhythm, separation, theory
from .arrange import ARRANGE_VERSION, arrange
from .config import MAX_ANALYZE_SEC
from .difficulty import measure_beats
from .jobs import Job, JobManager, job_dir
from .sources import download_audio
from .transcription import basicpitch, builtin, bytedance, pitchtrack

SEPARATION_BY_MODE = {"easy": "fast", "standard": "high", "virtuoso": "ultra"}


def save_json(path: Path, data) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def run(job: Job, jm: JobManager) -> None:
    d = job_dir(job.id)
    req = job.data["request"]
    mode = req.get("mode", "standard")

    # ---------------------------------------------------------------- 1. 取得
    jm.step(job, "download", "running")
    if req["kind"] == "url":
        src, meta = download_audio(
            req["url"], d / "src", progress=jm.progress_cb(job, "download"), should_cancel=job.cancel_flag.is_set
        )
        jm.update(job, meta={**job.data.get("meta", {}), **meta})
        dur_txt = f"（{meta['duration']:.0f}秒）" if meta.get("duration") else ""
        jm.step(job, "download", "done", f"{meta['title']}{dur_txt}")
    else:
        src = Path(req["uploadPath"])
        jm.step(job, "download", "done", f"アップロード: {req.get('filename')}")
    jm.check_cancel(job)

    # ---------------------------------------------------------------- 2. 前処理
    jm.step(job, "preprocess", "running")
    start = float(req.get("start") or 0) or None
    end = float(req.get("end") or 0) or None
    note = ""
    dur = float(job.data["meta"].get("duration") or 0)
    if end is None and dur and dur - (start or 0) > MAX_ANALYZE_SEC:
        end = (start or 0) + MAX_ANALYZE_SEC
        note = f"（長いので先頭{MAX_ANALYZE_SEC / 60:.0f}分を解析）"
    if start and end and end - start > MAX_ANALYZE_SEC:
        end = start + MAX_ANALYZE_SEC
    decoded = d / "decoded.wav"
    A.decode_to_wav(src, decoded, start=start, end=end)
    data, sr = A.read(decoded)
    data, trim_off = A.trim_silence(data, sr)
    if data.shape[1] / sr > MAX_ANALYZE_SEC:
        data = data[:, : int(MAX_ANALYZE_SEC * sr)]
    if data.shape[1] < sr * 2:
        raise A.AudioError("音声が短すぎるか、ほぼ無音でした。")
    data = A.normalize(data)
    A.write(d / "mix.wav", data, sr)
    A.encode_mp3(d / "mix.wav", d / "original.mp3")
    decoded.unlink(missing_ok=True)
    analyzed = data.shape[1] / sr
    meta = {**job.data["meta"], "analyzedDuration": round(analyzed, 2), "offset": round((start or 0) + trim_off, 3)}
    if req["kind"] != "url":
        meta.setdefault("title", Path(req.get("filename") or "upload").stem)
        meta.setdefault("platform", "upload")
        meta["duration"] = meta.get("duration") or analyzed
    jm.update(job, meta=meta)
    jm.step(job, "preprocess", "done", f"{analyzed:.1f}秒 / 44.1kHz ステレオ{note}")
    jm.check_cancel(job)

    # ---------------------------------------------------------------- 3. 音源分離
    jm.step(job, "separate", "running")
    sep_q = req.get("separation") or "auto"
    if sep_q == "auto":
        sep_q = SEPARATION_BY_MODE.get(mode, "high")
    stems: dict[str, Path] = {}
    source_req = req.get("sourceType", "auto")
    sep_info = {"quality": sep_q, "engine": None}
    if sep_q != "off":
        if separation.demucs_available():
            stems = separation.separate_demucs(d / "mix.wav", d / "stems", sep_q, jm.progress_cb(job, "separate"))
            sep_info["engine"] = f"Demucs {separation.MODEL_NAME}（{separation.QUALITY[sep_q]['label']}）"
        else:
            stems = separation.separate_hpss(d / "mix.wav", d / "stems", jm.progress_cb(job, "separate"))
            sep_info["engine"] = "HPSS（打楽器除去のみ）"
    cls = separation.classify_source(separation.stem_energies(stems)) if stems else {"kind": "piano", "ratios": {}, "drums": 0}
    kind = source_kind(cls["kind"], list(stems), source_req)

    target_label = prepare_target(d, kind, stems, cls["ratios"])
    target = d / "target.wav"
    A.encode_mp3(target, d / "target.mp3")
    ratio_txt = " / ".join(f"{k}:{v * 100:.0f}%" for k, v in cls["ratios"].items())
    jm.step(
        job,
        "separate",
        "done" if stems else "skipped",
        f"{sep_info['engine'] or '分離なし'} → 判定: {KIND_LABEL.get(kind, kind)}（{ratio_txt}）",
    )
    jm.check_cancel(job)

    # ---------------------------------------------------------------- 4. 採譜
    jm.step(job, "transcribe", "running")
    cb = jm.progress_cb(job, "transcribe")
    notes, pedals, engine, engine_id = transcribe_target(d, kind, stems, cls["ratios"], mode, req.get("engine", "auto"), cb)
    raw = save_raw(d, notes, pedals, engine, engine_id, kind, target_label)
    jm.step(job, "transcribe", "done", transcribe_detail(raw))
    jm.check_cancel(job)

    # ---------------------------------------------------------------- 5. 解析
    jm.step(job, "analyze", "running")
    y = A.load_mono(d / "mix.wav", A.ANALYSIS_SR)
    analysis = analyze(d, y, A.ANALYSIS_SR, notes, kind, cls["ratios"])
    jm.step(job, "analyze", "done", analyze_detail(analysis))
    jm.check_cancel(job)

    # ---------------------------------------------------------------- 6. アレンジ
    jm.step(job, "arrange", "running")
    score = first_arrange(d, raw, analysis, mode, req.get("options") or {}, job.data["meta"])
    jm.step(job, "arrange", "done", arrange_detail(score))
    jm.update(
        job,
        status="done",
        progress=1.0,
        message="完成",
        result=job_result(mode, kind, engine, engine_id, sep_info, target_label, analysis),
        force=True,
    )


# ---------------------------------------------------------------------------
# 採譜の後ろ半分（PC 版とブラウザ版で共通）
# ---------------------------------------------------------------------------
NO_NOTES = "音符を1つも検出できませんでした。音源の種類を「ピアノ」に変えて試すか、別の区間を指定してください。"


def save_raw(d: Path, notes, pedals, engine: str, engine_id: str, kind: str, target_label: str) -> dict:
    if not notes:
        raise A.AudioError(NO_NOTES)
    raw = {"notes": notes, "pedals": pedals, "engine": engine, "engineId": engine_id, "sourceKind": kind, "target": target_label}
    save_json(d / "raw.json", raw)
    return raw


def transcribe_detail(raw: dict) -> str:
    counts: dict[str, int] = {}
    for n in raw["notes"]:
        counts[n["role"]] = counts.get(n["role"], 0) + 1
    eid = raw["engineId"]
    return f"{raw['engine']}: " + "、".join(
        f"{(ROLE_LABEL[k] if not (k == 'piano' and eid != 'bytedance') else '楽器')} {v}音" for k, v in counts.items()
    )


def analyze(d: Path, y: np.ndarray, sr: int, notes: list[dict], kind: str, ratios: dict, dnn: dict | None = None) -> dict:
    """テンポ・拍・調を解析して analysis.json に保存する。

    dnn: ブラウザ版で先に求めた Beat This! の結果 {"beats", "downbeats"}（PC 版は refine_beats の中で求める）
    """
    beats = rhythm.track_beats(y, sr)
    key_src = [n for n in notes if n["role"] != "melody"] or notes
    key = theory.estimate_key(key_src)
    analysis = {
        "beats": beats["beats"],
        "audioBeats": beats["beats"],
        "bpm": beats["bpm"],
        "duration": beats["duration"],
        "firstOnset": rhythm.first_strong_onset(y, sr),
        "key": key,
        "sourceKind": kind,
        "ratios": ratios,
        "audioBpm": beats["bpm"],
    }
    if dnn is not None:
        if dnn.get("error"):
            analysis["dnnError"] = str(dnn["error"])[:200]
        else:
            analysis["dnnBeats"] = [round(float(x), 4) for x in dnn.get("beats", [])]
            analysis["dnnDownbeats"] = [round(float(x), 4) for x in dnn.get("downbeats", [])]
    refine_beats(analysis, notes, d)
    save_json(d / "analysis.json", analysis)
    return analysis


def analyze_detail(analysis: dict) -> str:
    key = analysis["key"]
    return (
        f"テンポ 約{analysis['bpm']:.0f} BPM ・ {key['ja']}（{key['en']}） ・ "
        f"拍の精度 {BEAT_LABEL.get(analysis['beatMethod'], analysis['beatMethod'])}"
    )


def first_arrange(d: Path, raw: dict, analysis: dict, mode: str, options: dict, meta: dict) -> dict:
    score = arrange(raw, analysis, mode, options, meta)
    save_json(d / f"score_{mode}.json", score)
    save_json(d / f"options_{mode}.json", options)
    return score


def arrange_detail(score: dict) -> str:
    return (
        f"{score['modeLabel']}: {len(score['notes'])}音 / {score['totalBeats'] // measure_beats(score)}小節 ・ "
        f"推定難易度 {score['difficulty']['overall']}（{score['difficulty']['label']}）"
    )


def job_result(mode, kind, engine, engine_id, sep_info, target_label, analysis) -> dict:
    return {
        "defaultMode": mode,
        "sourceKind": kind,
        "sourceKindLabel": KIND_LABEL.get(kind, kind),
        "engine": engine,
        "engineId": engine_id,
        "separation": sep_info,
        "target": target_label,
        "hasTarget": True,
        "bpm": analysis.get("audioBpm", analysis["bpm"]),
        "key": analysis["key"],
    }


ENGINE_LABEL = {
    "bytedance": "ByteDance 高解像度ピアノ採譜 (CRNN)",
    "basicpitch": "Basic Pitch（楽器を問わない採譜AI）",
    "builtin": "内蔵 CQT+NMF 採譜",
}


def source_kind(cls_kind: str, stem_names: list[str], source_req: str = "auto") -> str:
    """分離した音の割合による判定と、利用者の指定（音源の種類）から、曲の種類を決める。"""
    kind = cls_kind
    if source_req == "piano":
        kind = "piano_solo" if not stem_names or kind == "piano_solo" else "piano_band"
    elif source_req == "song":
        kind = "song" if "vocals" in stem_names else "instrumental"
    elif source_req == "instrumental":
        kind = "instrumental"
    if kind == "piano" and stem_names:
        kind = "piano_band"
    return kind


def target_plan(kind: str, stem_names: list[str], ratios: dict) -> tuple[list[str] | None, str]:
    """採譜にかける音の作り方：(足し合わせる Stem の名前（None なら元音源そのまま）, 説明)。"""
    stems = set(stem_names)
    accomp = sum(ratios.get(k, 0.0) for k in ("piano", "other", "guitar"))
    if kind == "piano_solo" or not stems:
        return None, ("元音源（ピアノ独奏と判定したので分離音は使わない）" if stems else "元音源")
    if kind == "piano_band":
        if "piano" in stems and ratios.get("piano", 0) >= 0.2:
            return ["piano"], "ピアノ Stem"
        return [k for k in ("piano", "other", "guitar", "harmonic") if k in stems], "ドラム・ボーカル・ベース以外"
    if kind == "song":
        if accomp < 0.08 and "vocals" in stems:
            # アカペラ：伴奏が無いので、歌声の重なり（ハモリ）そのものを採譜する
            return ["vocals"], "歌声（アカペラ：ハモリも含めて採譜）"
        return [k for k in ("piano", "other", "guitar") if k in stems], "伴奏（ピアノ+ギター+その他）＋歌メロ＋ベース"
    names = [k for k in ("piano", "other", "guitar", "vocals", "bass", "harmonic") if k in stems]
    if ratios.get("guitar", 0) >= 0.6:
        return names, "ギター（ドラム以外の全楽器）"
    return names, "ドラム以外の全楽器"


def prepare_target(d: Path, kind: str, stems: dict[str, Path], ratios: dict) -> str:
    """採譜にかける音（target.wav）を作り、その説明を返す。"""
    target = d / "target.wav"
    names, label = target_plan(kind, list(stems), ratios)
    if names is None:
        A.write(target, *A.read(d / "mix.wav"))
    elif len(names) == 1:
        A.write(target, *A.read(stems[names[0]]))
    else:
        separation.mix_stems(stems, names, target)
    return label


def choose_engine(kind: str, ratios: dict, requested: str = "auto") -> tuple[str, str]:
    """採譜AIを選ぶ。戻り値 (engine_id, profile)。

    ピアノの音ならピアノ専用の ByteDance、ギター・合唱・バンドなど
    ピアノ以外の音色なら楽器を問わない Basic Pitch。
    """
    bd, bp = bytedance.available(), basicpitch.available()
    accomp = sum(ratios.get(k, 0.0) for k in ("piano", "other", "guitar"))
    guitar = ratios.get("guitar", 0.0)
    if kind == "song" and accomp < 0.08:
        profile = "voices"
    elif guitar >= 0.6 or (kind == "instrumental" and guitar >= 0.4):
        profile = "guitar"
    else:
        profile = "ensemble"
    if requested == "bytedance" and bd:
        return "bytedance", profile
    if requested == "basicpitch" and bp:
        return "basicpitch", profile
    if requested == "builtin":
        return "builtin", profile
    piano_like = kind in ("piano_solo", "piano_band", "piano")
    if kind == "song" and accomp >= 0.08 and ratios.get("piano", 0) / max(accomp, 1e-9) >= 0.6:
        piano_like = True
    if kind == "instrumental" and ratios.get("piano", 0) >= 0.4:
        piano_like = True
    if piano_like and bd:
        return "bytedance", profile
    if bp:
        return "basicpitch", profile
    return ("bytedance" if bd else "builtin"), profile


def transcribe_target(d: Path, kind: str, stems: dict[str, Path], ratios: dict, mode: str, requested: str, cb):
    """target.wav（と歌もののボーカル・ベース）を採譜する。戻り値 (notes, pedals, 表示名, engine_id)。"""
    target = d / "target.wav"
    pedals: list[dict] = []
    song = kind == "song" and "vocals" in stems
    sub = 0.75 if song else 1.0

    def sub_cb(lo: float, hi: float):
        return lambda f, m="": cb(lo + (hi - lo) * f, m)

    engine_id, profile = choose_engine(kind, ratios, requested)
    if engine_id == "bytedance":
        if not bytedance.checkpoint_ready():
            bytedance.ensure_checkpoint(sub_cb(0, 0.1))
        y16 = A.load_mono(target, bytedance.SAMPLE_RATE)
        notes, pedals = bytedance.transcribe(y16, sub_cb(0.1, sub))
    elif engine_id == "basicpitch":
        notes = basicpitch.transcribe(target, profile, sub_cb(0, sub))
    else:
        y = A.load_mono(target, builtin.SR)
        notes = builtin.transcribe(y, builtin.SR, sensitivity=0.6 if mode == "virtuoso" else 0.45, progress=sub_cb(0, sub))
    engine = ENGINE_LABEL[engine_id]
    role = "harmony" if kind == "song" else "piano"
    for n in notes:
        n["role"] = role
    if song:
        yv = A.load_mono(stems["vocals"], pitchtrack.SR)
        mel = pitchtrack.track(yv, pitchtrack.SR, "vocals", sub_cb(0.75, 0.9))
        for n in mel:
            n["role"] = "melody"
        notes += mel
        if "bass" in stems and ratios.get("bass", 0) > 0.01:
            yb = A.load_mono(stems["bass"], pitchtrack.SR)
            bass = pitchtrack.track(yb, pitchtrack.SR, "bass", sub_cb(0.9, 1.0))
            for n in bass:
                n["role"] = "bass"
            notes += bass
        engine += " ＋ pYIN（歌メロ・ベース）"
    return notes, pedals, engine, engine_id


def retranscribe(job: Job, jm: JobManager) -> None:
    """ダウンロード・音源分離は再利用して、採譜からやり直す（採譜AIの切り替え用）。"""
    d = job_dir(job.id)
    req = job.data["request"]
    mode = req.get("mode", "standard")
    raw_old = load_json(d / "raw.json")
    analysis = load_json(d / "analysis.json")
    kind = raw_old.get("sourceKind", "piano")
    ratios = analysis.get("ratios") or {}
    stems = {p.stem: p for p in (d / "stems").glob("*.wav")} if (d / "stems").exists() else {}
    jm.step(job, "transcribe", "running")
    target_label = prepare_target(d, kind, stems, ratios)
    A.encode_mp3(d / "target.wav", d / "target.mp3")
    notes, pedals, engine, engine_id = transcribe_target(
        d, kind, stems, ratios, mode, req.get("engine", "auto"), jm.progress_cb(job, "transcribe")
    )
    if not notes:
        raise A.AudioError("音符を1つも検出できませんでした。別の採譜AIを選んでください。")
    raw = save_raw(d, notes, pedals, engine, engine_id, kind, target_label)
    jm.step(job, "transcribe", "done", transcribe_detail(raw))
    finish_retranscribe(job, jm, raw, analysis)


def finish_retranscribe(job: Job, jm: JobManager, raw: dict, analysis: dict) -> None:
    """採譜をやり直した後の解析・編曲（PC 版とブラウザ版で共通）。"""
    d = job_dir(job.id)
    mode = job.data["request"].get("mode", "standard")
    notes = raw["notes"]
    jm.step(job, "analyze", "running")
    analysis["key"] = theory.estimate_key([n for n in notes if n["role"] != "melody"] or notes)
    refine_beats(analysis, notes, d)
    save_json(d / "analysis.json", analysis)
    jm.step(job, "analyze", "done", f"テンポ 約{analysis['bpm']:.0f} BPM ・ {analysis['key']['ja']}")
    jm.step(job, "arrange", "running")
    for pat in ("score_*.json", "guitar_*.json", "edited_*.json"):
        for p in d.glob(pat):
            p.unlink(missing_ok=True)
    score = ensure_score(job.id, mode)
    result = dict(job.data.get("result") or {})
    result.update({"engine": raw["engine"], "engineId": raw["engineId"], "target": raw["target"], "key": analysis["key"], "bpm": analysis["bpm"]})
    jm.step(job, "arrange", "done", f"{score['modeLabel']}: {len(score['notes'])}音")
    jm.update(job, status="done", progress=1.0, message="完成", result=result, force=True)


BEAT_LABEL = {
    "audio": "音声の拍",
    "notes": "音符の立ち上がりで再追跡",
    "audio+fit": "音声の拍を音符に吸着",
    "notes+fit": "再追跡＋音符に吸着",
    "audio+fit+const": "一定テンポに整列",
    "notes+fit+const": "一定テンポに整列",
    "dnn": "深層学習 (Beat This!)",
    "dnn+fit": "深層学習の拍を音符に吸着",
    "dnn+fit+const": "一定テンポに整列",
}


def refine_beats(analysis: dict, notes: list[dict], jdir: Path | None = None) -> None:
    """拍を採譜した音符に合わせ込み、analysis を書き換える（beatfit.best_beats）。

    Beat This!（深層学習の拍・小節頭推定）が使えれば、その拍も候補に入れ、
    小節頭は小節マップの手がかりとして analysis["dnnDownbeats"] に残す。
    """
    from . import dnnbeat

    if jdir is not None and "dnnBeats" not in analysis and dnnbeat.available() and (jdir / "mix.wav").exists():
        try:
            r = dnnbeat.track(A.load_mono(jdir / "mix.wav", 22050), 22050)
            analysis["dnnBeats"] = [round(float(x), 4) for x in r["beats"]]
            analysis["dnnDownbeats"] = [round(float(x), 4) for x in r["downbeats"]]
        except Exception as e:  # 失敗しても従来の方法で続ける
            analysis["dnnError"] = str(e)[:200]
    src = analysis.get("audioBeats") or analysis["beats"]
    fit = beatfit.best_beats(src, notes, analysis["duration"], analysis.get("dnnBeats"))
    analysis["audioBeats"] = src
    analysis["beats"] = fit["beats"]
    analysis["beatMethod"] = fit["method"]
    analysis["beatAlignment"] = fit["scores"][fit["method"]]
    analysis["beatScores"] = fit["scores"]
    b = np.asarray(fit["beats"], dtype=float)
    analysis["bpm"] = round(60.0 / float(np.median(np.diff(b))), 2)


KIND_LABEL = {
    "piano_solo": "ピアノ独奏",
    "piano_band": "ピアノ入りアンサンブル",
    "song": "歌もの（メロディ+伴奏）",
    "instrumental": "器楽アンサンブル",
    "piano": "ピアノ",
}
ROLE_LABEL = {"piano": "ピアノ", "harmony": "伴奏", "melody": "歌メロ", "bass": "ベース"}


def ensure_score(job_id: str, mode: str, options: dict | None = None, force: bool = False) -> dict:
    """指定モードのスコアを返す。編集済みがあればそれ、無ければ生データから作る。"""
    d = job_dir(job_id)
    edited = d / f"edited_{mode}.json"
    if edited.exists() and not force:
        s = load_json(edited)
        s["edited"] = True
        return s
    gen = d / f"score_{mode}.json"
    opt_path = d / f"options_{mode}.json"
    if gen.exists() and not force:
        s = load_json(gen)
        if s.get("engineVersion") == ARRANGE_VERSION:
            return s
        # 古い計算方式で作った楽譜（編集なし）は、保存済みの編曲設定のまま作り直す
    if options is None and opt_path.exists():
        options = load_json(opt_path)
    raw = load_json(d / "raw.json")
    analysis = load_json(d / "analysis.json")
    if "beatMethod" not in analysis or "dnnBeats" not in analysis:  # 古いジョブは初回に拍を精密化して保存し直す
        refine_beats(analysis, raw["notes"], d)
        save_json(d / "analysis.json", analysis)
    job_meta = {}
    jp = d / "job.json"
    if jp.exists():
        job_meta = load_json(jp).get("meta", {})
    score = arrange(raw, analysis, mode, options or {}, job_meta)
    save_json(gen, score)
    save_json(opt_path, options or {})
    if force:
        edited.unlink(missing_ok=True)
    return score


def ensure_guitar_score(
    job_id: str,
    mode: str,
    options: dict | None = None,
    guitar_options: dict | None = None,
    force: bool = False,
) -> dict:
    """ギター譜を返す。編集済みがあればそれ、無ければ同じ難易度のピアノ編曲から作る。"""
    from .guitar import build_guitar_score

    d = job_dir(job_id)
    edited = d / f"edited_guitar_{mode}.json"
    if edited.exists() and not force:
        s = load_json(edited)
        s["edited"] = True
        return s
    gen = d / f"guitar_{mode}.json"
    opt_path = d / f"gopts_{mode}.json"
    if gen.exists() and not force:
        s = load_json(gen)
        if s.get("engineVersion") == ARRANGE_VERSION:
            return s
    saved = load_json(opt_path) if opt_path.exists() else {}
    if options is None:
        options = saved.get("arrange")
        if options is None and (d / f"options_{mode}.json").exists():
            options = load_json(d / f"options_{mode}.json")
    if guitar_options is None:
        guitar_options = saved.get("guitar") or {}
    raw = load_json(d / "raw.json")
    analysis = load_json(d / "analysis.json")
    if "beatMethod" not in analysis or "dnnBeats" not in analysis:
        refine_beats(analysis, raw["notes"], d)
        save_json(d / "analysis.json", analysis)
    job_meta = load_json(d / "job.json").get("meta", {}) if (d / "job.json").exists() else {}
    piano = arrange(raw, analysis, mode, options or {}, job_meta)
    score = build_guitar_score(piano, guitar_options)
    save_json(gen, score)
    save_json(opt_path, {"arrange": options or {}, "guitar": guitar_options})
    if force:
        edited.unlink(missing_ok=True)
    return score


def stats_summary(score: dict) -> dict:
    ps = [n["pitch"] for n in score["notes"]]
    return {
        "notes": len(ps),
        "range": [min(ps), max(ps)] if ps else None,
        "measures": score["totalBeats"] // measure_beats(score),
        "meanVel": float(np.mean([n["vel"] for n in score["notes"]])) if ps else 0,
    }
