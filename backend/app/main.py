"""FastAPI アプリ本体。"""
from __future__ import annotations

import re
import shutil
from pathlib import Path
from urllib.parse import quote

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import __version__, pipeline, service
from .config import (
    CORS_ORIGINS,
    FRONTEND_OUT,
    MAX_ANALYZE_SEC,
    MAX_SOURCE_SEC,
    MAX_UPLOAD_MB,
    ffmpeg_path,
    torch_device,
)
from .jobs import job_dir, manager
from .separation import demucs_available
from .service import SEPARATIONS, SOURCE_TYPES, ServiceError
from .sources import SourceError, detect_platform, probe
from .transcription import basicpitch, bytedance

app = FastAPI(title="楽譜AI gakufu", version=__version__)

ALLOW_ALL_ORIGINS = "*" in CORS_ORIGINS


@app.middleware("http")
async def guard_origin(request, call_next):
    """自分の PC で動くエンジンなので、許可したページ以外からの操作は断る。

    ブラウザは別サイトへのリクエストに Origin を付ける。悪意のあるサイトが
    このエンジンに勝手に動画を取らせたり GPU を使わせたりできないようにする。
    また Chrome の「公開サイト → 自分の PC（ローカルネットワーク）」通信の
    事前確認（Private Network Access）に応える。
    """
    origin = (request.headers.get("origin") or "").rstrip("/")
    if origin and not ALLOW_ALL_ORIGINS and origin not in CORS_ORIGINS:
        return JSONResponse({"detail": "このページからは楽譜AIエンジンを操作できません"}, status_code=403)
    response = await call_next(request)
    if request.headers.get("access-control-request-private-network"):
        response.headers["Access-Control-Allow-Private-Network"] = "true"
    return response


app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Content-Disposition"],
    allow_private_network=True,
)

UPLOAD_EXT = {".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg", ".opus", ".webm", ".mp4", ".mov", ".mkv"}


# --------------------------------------------------------------------------
# 入力モデル
# --------------------------------------------------------------------------
class JobIn(BaseModel):
    url: str = Field(..., max_length=2000)
    mode: str = "standard"
    sourceType: str = "auto"
    separation: str = "auto"
    start: float | None = None
    end: float | None = None
    options: dict = Field(default_factory=dict)


class ProbeIn(BaseModel):
    url: str = Field(..., max_length=2000)


class ArrangeIn(BaseModel):
    mode: str
    options: dict = Field(default_factory=dict)
    instrument: str = "piano"
    guitar: dict = Field(default_factory=dict)


class RetranscribeIn(BaseModel):
    engine: str = "auto"


class ScoreIn(BaseModel):
    score: dict
    options: dict = Field(default_factory=dict)
    tempoScale: float = 1.0


_check_mode = service.check_mode
_clean_options = service.clean_options
_job_or_404 = service.job_or_404


@app.exception_handler(ServiceError)
def _service_error(_req, exc: ServiceError):
    return JSONResponse({"detail": exc.detail}, status_code=exc.status)


def _attachment(data: bytes | str, filename: str, media: str) -> Response:
    ascii_name = re.sub(r"[^A-Za-z0-9._-]+", "_", filename) or "download"
    disp = f"attachment; filename=\"{ascii_name}\"; filename*=UTF-8''{quote(filename)}"
    return Response(content=data, media_type=media, headers={"Content-Disposition": disp})


# --------------------------------------------------------------------------
# システム
# --------------------------------------------------------------------------
@app.get("/api/system")
def system():
    gpu = None
    try:
        import torch

        if torch_device() == "cuda":
            gpu = torch.cuda.get_device_name(0)
    except Exception:
        pass
    try:
        import yt_dlp

        ytdlp = yt_dlp.version.__version__
    except Exception:
        ytdlp = None
    return {
        "version": __version__,
        "device": torch_device(),
        "gpu": gpu,
        "ffmpeg": bool(ffmpeg_path()),
        "ytdlp": ytdlp,
        "engines": {
            "bytedance": {"available": bytedance.available(), "weights": bytedance.checkpoint_ready()},
            "demucs": demucs_available(),
            "basicpitch": basicpitch.available(),
            "builtin": True,
        },
        "limits": {"maxAnalyzeSec": MAX_ANALYZE_SEC, "maxSourceSec": MAX_SOURCE_SEC, "maxUploadMB": MAX_UPLOAD_MB},
        **service.static_info(),
    }


@app.post("/api/probe")
def probe_url(body: ProbeIn):
    try:
        return probe(body.url)
    except SourceError as e:
        raise HTTPException(400, str(e))


# --------------------------------------------------------------------------
# ジョブ
# --------------------------------------------------------------------------
@app.post("/api/jobs")
def create_job(body: JobIn):
    _check_mode(body.mode)
    try:
        platform = detect_platform(body.url)
    except SourceError as e:
        raise HTTPException(400, str(e))
    if body.sourceType not in SOURCE_TYPES or body.separation not in SEPARATIONS:
        raise HTTPException(400, "sourceType / separation の値が不正です")
    req = {
        "kind": "url",
        "url": body.url.strip(),
        "platform": platform,
        "mode": body.mode,
        "sourceType": body.sourceType,
        "separation": body.separation,
        "start": body.start,
        "end": body.end,
        "options": _clean_options(body.options),
    }
    job = manager.create(req, {"platform": platform})
    manager.submit(job, pipeline.run)
    return job.public()


@app.post("/api/jobs/upload")
async def create_job_upload(
    file: UploadFile = File(...),
    mode: str = Form("standard"),
    sourceType: str = Form("auto"),
    separation: str = Form("auto"),
    start: float | None = Form(None),
    end: float | None = Form(None),
    options: str = Form("{}"),
):
    _check_mode(mode)
    try:
        import json as _json

        opts = _clean_options(_json.loads(options or "{}"))
    except ValueError:
        opts = {}
    if sourceType not in SOURCE_TYPES or separation not in SEPARATIONS:
        raise HTTPException(400, "sourceType / separation の値が不正です")
    name = Path(file.filename or "upload").name
    ext = Path(name).suffix.lower()
    if ext not in UPLOAD_EXT:
        raise HTTPException(400, f"対応形式: {' '.join(sorted(UPLOAD_EXT))}")
    req = {
        "kind": "upload",
        "filename": name,
        "platform": "upload",
        "mode": mode,
        "sourceType": sourceType,
        "separation": separation,
        "start": start,
        "end": end,
        "options": opts,
    }
    job = manager.create(req, {"platform": "upload", "title": Path(name).stem})
    dst_dir = job_dir(job.id) / "src"
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / f"upload{ext}"
    size = 0
    with open(dst, "wb") as f:
        while chunk := await file.read(1 << 20):
            size += len(chunk)
            if size > MAX_UPLOAD_MB * 1024 * 1024:
                f.close()
                manager.delete(job.id)
                raise HTTPException(413, f"ファイルが大きすぎます（{MAX_UPLOAD_MB}MBまで）")
            f.write(chunk)
    req["uploadPath"] = str(dst)
    manager.update(job, request=req, force=True)
    manager.submit(job, pipeline.run)
    return job.public()


@app.get("/api/jobs")
def list_jobs():
    return manager.list()


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    return _job_or_404(job_id).public()


@app.post("/api/jobs/{job_id}/cancel")
def cancel_job(job_id: str):
    _job_or_404(job_id)
    return {"ok": manager.cancel(job_id)}


@app.delete("/api/jobs/{job_id}")
def delete_job(job_id: str):
    _job_or_404(job_id)
    return {"ok": manager.delete(job_id)}


@app.post("/api/jobs/{job_id}/retranscribe")
def retranscribe(job_id: str, body: RetranscribeIn):
    """ダウンロード・音源分離はそのままに、採譜AIを選び直して採譜からやり直す。"""
    job = _job_or_404(job_id)
    if job.data["status"] in ("queued", "running"):
        raise HTTPException(409, "処理中です")
    if body.engine not in ("auto", "bytedance", "basicpitch", "builtin"):
        raise HTTPException(400, "engine は auto / bytedance / basicpitch / builtin")
    if not (job_dir(job_id) / "raw.json").exists():
        raise HTTPException(409, "まだ採譜が終わっていないジョブです")
    req = dict(job.data["request"], engine=body.engine)
    steps = job.data["steps"]
    for k in ("transcribe", "analyze", "arrange"):
        steps[k]["state"] = "pending"
        steps[k]["detail"] = ""
    manager.update(job, request=req, status="queued", message="採譜をやり直します", error=None, steps=steps, force=True)
    job.cancel_flag.clear()
    manager.submit(job, pipeline.retranscribe)
    return job.public()


@app.get("/api/jobs/{job_id}/audio/{which}")
def job_audio(job_id: str, which: str):
    _job_or_404(job_id)
    if which not in ("original", "target"):
        raise HTTPException(404)
    p = job_dir(job_id) / f"{which}.mp3"
    if not p.exists():
        raise HTTPException(404, "音声がまだありません")
    return FileResponse(p, media_type="audio/mpeg")


@app.get("/api/jobs/{job_id}/score")
def get_score(job_id: str, mode: str = Query("standard"), instrument: str = Query("piano")):
    return service.get_score(job_id, mode, instrument)


@app.put("/api/jobs/{job_id}/score")
def put_score(job_id: str, body: ScoreIn, mode: str = Query("standard"), instrument: str = Query("piano")):
    return service.put_score(job_id, mode, instrument, body.score)


@app.post("/api/jobs/{job_id}/arrange")
def rearrange(job_id: str, body: ArrangeIn):
    return service.rearrange(job_id, body.mode, body.options, body.instrument, body.guitar)


@app.post("/api/jobs/{job_id}/reset")
def reset_score(job_id: str, mode: str = Query("standard"), instrument: str = Query("piano")):
    return service.reset(job_id, mode, instrument)


@app.get("/api/jobs/{job_id}/export/raw-midi")
def export_raw_midi(job_id: str):
    return _attachment(*service.raw_midi(job_id))


# --------------------------------------------------------------------------
# 楽譜処理（編集中のスコアを受け取って返す）
# --------------------------------------------------------------------------
@app.post("/api/notation/musicxml")
def notation_musicxml(body: ScoreIn):
    xml = service.musicxml(body.score, body.options)
    return Response(content=xml, media_type="application/vnd.recordare.musicxml+xml")


@app.post("/api/notation/fingering")
def notation_fingering(body: ScoreIn):
    return service.fingering(body.score)


@app.post("/api/export/tab")
def export_tab(body: ScoreIn):
    return _attachment(*service.export_tab(body.score))


@app.post("/api/export/midi")
def export_midi(body: ScoreIn):
    return _attachment(*service.export_midi(body.score, body.tempoScale))


@app.post("/api/export/musicxml")
def export_musicxml(body: ScoreIn):
    return _attachment(*service.export_musicxml(body.score, body.options))


@app.exception_handler(SourceError)
def _source_error(_req, exc: SourceError):
    return JSONResponse({"detail": str(exc)}, status_code=400)


# ビルド済みフロントエンド（next build の静的出力）があれば同じポートで配る。
# 無ければ（インストール版のエンジン）、トップは公開サイトへ案内する
if FRONTEND_OUT.exists() and (FRONTEND_OUT / "index.html").exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_OUT), html=True), name="frontend")
else:
    from fastapi.responses import RedirectResponse

    from .config import SITE_URL

    @app.get("/", include_in_schema=False)
    def _root():
        return RedirectResponse(SITE_URL)
