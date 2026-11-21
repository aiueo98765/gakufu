"""採譜ジョブの管理。1ジョブ = data/jobs/<id>/ フォルダ。

GPU を取り合わないよう既定では1件ずつ順番に処理する（GAKUFU_WORKERS で変更可）。
"""
from __future__ import annotations

import json
import shutil
import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable

from .config import JOBS_DIR, WORKERS

STEPS: list[tuple[str, str]] = [
    ("download", "音声の取得"),
    ("preprocess", "前処理（変換・正規化・無音カット）"),
    ("separate", "音源分離（Stem）"),
    ("transcribe", "AI採譜"),
    ("analyze", "テンポ・拍子・調の解析"),
    ("arrange", "難易度別アレンジ・運指・難所分析"),
]
WEIGHTS = {"download": 0.10, "preprocess": 0.04, "separate": 0.34, "transcribe": 0.38, "analyze": 0.07, "arrange": 0.07}


class JobCanceled(Exception):
    pass


def job_dir(job_id: str) -> Path:
    return JOBS_DIR / job_id


class Job:
    def __init__(self, data: dict):
        self.data = data
        self.cancel_flag = threading.Event()
        self._last_save = 0.0

    @property
    def id(self) -> str:
        return self.data["id"]

    def public(self) -> dict:
        return self.data


class JobManager:
    def __init__(self) -> None:
        self.jobs: dict[str, Job] = {}
        self.lock = threading.RLock()
        self.pool = ThreadPoolExecutor(max_workers=WORKERS, thread_name_prefix="gakufu-job")
        self._load()

    # ---- 永続化 ----
    def _load(self) -> None:
        for p in sorted(JOBS_DIR.glob("*/job.json")):
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                continue
            if data.get("status") in ("queued", "running"):
                data["status"] = "error"
                data["error"] = "サーバーの再起動で処理が中断されました。もう一度実行してください。"
            self.jobs[data["id"]] = Job(data)

    def _save(self, job: Job, force: bool = False) -> None:
        now = time.time()
        if not force and now - job._last_save < 0.4:
            return
        job._last_save = now
        d = job_dir(job.id)
        d.mkdir(parents=True, exist_ok=True)
        tmp = d / "job.json.tmp"
        tmp.write_text(json.dumps(job.data, ensure_ascii=False, indent=1), encoding="utf-8")
        tmp.replace(d / "job.json")

    # ---- 作成・取得 ----
    def create(self, request: dict, meta: dict | None = None) -> Job:
        job_id = uuid.uuid4().hex[:12]
        data = {
            "id": job_id,
            "created": time.time(),
            "updated": time.time(),
            "status": "queued",
            "stage": None,
            "progress": 0.0,
            "message": "順番待ち",
            "steps": {k: {"label": lbl, "state": "pending", "detail": ""} for k, lbl in STEPS},
            "stepOrder": [k for k, _ in STEPS],
            "request": request,
            "meta": meta or {},
            "result": None,
            "error": None,
            "log": [],
        }
        job = Job(data)
        with self.lock:
            self.jobs[job_id] = job
        job_dir(job_id).mkdir(parents=True, exist_ok=True)
        self._save(job, force=True)
        return job

    def get(self, job_id: str) -> Job | None:
        return self.jobs.get(job_id)

    def list(self) -> list[dict]:
        with self.lock:
            items = [j.data for j in self.jobs.values()]
        items.sort(key=lambda d: d["created"], reverse=True)
        return [
            {
                "id": d["id"],
                "created": d["created"],
                "status": d["status"],
                "progress": d["progress"],
                "title": d["meta"].get("title") or d["request"].get("url") or d["request"].get("filename"),
                "platform": d["meta"].get("platform") or d["request"].get("platform"),
                "thumbnail": d["meta"].get("thumbnail", ""),
                "mode": d["request"].get("mode"),
                "error": d.get("error"),
            }
            for d in items
        ]

    def delete(self, job_id: str) -> bool:
        job = self.jobs.get(job_id)
        if not job:
            return False
        job.cancel_flag.set()
        with self.lock:
            self.jobs.pop(job_id, None)
        shutil.rmtree(job_dir(job_id), ignore_errors=True)
        return True

    def cancel(self, job_id: str) -> bool:
        job = self.jobs.get(job_id)
        if not job or job.data["status"] not in ("queued", "running"):
            return False
        job.cancel_flag.set()
        if job.data["status"] == "queued":
            self.update(job, status="canceled", message="キャンセルしました", force=True)
        return True

    # ---- 進捗 ----
    def update(self, job: Job, force: bool = False, **kw: Any) -> None:
        with self.lock:
            job.data.update(kw)
            job.data["updated"] = time.time()
        self._save(job, force=force)

    def log(self, job: Job, text: str) -> None:
        with self.lock:
            lg = job.data.setdefault("log", [])
            lg.append({"t": round(time.time(), 2), "text": text})
            del lg[:-80]

    def step(self, job: Job, key: str, state: str, detail: str = "") -> None:
        with self.lock:
            st = job.data["steps"][key]
            st["state"] = state
            if detail:
                st["detail"] = detail
            if state == "running":
                job.data["stage"] = key
                job.data["message"] = st["label"]
        if detail:
            self.log(job, f"[{job.data['steps'][key]['label']}] {detail}")
        self._recalc(job, key, 0.0 if state == "running" else 1.0)
        self._save(job, force=True)

    def _recalc(self, job: Job, current: str | None, frac: float) -> None:
        total = 0.0
        for k, _ in STEPS:
            st = job.data["steps"][k]["state"]
            if st in ("done", "skipped"):
                total += WEIGHTS[k]
            elif k == current and st == "running":
                total += WEIGHTS[k] * max(0.0, min(1.0, frac))
        job.data["progress"] = round(min(total, 0.999), 4)

    def progress_cb(self, job: Job, key: str) -> Callable[[float, str], None]:
        def cb(frac: float, msg: str = "") -> None:
            if job.cancel_flag.is_set():
                raise JobCanceled()
            with self.lock:
                self._recalc(job, key, frac)
                if msg:
                    job.data["message"] = msg
                    job.data["steps"][key]["detail"] = msg
                job.data["updated"] = time.time()
            self._save(job)

        return cb

    def check_cancel(self, job: Job) -> None:
        if job.cancel_flag.is_set():
            raise JobCanceled()

    # ---- 実行 ----
    def submit(self, job: Job, fn: Callable[[Job, "JobManager"], None]) -> None:
        def runner() -> None:
            if job.cancel_flag.is_set():
                return
            self.update(job, status="running", force=True)
            try:
                fn(job, self)
            except JobCanceled:
                self.update(job, status="canceled", message="キャンセルしました", force=True)
            except Exception as e:  # 想定外のエラーも利用者に見せる
                if job.cancel_flag.is_set():
                    self.update(job, status="canceled", message="キャンセルしました", force=True)
                    return
                from .audio import AudioError
                from .sources import SourceError

                if isinstance(e, (SourceError, AudioError)):
                    msg = str(e)
                else:
                    msg = f"処理中にエラーが発生しました: {type(e).__name__}: {e}"
                    (job_dir(job.id) / "error.log").write_text(traceback.format_exc(), encoding="utf-8")
                with self.lock:
                    for k, _ in STEPS:
                        if job.data["steps"][k]["state"] == "running":
                            job.data["steps"][k]["state"] = "error"
                self.log(job, msg)
                self.update(job, status="error", error=msg, message="エラー", force=True)

        self.pool.submit(runner)


manager = JobManager()
