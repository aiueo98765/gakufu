"""パス・上限値・外部ツールの場所など、環境ごとに変わる設定をまとめる。

すべて環境変数で上書きできる（README 参照）。
"""
from __future__ import annotations

import os
import shutil
from functools import lru_cache
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("GAKUFU_DATA", BASE_DIR / "data"))
JOBS_DIR = DATA_DIR / "jobs"
MODELS_DIR = Path(os.environ.get("GAKUFU_MODELS", BASE_DIR / "models"))
FRONTEND_OUT = Path(os.environ.get("GAKUFU_FRONTEND", BASE_DIR.parent / "frontend" / "out"))

HOST = os.environ.get("GAKUFU_HOST", "127.0.0.1")
PORT = int(os.environ.get("GAKUFU_PORT", "8541"))

# 解析する区間の長さの上限（秒）。区間指定なしでこれを超える動画は先頭から切り出す。
MAX_ANALYZE_SEC = float(os.environ.get("GAKUFU_MAX_ANALYZE_SEC", "900"))
# ダウンロードする元動画の長さの上限（秒）。これを超えるものは取得しない。
MAX_SOURCE_SEC = float(os.environ.get("GAKUFU_MAX_SOURCE_SEC", "3600"))
MAX_UPLOAD_MB = int(os.environ.get("GAKUFU_MAX_UPLOAD_MB", "300"))

# Instagram や鍵つきでない X の一部はログイン Cookie が要る。
# Netscape 形式の cookies.txt のパス、またはブラウザ名（chrome / edge / firefox）。
COOKIES_FILE = os.environ.get("GAKUFU_COOKIES") or None
COOKIES_FROM_BROWSER = os.environ.get("GAKUFU_COOKIES_FROM_BROWSER") or None

# "auto" | "cuda" | "cpu"
DEVICE_PREF = os.environ.get("GAKUFU_DEVICE", "auto").lower()
WORKERS = max(1, int(os.environ.get("GAKUFU_WORKERS", "1")))

# このエンジンを操作してよいページのオリジン（カンマ区切り）。
# 公開サイト（GitHub Pages）と、自分の PC で開いた画面だけ。ほかのサイトからは操作できない。
PUBLIC_SITE = os.environ.get("GAKUFU_SITE", "https://aiueo98765.github.io")
SITE_URL = os.environ.get("GAKUFU_SITE_URL", PUBLIC_SITE.rstrip("/") + "/gakufu/")
_default_origins = [
    PUBLIC_SITE,
    "http://localhost:8540",
    "http://127.0.0.1:8540",
    f"http://localhost:{PORT}",
    f"http://127.0.0.1:{PORT}",
]
CORS_ORIGINS = [o.strip().rstrip("/") for o in os.environ.get("GAKUFU_CORS", ",".join(_default_origins)).split(",") if o.strip()]

for _d in (DATA_DIR, JOBS_DIR, MODELS_DIR):
    _d.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def ffmpeg_path() -> str | None:
    """ffmpeg 実行ファイル。PATH になければ imageio-ffmpeg 同梱版を使う。"""
    env = os.environ.get("GAKUFU_FFMPEG")
    if env and Path(env).exists():
        return env
    found = shutil.which("ffmpeg")
    if found:
        return found
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


@lru_cache(maxsize=1)
def torch_device() -> str:
    """推論に使うデバイス。GPU があれば cuda。"""
    if DEVICE_PREF in ("cpu", "cuda"):
        return DEVICE_PREF
    try:
        import torch

        if torch.cuda.is_available():
            # 新しい世代の GPU で古い torch だとカーネルが無い。小さい演算で確かめる。
            try:
                (torch.ones(8, device="cuda") * 2).sum().item()
                return "cuda"
            except Exception:
                return "cpu"
    except Exception:
        pass
    return "cpu"
