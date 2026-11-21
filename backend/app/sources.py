"""YouTube / X(Twitter) / Instagram の動画URLから音声を取り出す。

yt-dlp を Python から呼ぶ。許可するホストを明示的に絞り（SSRF 対策）、
長すぎる動画は落とす前に弾く。
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Callable
from urllib.parse import urlparse

from .config import COOKIES_FILE, COOKIES_FROM_BROWSER, MAX_SOURCE_SEC, ffmpeg_path

PLATFORM_HOSTS: dict[str, tuple[str, ...]] = {
    "youtube": ("youtube.com", "youtu.be", "youtube-nocookie.com"),
    "x": ("x.com", "twitter.com"),
    "instagram": ("instagram.com",),
}
PLATFORM_LABEL = {"youtube": "YouTube", "x": "X (Twitter)", "instagram": "Instagram", "upload": "ファイル"}


class SourceError(Exception):
    """利用者に見せるメッセージを持つ取得エラー。"""


def detect_platform(url: str) -> str:
    """URL がどのサービスか。対象外なら SourceError。"""
    url = url.strip()
    if not re.match(r"^https?://", url, re.I):
        url = "https://" + url
    try:
        parsed = urlparse(url)
    except ValueError as e:
        raise SourceError("URLの形式が正しくありません。") from e
    if parsed.scheme not in ("http", "https"):
        raise SourceError("http(s) の URL を入力してください。")
    host = (parsed.hostname or "").lower().rstrip(".")
    for platform, roots in PLATFORM_HOSTS.items():
        for root in roots:
            if host == root or host.endswith("." + root):
                return platform
    raise SourceError("対応しているのは YouTube・X (Twitter)・Instagram の動画URLです。")


def normalize_url(url: str) -> str:
    url = url.strip()
    if not re.match(r"^https?://", url, re.I):
        url = "https://" + url
    return url


def _ydl_base_opts() -> dict:
    opts: dict = {
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
        "retries": 3,
        "fragment_retries": 3,
        "socket_timeout": 30,
        "noprogress": True,
        "restrictfilenames": True,
    }
    ff = ffmpeg_path()
    if ff:
        opts["ffmpeg_location"] = ff
    if COOKIES_FILE and Path(COOKIES_FILE).exists():
        opts["cookiefile"] = COOKIES_FILE
    elif COOKIES_FROM_BROWSER:
        opts["cookiesfrombrowser"] = (COOKIES_FROM_BROWSER,)
    return opts


def _friendly_error(platform: str, err: Exception) -> SourceError:
    msg = str(err)
    low = msg.lower()
    if any(k in low for k in ("login", "log in", "sign in", "cookies", "rate-limit", "authentication")):
        hint = (
            "この投稿は取得にログインが必要です。"
            "ブラウザの cookies.txt を GAKUFU_COOKIES に設定するか、"
            "GAKUFU_COOKIES_FROM_BROWSER=chrome のように指定して再起動してください。"
        )
        return SourceError(hint)
    if "private" in low:
        return SourceError("非公開（または鍵つき）の投稿は取得できません。")
    if "no video" in low or "no media" in low or "there's no video" in low:
        return SourceError("この投稿には動画（音声）が含まれていません。")
    if "unsupported url" in low:
        return SourceError("動画のページURLではないようです。投稿・動画そのもののURLを入れてください。")
    if "not available" in low or "unavailable" in low or "removed" in low:
        return SourceError("動画が見つからないか、地域・年齢制限で再生できません。")
    if "ffmpeg" in low:
        return SourceError("ffmpeg が見つかりません。ffmpeg を入れるか GAKUFU_FFMPEG で場所を指定してください。")
    return SourceError(f"{PLATFORM_LABEL.get(platform, platform)} から音声を取得できませんでした: {msg[:300]}")


def probe(url: str) -> dict:
    """ダウンロードせずに題名・長さなどを取る。"""
    import yt_dlp

    platform = detect_platform(url)
    url = normalize_url(url)
    try:
        with yt_dlp.YoutubeDL(_ydl_base_opts()) as ydl:
            info = ydl.extract_info(url, download=False)
    except Exception as e:  # yt_dlp.utils.DownloadError など
        raise _friendly_error(platform, e) from e
    if info.get("_type") == "playlist":
        entries = [e for e in (info.get("entries") or []) if e]
        if not entries:
            raise SourceError("この投稿には動画が含まれていません。")
        info = entries[0]
    return _meta_from_info(info, platform, url)


def _meta_from_info(info: dict, platform: str, url: str) -> dict:
    title = info.get("title") or info.get("fulltitle") or "無題"
    # X の題名は「ユーザー名 - 本文」になりがちなので長すぎるときは詰める
    title = re.sub(r"\s+", " ", title).strip()
    if len(title) > 120:
        title = title[:118] + "…"
    return {
        "title": title,
        "uploader": info.get("uploader") or info.get("channel") or info.get("uploader_id") or "",
        "duration": float(info.get("duration") or 0.0),
        "thumbnail": info.get("thumbnail") or "",
        "webpage_url": info.get("webpage_url") or url,
        "platform": platform,
        "extractor": info.get("extractor_key") or "",
    }


def download_audio(
    url: str,
    out_dir: Path,
    progress: Callable[[float, str], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> tuple[Path, dict]:
    """音声だけを取得して out_dir/source.<ext> に保存する。"""
    import yt_dlp

    platform = detect_platform(url)
    url = normalize_url(url)
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = probe(url)
    if meta["duration"] and meta["duration"] > MAX_SOURCE_SEC:
        raise SourceError(
            f"動画が長すぎます（{meta['duration'] / 60:.0f}分）。"
            f"{MAX_SOURCE_SEC / 60:.0f}分以内の動画にしてください。"
        )

    def hook(d: dict) -> None:
        if should_cancel and should_cancel():
            raise yt_dlp.utils.DownloadCancelled("canceled")
        if d.get("status") == "downloading" and progress:
            total = d.get("total_bytes") or d.get("total_bytes_estimate") or 0
            got = d.get("downloaded_bytes") or 0
            frac = (got / total) if total else 0.0
            progress(min(frac, 1.0), f"ダウンロード中 {got / 1e6:.1f}MB")

    opts = _ydl_base_opts()
    opts.update(
        {
            # 音声のみのストリームを優先。X / Instagram は映像付きしか無いことが多いのでフォールバック
            "format": "bestaudio/best",
            "outtmpl": str(out_dir / "source.%(ext)s"),
            "progress_hooks": [hook],
            "max_filesize": 800 * 1024 * 1024,
            "overwrites": True,
        }
    )
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=True)
    except yt_dlp.utils.DownloadCancelled as e:
        raise SourceError("キャンセルされました。") from e
    except Exception as e:
        raise _friendly_error(platform, e) from e

    if info.get("_type") == "playlist":
        entries = [e for e in (info.get("entries") or []) if e]
        info = entries[0] if entries else info

    files = sorted(
        (p for p in out_dir.glob("source.*") if p.suffix not in (".part", ".ytdl", ".json")),
        key=lambda p: p.stat().st_size,
        reverse=True,
    )
    if not files:
        raise SourceError("音声ファイルが保存されませんでした。")
    meta.update(_meta_from_info(info, platform, url))
    return files[0], meta
