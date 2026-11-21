"""ブラウザ版エンジンが Pyodide に読み込む Python 一式（zip）を作る。

    python tools/build_web_engine.py

backend/app の Python（サーバー専用の main.py などを除く）と、MIDI 書き出しに使う mido を
frontend/public/engine/py/gakufu-<内容のハッシュ>.zip にまとめ、manifest.json に名前を書く。
（ハッシュ付きの名前なので、中身が変わったときだけブラウザが取り直す）
GitHub Pages への公開時も CI でこれを実行する。
"""
from __future__ import annotations

import hashlib
import io
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
APP = ROOT / "backend" / "app"
OUT = ROOT / "frontend" / "public" / "engine" / "py"
# サーバー専用（FastAPI・yt-dlp・PyTorch を使う部分）。ブラウザ版では読まない
EXCLUDE = {"main.py"}


def _add_tree(z: zipfile.ZipFile, src: Path, arc: str) -> None:
    for p in sorted(src.rglob("*.py")):
        if "__pycache__" in p.parts or (src == APP and p.relative_to(src).as_posix() in EXCLUDE):
            continue
        info = zipfile.ZipInfo(f"{arc}/{p.relative_to(src).as_posix()}", date_time=(2020, 1, 1, 0, 0, 0))
        info.compress_type = zipfile.ZIP_DEFLATED
        z.writestr(info, p.read_bytes())


def main() -> None:
    import mido

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        _add_tree(z, APP, "app")
        _add_tree(z, Path(mido.__file__).parent, "mido")
    data = buf.getvalue()
    h = hashlib.sha256(data).hexdigest()[:12]
    OUT.mkdir(parents=True, exist_ok=True)
    for old in OUT.glob("gakufu-*.zip"):
        old.unlink()
    name = f"gakufu-{h}.zip"
    (OUT / name).write_bytes(data)
    (OUT / "manifest.json").write_text(json.dumps({"zip": name, "bytes": len(data)}), encoding="utf-8")
    print(f"{name}: {len(data) / 1024:.0f} KB")


if __name__ == "__main__":
    main()
