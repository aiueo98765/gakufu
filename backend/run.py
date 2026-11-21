"""楽譜AI エンジンの起動スクリプト。

python run.py           … エンジンを起動（http://127.0.0.1:8541）
python run.py --open    … 起動して、ブラウザで公開サイトを開く
python run.py --reload  … 開発用（コードの変更で自動再起動）
"""
import sys
import threading
import webbrowser

import uvicorn

from app.config import HOST, PORT, SITE_URL

if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    print("=" * 60)
    print(" 楽譜AI エンジン")
    print(f"  エンジン : http://{HOST}:{PORT}")
    print(f"  サイト   : {SITE_URL}")
    print("  この窓を開いている間、サイトから採譜できます。終了は窓を閉じるか Ctrl+C。")
    print("=" * 60, flush=True)
    if "--open" in sys.argv:
        threading.Timer(2.5, lambda: webbrowser.open(SITE_URL)).start()
    uvicorn.run(
        "app.main:app",
        host=HOST,
        port=PORT,
        reload="--reload" in sys.argv,
        reload_dirs=["app"] if "--reload" in sys.argv else None,
        log_level="warning" if "--quiet" in sys.argv else "info",
    )
