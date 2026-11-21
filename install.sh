#!/usr/bin/env bash
# 楽譜AI エンジンのインストール（macOS / Linux）
#
#   ターミナルに貼り付けて実行:
#     curl -fsSL https://raw.githubusercontent.com/aiueo98765/gakufu/main/install.sh | bash
#
# すべてあなたの PC の中だけで完結します。~/.gakufu に入れ、~/.gakufu/start.sh で起動します。
set -euo pipefail

REPO="${GAKUFU_REPO:-aiueo98765/gakufu}"
BRANCH="${GAKUFU_BRANCH:-main}"
ROOT="${GAKUFU_HOME:-$HOME/.gakufu}"
SITE="https://aiueo98765.github.io/gakufu/"
say() { printf '\n\033[36m>> %s\033[0m\n' "$1"; }

say "Python 3.11〜3.13 を探しています"
PY=""
for v in 3.13 3.12 3.11; do
  if command -v "python$v" >/dev/null 2>&1; then PY="python$v"; break; fi
done
if [ -z "$PY" ] && command -v python3 >/dev/null 2>&1; then
  ver="$(python3 -c 'import sys;print("%d.%d" % sys.version_info[:2])')"
  case "$ver" in 3.11|3.12|3.13) PY=python3 ;; esac
fi
if [ -z "$PY" ]; then
  echo "Python 3.11〜3.13 が見つかりません。Mac は 'brew install python@3.13'、Linux はパッケージマネージャで入れてから再実行してください。"
  exit 1
fi
echo "  使う Python: $PY"

say "エンジンを取得しています（$REPO / $BRANCH）"
mkdir -p "$ROOT"
tmp="$(mktemp -d)"
curl -fsSL "https://codeload.github.com/$REPO/tar.gz/refs/heads/$BRANCH" | tar -xz -C "$tmp"
src="$(find "$tmp" -mindepth 1 -maxdepth 1 -type d | head -n 1)"
rm -rf "$ROOT/engine"
cp -R "$src/backend" "$ROOT/engine"
rm -rf "$tmp"

if [ ! -x "$ROOT/venv/bin/python" ]; then
  say "専用の Python 環境を作っています"
  "$PY" -m venv "$ROOT/venv"
fi
VPY="$ROOT/venv/bin/python"
"$VPY" -m pip install --upgrade pip --quiet

if command -v nvidia-smi >/dev/null 2>&1; then
  say "PyTorch（NVIDIA GPU 用）を入れています。しばらくかかります"
  "$VPY" -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
elif [ "$(uname)" = "Darwin" ]; then
  say "PyTorch（Mac 用）を入れています"
  "$VPY" -m pip install torch torchaudio
else
  say "PyTorch（CPU 用）を入れています"
  "$VPY" -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
fi

say "音源分離・採譜AI のライブラリを入れています"
"$VPY" -m pip install -r "$ROOT/engine/requirements.txt"
"$VPY" -m pip install --no-deps basic-pitch==0.4.0 --quiet

say "AI の重みを取得しています（約400MB）"
export GAKUFU_DATA="$ROOT/data" GAKUFU_MODELS="$ROOT/models"
(cd "$ROOT/engine" && "$VPY" prefetch.py) || true

cat > "$ROOT/start.sh" <<EOF
#!/usr/bin/env bash
# 楽譜AI エンジンを起動（このターミナルを開いている間、サイトから採譜できます）
export GAKUFU_DATA="$ROOT/data"
export GAKUFU_MODELS="$ROOT/models"
cd "$ROOT/engine"
exec "$VPY" run.py --open
EOF
chmod +x "$ROOT/start.sh"

say "インストールが終わりました"
echo "  起動: $ROOT/start.sh"
echo "  サイト: $SITE"
exec "$ROOT/start.sh"
