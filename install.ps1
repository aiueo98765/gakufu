# 楽譜AI エンジンのインストール（Windows）
#
#   PowerShell に貼り付けて実行:
#     irm https://raw.githubusercontent.com/aiueo98765/gakufu/main/install.ps1 | iex
#
# やること（すべてあなたの PC の中だけで完結します）
#   1. Python 3.11〜3.13 を探す
#   2. エンジン（backend）を %LOCALAPPDATA%\gakufu\engine に置く（更新時は作り直す。採譜した曲は残る）
#   3. 専用の Python 環境を作り、PyTorch（NVIDIA GPU があれば CUDA 版）と AI ライブラリを入れる
#   4. 採譜AIの重みを前もって取得する
#   5. デスクトップに「楽譜AIエンジン」のショートカットを作って起動する

function Install-GakufuEngine {
    $ErrorActionPreference = "Stop"
    $Repo = if ($env:GAKUFU_REPO) { $env:GAKUFU_REPO } else { "aiueo98765/gakufu" }
    $Branch = if ($env:GAKUFU_BRANCH) { $env:GAKUFU_BRANCH } else { "main" }
    $Root = if ($env:GAKUFU_HOME) { $env:GAKUFU_HOME } else { Join-Path $env:LOCALAPPDATA "gakufu" }
    $Site = "https://aiueo98765.github.io/gakufu/"
    function Say($m) { Write-Host "`n>> $m" -ForegroundColor Cyan }

    # ---- 1. Python ----
    Say "Python 3.11〜3.13 を探しています"
    $py = $null
    if (Get-Command py -ErrorAction SilentlyContinue) {
        foreach ($v in "3.13", "3.12", "3.11") {
            & py "-$v" -c "import sys" 2>$null
            if ($LASTEXITCODE -eq 0) { $py = @("py", "-$v"); break }
        }
    }
    if (-not $py) {
        foreach ($cmd in "python", "python3") {
            if (Get-Command $cmd -ErrorAction SilentlyContinue) {
                $ver = & $cmd -c "import sys;print('%d.%d' % sys.version_info[:2])" 2>$null
                if ($ver -in "3.11", "3.12", "3.13") { $py = @($cmd); break }
            }
        }
    }
    if (-not $py) {
        Write-Host "Python 3.11〜3.13 が見つかりませんでした。" -ForegroundColor Yellow
        Write-Host "  winget install Python.Python.3.13" -ForegroundColor Yellow
        Write-Host "を実行して Python を入れてから、もう一度このコマンドを実行してください。" -ForegroundColor Yellow
        return
    }
    $pyExe = $py[0]
    $pyArgs = @($py | Select-Object -Skip 1)
    Write-Host "  使う Python: $($py -join ' ')"

    # ---- 2. エンジン本体 ----
    Say "エンジンを取得しています（$Repo / $Branch）"
    New-Item -ItemType Directory -Force $Root | Out-Null
    $tmp = Join-Path $env:TEMP ("gakufu-" + [guid]::NewGuid().ToString("N"))
    New-Item -ItemType Directory -Force $tmp | Out-Null
    $zip = Join-Path $tmp "src.zip"
    Invoke-WebRequest -UseBasicParsing "https://codeload.github.com/$Repo/zip/refs/heads/$Branch" -OutFile $zip
    Expand-Archive -Path $zip -DestinationPath $tmp -Force
    $src = Get-ChildItem $tmp -Directory | Select-Object -First 1
    $engine = Join-Path $Root "engine"
    if (Test-Path $engine) { Remove-Item -Recurse -Force $engine }
    Copy-Item -Recurse (Join-Path $src.FullName "backend") $engine
    Remove-Item -Recurse -Force $tmp

    # ---- 3. Python 環境と AI ライブラリ ----
    $venv = Join-Path $Root "venv"
    $vpy = Join-Path $venv "Scripts\python.exe"
    if (-not (Test-Path $vpy)) {
        Say "専用の Python 環境を作っています"
        & $pyExe @pyArgs -m venv $venv
    }
    & $vpy -m pip install --upgrade pip --quiet
    $hasNvidia = [bool](Get-Command nvidia-smi -ErrorAction SilentlyContinue)
    if ($hasNvidia) {
        Say "PyTorch（NVIDIA GPU 用・約3GB）を入れています。しばらくかかります"
        & $vpy -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
    } else {
        Say "PyTorch（CPU 用）を入れています（NVIDIA の GPU が見つからなかったので CPU で動かします）"
        & $vpy -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
    }
    if ($LASTEXITCODE -ne 0) { Write-Host "PyTorch を入れられませんでした。ネット接続を確かめてもう一度実行してください。" -ForegroundColor Red; return }
    Say "音源分離・採譜AI のライブラリを入れています"
    & $vpy -m pip install -r (Join-Path $engine "requirements.txt")
    if ($LASTEXITCODE -ne 0) { Write-Host "ライブラリを入れられませんでした。表示されたエラーを確かめてください。" -ForegroundColor Red; return }
    # Basic Pitch は TensorFlow 無し（同梱の ONNX モデル）で使う
    & $vpy -m pip install --no-deps basic-pitch==0.4.0 --quiet

    # ---- 4. AI の重み ----
    Say "採譜AI・音源分離AI・拍のAI の重みを取得しています（約400MB）"
    $env:GAKUFU_DATA = Join-Path $Root "data"
    $env:GAKUFU_MODELS = Join-Path $Root "models"
    Push-Location $engine
    try { & $vpy prefetch.py } finally { Pop-Location }

    # ---- 5. 起動用のファイルとショートカット ----
    $start = Join-Path $Root "start.cmd"
    $cmdText = @"
@echo off
chcp 65001 > nul
title gakufu engine
set "GAKUFU_DATA=$Root\data"
set "GAKUFU_MODELS=$Root\models"
cd /d "$engine"
"$vpy" run.py --open
pause
"@
    # ユーザー名に日本語があっても読めるよう UTF-8（BOM なし）で書く。1行目の chcp 65001 で UTF-8 として読まれる
    [IO.File]::WriteAllText($start, $cmdText.Replace("`n", "`r`n"), (New-Object System.Text.UTF8Encoding $false))
    $desktop = [Environment]::GetFolderPath("Desktop")
    $ws = New-Object -ComObject WScript.Shell
    $lnk = $ws.CreateShortcut((Join-Path $desktop "楽譜AIエンジン.lnk"))
    $lnk.TargetPath = $start
    $lnk.WorkingDirectory = $engine
    $lnk.Description = "楽譜AI エンジン（この窓が開いている間、サイトから採譜できます）"
    $lnk.Save()

    Say "インストールが終わりました"
    Write-Host "  次からはデスクトップの「楽譜AIエンジン」をダブルクリックして起動します。"
    Write-Host "  サイト: $Site"
    Start-Process $start
}

Install-GakufuEngine
