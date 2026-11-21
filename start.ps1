# 楽譜AI 開発用の起動スクリプト（Windows PowerShell）
#   .\start.ps1          … API(:8541) と画面(:8540) を別ウィンドウで起動
#   .\start.ps1 -Prod    … 画面をビルドして API サーバー1つ(:8541)で配る
param([switch]$Prod)

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$py = Join-Path $root "backend\.venv\Scripts\python.exe"
if (-not (Test-Path $py)) {
    Write-Host "backend\.venv がありません。README の「セットアップ」を先に実行してください。" -ForegroundColor Yellow
    exit 1
}

if ($Prod) {
    Push-Location (Join-Path $root "frontend")
    npm run build
    Pop-Location
    Write-Host "http://127.0.0.1:8541 を開いてください" -ForegroundColor Cyan
    Push-Location (Join-Path $root "backend")
    & $py run.py
    Pop-Location
    exit 0
}

Start-Process -FilePath $py -ArgumentList "run.py" -WorkingDirectory (Join-Path $root "backend")
Start-Process -FilePath "npm.cmd" -ArgumentList "run", "dev" -WorkingDirectory (Join-Path $root "frontend")
Write-Host "API: http://127.0.0.1:8541  /  画面: http://localhost:8540" -ForegroundColor Cyan
