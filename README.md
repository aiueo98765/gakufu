# 楽譜AI（gakufu） — 動画URLからピアノ譜へ

YouTube・X (Twitter)・Instagram の動画URL（または音声/動画ファイル）から音声を取り出し、
音源分離と AI 採譜でピアノ譜にする Web アプリ。
**簡単 / 標準 / 超プロ (Virtuoso)** の3段階で自動アレンジし、**ピアノ譜**と**ギターのタブ譜**（五線譜＋TAB）を作る。
五線譜とピアノロールで確認・修正、再生（スロー・小節ループ・片手練習・メトロノーム）、運指（ピアノの指番号／ギターの弦とフレット）の自動付与、
PDF / MIDI / MusicXML / テキスト TAB の書き出しまでできる。

```
frontend/         Next.js 16 (App Router) + React 19 + Tailwind CSS 4 + OpenSheetMusicDisplay + Tone.js
frontend/public/engine/  ブラウザ版エンジン（Web Worker・onnxruntime-web(WebGPU)・Pyodide・自前の WebGPU GRU）
backend/          FastAPI + yt-dlp + ffmpeg + Demucs(htdemucs_6s) + ByteDance ピアノ採譜(CRNN) + librosa（PC 版エンジン）
```

## 使い方（公開サイト・インストール不要）

**サイト: https://aiueo98765.github.io/gakufu/** を開くだけで使えます。

音源分離・AI 採譜・拍の推定・編曲・楽譜の書き出しは、**開いた人のブラウザの中（その PC の GPU / WebGPU）**で行います。
サイト運営側のサーバーはなく、音声も楽譜もどこにも送られません（曲のデータはブラウザの IndexedDB に保存）。
初回だけ AI モデル（約190MB）を読み込み、2回目からはブラウザに保存したものを使います。

- **ファイル**：音声・動画ファイル（mp3 / m4a / wav / mp4 など）を選ぶ。
- **動画 URL**（YouTube・X・Instagram）：サーバーが無いので動画を直接は取得できない。代わりに
  ①「動画を別のタブで開く」→ ②「録音を始める」で動画のタブを選び「タブの音声も共有する」をオン → 動画を再生 → ③「止めて楽譜にする」。
  パソコンの Chrome / Edge で使える（タブの音声の共有に対応したブラウザ）。
- 処理はそのタブの中で行うので、終わるまでタブを閉じたり再読み込みしたりしない。
- WebGPU が使えないブラウザでは CPU で動く（1分の曲で5分ほど、音源分離は省略）。

### しくみ（ブラウザ版エンジン）

| 部分 | ブラウザでの実行方法 |
|---|---|
| 音源分離 Demucs htdemucs_6s | STFT/iSTFT（複素数）を外に出した中身だけを ONNX にし、onnxruntime-web の WebGPU で実行。STFT・逆変換・重なりの重み付け・時間ずらしの平均は JS（`dsp.js`）で PC 版と同じ計算 |
| ピアノ採譜 ByteDance | 畳み込み部分は ONNX（WebGPU）。双方向 GRU は onnxruntime-web だと 1 時刻ずつ GPU を呼ぶため 1 区間 14 秒かかるので、**自前の WebGPU シェーダ**（7 モデル×2 方向を 1 回の呼び出しで 1 時刻進める、`pianornn.js`）で 0.16 秒に。GPU が無いときは GRU も ONNX（WASM）で |
| 楽器を問わない採譜 Basic Pitch / 拍 Beat This! | ONNX（WebGPU） |
| それ以外（音符への変換・拍の精密化・編曲・五線譜・MIDI・TAB） | PC 版と同じ Python のコードを **Pyodide** で実行。librosa の代わりに同じ計算をする numpy 版（`minilibrosa.py`）、scipy の代わりに `npcompat.py` |

モデルは `tools/export_onnx.py` で PyTorch から書き出し（重みは fp16 で保存、計算は fp32）、各段階で元のモデルとの一致を確かめている
（ピアノ採譜は音符が 196/196 一致、Beat This! は拍・小節頭が全一致、Demucs は元との差 50dB 以上）。
`tools/check_minilibrosa.py`・`tools/check_postprocess.py`・`tools/check_webapi.py` で Python 側が本物の librosa / 元の実装と同じ結果になることを確認できる。

## 使い方（PC 版エンジン・任意）

画面右上の「PCエンジン」に切り替えると、PC にインストールした「楽譜AIエンジン」（PyTorch / CUDA）で処理する。
URL から動画を直接取得できるのはこちら。

1. Python 3.11〜3.13 を入れる（Windows: `winget install Python.Python.3.13` / Mac: `brew install python@3.13`）
2. エンジンを入れる（10〜20分・約8GB）
   - Windows（PowerShell）:
     ```powershell
     irm https://raw.githubusercontent.com/aiueo98765/gakufu/main/install.ps1 | iex
     ```
   - Mac / Linux（ターミナル）:
     ```bash
     curl -fsSL https://raw.githubusercontent.com/aiueo98765/gakufu/main/install.sh | bash
     ```
3. 起動する：Windows はデスクトップの「楽譜AIエンジン」、Mac / Linux は `~/.gakufu/start.sh`。黒い窓が開いている間、サイトから採譜できます。
   ブラウザが「このデバイス上のアプリへのアクセス」の許可を求めたら許可してください。

NVIDIA の GPU があれば自動で GPU 版の PyTorch を入れます（無ければ CPU で動き、1曲に数分かかります）。
もう一度同じコマンドを実行するとエンジンが最新版に更新されます（採譜した曲は残ります）。

**安全のため**、エンジンは自分の PC（127.0.0.1）でだけ待ち受け、公開サイトと自分の PC で開いた画面以外のページからの操作は拒否します。

---

## 処理の流れ

| 段階 | 中身 | 主なファイル |
|---|---|---|
| 1. 音声の取得 | yt-dlp で音声のみ取得。対応ホストを許可リストで限定（SSRF 対策）。長すぎる動画は取得前に弾く | `backend/app/sources.py` |
| 2. 前処理 | ffmpeg で 44.1kHz ステレオ WAV に。区間切り出し・前後の無音カット・ピーク正規化 | `backend/app/audio.py` |
| 3. 音源分離 | Demucs **htdemucs_6s** で drums / bass / other / vocals / guitar / piano の6本に。難易度で品質を変える（超プロは時間ずらし5回のアンサンブル）。Stem のエネルギー比から「ピアノ独奏 / ピアノ入り / 歌もの / 器楽」を判定 | `backend/app/separation.py` |
| 4. AI採譜 | 音色で採譜AIを自動で選ぶ：ピアノは **ByteDance 高解像度ピアノ採譜**（Kong et al. 2021, 立ち上がり・離鍵・強さ・ペダル）、ギター・合唱（アカペラ）・バンドは楽器を問わない **Basic Pitch**（Spotify, ONNX で実行）。歌ものは歌メロとベースを **pYIN** で別に追う。アカペラは歌声のトラックそのものを採譜してハモリも拾う。torch が無い環境では内蔵の **CQT + 倍音テンプレート NMF**。取得済みのジョブは画面の「情報」タブから採譜AIを選び直してやり直せる | `backend/app/transcription/` |
| 5. 解析 | 拍の検出（テンポ地図）・拍子の推定（4/4・3/4・6/8・12/8。並べ替え検定で周期の強さを比べる）・小節頭の推定（低音・長い音・和音の変わり目、コードが変わる位置への補正）・調の推定（Krumhansl）・コード認識（テンプレート + ビタビ） | `rhythm.py` `theory.py` |
| 5a. 小節マップ | 4/4 の途中に 2/4・3/4 の小節が挟まる曲（J-POP に多い）でも小節線がずれ続けないよう、拍ごとに「小節の何拍目か」を隠れマルコフモデルで追う。手がかりは低音・長い音・和音の変わり目・**Beat This!**（深層学習の拍・小節頭推定, CPJKU ISMIR 2024）の小節頭。楽譜・MIDI・TAB はその小節で拍子記号を切り替える | `barmap.py` `bars.py` `dnnbeat.py` |
| 5b. 拍の精密化 | 音声から取った拍は打楽器の無い曲で 30〜80ms ずれる。採譜した音の立ち上がりで拍を取り直し／音に吸着させた候補を作り、「16分・3連の格子への乗り」が十分よい候補の中から「小節構造が一番はっきりした列」を選ぶ。速い曲を半分のテンポで取っていたら（32分の位置の音が多く密な曲）自動で×2 | `beatfit.py` |
| 5c. 雑音除去 | 打ち直しの二重検出・倍音の聴き違い・まわりより明らかに弱い残響音・ぽつんと短い音・鍵盤の端の弱い音を除く（強さは前後2秒の中央値との相対で判定。超プロでも雑音は残さない） | `denoise.py` |
| 6. アレンジ | 左右の手の振り分け（ビタビ）→ 拍ごとに 2/4/8分割 or 3/6連符を選ぶ量子化（音価が 75ms を切る細かさは選びにくい）→ 難易度別の編曲 → 運指 → 難所分析 | `hands.py` `quantize.py` `arrange.py` `fingering.py` `difficulty.py` |
| 6b. ギター | 同じ難易度のピアノ編曲から、簡単=単音メロディ＋コード名、標準=メロディ＋ベース＋強拍の内声、超プロ=押さえられる限り全部。メロディはフレーズごとに弾きやすいオクターブへ。弦とフレットは「指の開き・指の数・ポジションの高さ・移動・伸ばしている弦を切らない」をコストにビタビで決め、押さえられない和音は内声から減らす。調弦5種とカポ（自動推奨あり） | `guitar.py` |
| 7. 浄書 | MusicXML 4.0（大譜表・2声部・連符・タイ・連桁・臨時記号・音部記号変更・8va・コード・ペダル・強弱・指番号・トリル・アルペジオ） | `notation.py` |
| 8. 書き出し | MIDI（楽譜どおり / AI 生データ）、MusicXML、PDF（ブラウザで OSMD → svg2pdf のベクター PDF） | `midi_export.py` `frontend/lib/pdf.ts` |

採譜（重い処理）は1回だけ。モードの切り替えや編曲設定の変更は、保存した生データから数百ミリ秒で作り直す。

### 難易度モード

- **簡単**：右手は単旋律メロディ（8分音符まで・フレーズごとに弾きやすい音域へ移動）。左手はコードから「ルート / ルート+5度 / 3和音」。ハ長調・イ短調への移調も可。
- **標準**：原曲の音から、片手3音・1オクターブ以内の和音に整理。トリルは tr 記号、前打音は省略、速すぎる走句は16分に間引く。歌ものは「歌メロ + 強拍に和音」と「分散和音などの伴奏型」で作る。
- **超プロ (Virtuoso)**：一音も削らない。32分・6連符・3連符を拍ごと・片手ごとに判定（左右で違う連符＝ポリリズムも書ける。6/8・12/8 では2連符・4連符）。10度を超える和音はアルペジオ記号、ペダル・強弱も付ける。

### 運指（指番号）

Parncutt ら (1997) の運指コスト（指の組ごとの楽な幅・限界幅、弱い指、黒鍵の親指、指くぐり…）を片手ごとにビタビで最小化。
表示は「難所だけ / すべて / なし」。指を手で決めた音（黄色）は固定したまま再計算できる。
難所は「毎秒の打鍵数・和音の大きさと開き・跳躍の速さ・連符・運指コスト・両手の独立」から小節ごとに採点し、ワンクリックでその区間をループ練習できる。

---

## セットアップ（Windows / macOS / Linux）

必要なもの：Python 3.11〜3.13、Node.js 20.9+、ffmpeg（無ければ imageio-ffmpeg 同梱版を自動で使う）。
NVIDIA GPU があると分離・採譜が速い（CPU でも動く）。

### バックエンド

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\activate          # macOS/Linux: source .venv/bin/activate
# torch は GPU に合わせて（RTX 50 系は CUDA 12.8 版、CPU だけなら --index-url なし）
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
python run.py                      # http://127.0.0.1:8541
```

初回の採譜で ByteDance の重み（約165MB, zenodo）と Demucs の重み（約80MB）を自動で取得する。

### フロントエンド

```powershell
cd frontend
npm install
npm run dev                        # http://localhost:8540 （API は :8541 を見に行く）
```

### 本番（1ポートで配る）

```powershell
cd frontend; npm run build         # frontend/out/ に静的書き出し
cd ../backend; python run.py       # http://127.0.0.1:8541 で画面も API も配る
```

---

## 環境変数（backend）

| 変数 | 既定 | 説明 |
|---|---|---|
| `GAKUFU_HOST` / `GAKUFU_PORT` | 127.0.0.1 / 8541 | 待ち受け |
| `GAKUFU_DEVICE` | auto | `cuda` / `cpu` を強制 |
| `GAKUFU_MAX_ANALYZE_SEC` | 900 | 1回に解析する最大秒数（超えた分は区間指定で） |
| `GAKUFU_MAX_SOURCE_SEC` | 3600 | 取得する動画の最大長 |
| `GAKUFU_MAX_UPLOAD_MB` | 300 | アップロード上限 |
| `GAKUFU_COOKIES` | なし | Netscape 形式 cookies.txt のパス（Instagram・一部の X 投稿で必要） |
| `GAKUFU_COOKIES_FROM_BROWSER` | なし | `chrome` / `edge` / `firefox` などから Cookie を読む |
| `GAKUFU_WORKERS` | 1 | 同時に処理するジョブ数（GPU を取り合うので通常 1） |
| `GAKUFU_FFMPEG` | なし | ffmpeg の場所を明示 |
| `GAKUFU_CORS` | * | 許可するオリジン（カンマ区切り） |
| `GAKUFU_DATA` / `GAKUFU_MODELS` | backend/data, backend/models | 作業データ・重みの置き場 |

フロントの `NEXT_PUBLIC_API_BASE` で API の URL を固定できる（未設定なら :8540 で開いたときは :8541、それ以外は同じオリジン）。

---

## API

| メソッド | パス | 内容 |
|---|---|---|
| GET | `/api/system` | GPU・使えるエンジン・上限値 |
| POST | `/api/probe` | `{url}` → 題名・長さ・サムネイル（ダウンロードしない） |
| POST | `/api/jobs` | `{url, mode, sourceType, separation, start, end, options}` → ジョブ作成 |
| POST | `/api/jobs/upload` | multipart（file, mode, sourceType, separation, start, end, options） |
| GET | `/api/jobs` / `/api/jobs/{id}` | 一覧 / 進捗（段階ごとの状態・詳細） |
| POST | `/api/jobs/{id}/cancel` | 中止 |
| POST | `/api/jobs/{id}/retranscribe` | `{engine: auto / bytedance / basicpitch}` 採譜AIを選び直して採譜からやり直す |
| DELETE | `/api/jobs/{id}` | 削除 |
| GET | `/api/jobs/{id}/audio/{original\|target}` | 原曲 / 採譜に使った音（mp3） |
| GET | `/api/jobs/{id}/score?mode=&instrument=` | 楽譜データ（`instrument` は piano / guitar。編集済みがあればそれ） |
| PUT | `/api/jobs/{id}/score?mode=` | 編集の保存 |
| POST | `/api/jobs/{id}/arrange` | `{mode, options, instrument, guitar:{tuning, capo}}` 生データから作り直す |
| POST | `/api/jobs/{id}/reset?mode=` | 編集を破棄 |
| GET | `/api/jobs/{id}/export/raw-midi` | AI の生データ MIDI（量子化前） |
| POST | `/api/notation/musicxml` | `{score, options}` → MusicXML |
| POST | `/api/notation/fingering` | `{score}` → 運指（ギターは弦・フレット）と難所を再計算した score |
| POST | `/api/export/tab` | ギター譜 → テキスト TAB |
| POST | `/api/export/midi` / `/api/export/musicxml` | ダウンロード用 |

`options`（編曲）: `tempoScale`(1/2/0.5), `bpm`, `timeSig`(auto/4/4/3/4/2/4/6/8/12/8), `transpose`(半音), `easyKey`, `leftHand`(auto/original/root/fifth/triad/block/broken/alberti)

---

## 画面の操作

- 五線譜の小節クリック＝そこから再生、Shift+クリック＝ループ範囲
- ピアノロール：ダブルクリックで追加、ドラッグで移動（音高・位置）、右端ドラッグで長さ、空き地ドラッグで範囲選択、ルーラーで再生位置、Shift+ルーラーでループ、Ctrl+ホイールで拡大、左の鍵盤クリックで試聴
- キー：`Space` 再生、`↑↓` 半音、`Shift+↑↓` オクターブ、`←→` 位置、`[ ]` 長さ、`H` 右手⇄左手、`1-5` 指番号、`0` 指を自動に、`Del` 削除、`Ctrl+Z / Ctrl+Y` 元に戻す/やり直し、`L` ループ、`M` クリック
- 再生の音源：楽譜をピアノで（片手ミュート可）/ 原曲 / 採譜に使った分離音。どれもテンポ 25〜150%（原曲はピッチを保ったままスロー）

## 検証用スクリプト

```powershell
cd backend
.\.venv\Scripts\python tests\make_synth.py synth.wav          # 答えのわかっている曲を合成
.\.venv\Scripts\python tests\eval_synth.py bytedance synth.wav # 採譜の適合率・再現率
.\.venv\Scripts\python tests\test_engine.py synth.wav          # 採譜→3モード→MusicXML/MIDI
.\.venv\Scripts\python tests\test_song_path.py                 # 歌もの：歌声の音高追跡→伴奏型アレンジ
.\.venv\Scripts\python tests\rearrange_job.py <job_id> [6/8]   # 保存済みジョブを作り直して検算
.\.venv\Scripts\python tests\test_guitar.py <job_id> [dropd] [2] # ギター譜：弦・フレットの整合と TAB
```

起動は `.\start.ps1`（開発：2つのサーバーを別窓で）／`.\start.ps1 -Prod`（ビルドして :8541 だけで配る）。

## 注意

- AI 採譜は完璧ではない。とくに歌ものの伴奏・テンポの揺れが大きい曲は手直し前提。拍子の自動判定が外れたら「編曲の設定」で 4/4・3/4・2/4・6/8・12/8 を指定し直す（9/8 などは未対応）。
- 音声は採譜のためだけに使う（ブラウザ版はそのブラウザの中だけ、PC 版はその PC の中だけ）。著作物の楽譜化・配布は各サービスの規約と著作権法の範囲で。
- ピアノの再生音は Salamander Grand Piano のサンプル（tonejs.github.io から読み込み、読めなければシンセで代用）。
- ギターの再生音は本物のギターの録音（[nbrosowsky/tonejs-instruments](https://github.com/nbrosowsky/tonejs-instruments) のナイロン弦・スチール弦, CC BY 3.0）。和音は低い弦から少しずらして鳴らし、同じ弦の前の音は止める。読めなければ撥弦合成で代用。

## クレジット

- 音源分離: [Demucs](https://github.com/facebookresearch/demucs)（Meta, MIT）
- ピアノ採譜: [High-resolution Piano Transcription](https://github.com/qiuqiangkong/piano_transcription_inference)（ByteDance, Kong et al., MIT）
- 楽器を問わない採譜: [Basic Pitch](https://github.com/spotify/basic-pitch)（Spotify, Apache 2.0）
- 拍・小節頭: [Beat This!](https://github.com/CPJKU/beat_this)（CPJKU, JKU Linz, MIT）
- ブラウザでの推論: [ONNX Runtime Web](https://github.com/microsoft/onnxruntime)（Microsoft, MIT）／Python の実行: [Pyodide](https://github.com/pyodide/pyodide)（MPL 2.0）
- 動画の取得（PC 版）: [yt-dlp](https://github.com/yt-dlp/yt-dlp)
- 楽譜の表示: [OpenSheetMusicDisplay](https://github.com/opensheetmusicdisplay/opensheetmusicdisplay)
- ピアノの音: Salamander Grand Piano（Tone.js 配布）／ギターの音: [tonejs-instruments](https://github.com/nbrosowsky/tonejs-instruments)（CC BY 3.0）

ブラウザ版で配布している AI モデル（ONNX に書き出したもの）の出典とライセンスは [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) にまとめた。
各モデル・ライブラリ・音源はそれぞれのライセンスに従います。