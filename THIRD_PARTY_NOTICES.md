# 第三者のモデル・ソフトウェア

ブラウザ版エンジン（`frontend/public/engine/`）は、以下の学習済みモデルを ONNX 形式に書き出したもの
（`tools/export_onnx.py`、重みは fp16 で保存）を、GitHub Release `models-v1` から配布して使う。
モデルの中身（重み）は変えていない。

| ファイル | 元のモデル | 作者 | ライセンス |
|---|---|---|---|
| `htdemucs_6s.onnx` | Demucs v4 htdemucs_6s（音源分離） | Alexandre Défossez / Meta Platforms, Inc. — https://github.com/facebookresearch/demucs | MIT License |
| `piano_frontend.onnx` `piano_rnn.onnx` `piano_rnn.bin` `piano_rnn.json` | High-resolution Piano Transcription with Pedals（Note_pedal, `note_F1=0.9677_pedal_F1=0.9186`） | Qiuqiang Kong ほか（ByteDance）— https://github.com/qiuqiangkong/piano_transcription_inference ／ 重み: https://zenodo.org/record/4034264 | MIT License（コード）。重みは配布元の条件に従う |
| `beat_this.onnx` | Beat This!（final0） | Francesco Foscarin, Jan Schlüter, Gerhard Widmer（CPJKU, JKU Linz）— https://github.com/CPJKU/beat_this | MIT License |
| `basic_pitch.onnx` | Basic Pitch（ICASSP 2022, nmp） | Spotify AB — https://github.com/spotify/basic-pitch | Apache License 2.0 |

ブラウザ版の Python 側には、次のコードを移植・同梱している。

- `backend/app/transcription/bd_post.py` — piano_transcription_inference（Qiuqiang Kong, MIT License）の後処理
- `backend/app/transcription/bp_notes.py` — basic_pitch.note_creation（Copyright 2022 Spotify AB, Apache License 2.0）
- `backend/app/minilibrosa.py` — librosa（ISC License）の一部の関数を numpy で書き直したもの
- mido（MIT License）— MIDI の書き出し（Python 一式の zip に同梱）

実行時に CDN（jsDelivr）から読み込むもの: ONNX Runtime Web（Microsoft, MIT License）、Pyodide（Mozilla Public License 2.0）と numpy（BSD License）。

## 論文

- A. Défossez, "Hybrid Spectrogram and Waveform Source Separation", ISMIR MDX Workshop 2021 / S. Rouard, F. Massa, A. Défossez, "Hybrid Transformers for Music Source Separation", ICASSP 2023
- Q. Kong, B. Li, X. Song, Y. Wan, Y. Wang, "High-resolution Piano Transcription with Pedals by Regressing Onset and Offset Times", IEEE/ACM TASLP 2021
- F. Foscarin, J. Schlüter, G. Widmer, "Beat this! Accurate beat tracking without DBN postprocessing", ISMIR 2024
- R. M. Bittner ほか, "A Lightweight Instrument-Agnostic Model for Polyphonic Note Transcription and Multipitch Estimation", ICASSP 2022
