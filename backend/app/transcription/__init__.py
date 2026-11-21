"""採譜エンジン群。

- bytedance: ピアノ専用の高解像度 CRNN（GPU/CPU）。本命。
- builtin:   torch が無い環境向けの CQT + 倍音テンプレート NMF。依存は numpy/librosa のみ。
- pitchtrack: 歌・ベースなど単旋律の音高追跡（pYIN）。
"""
