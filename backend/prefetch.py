"""最初の採譜を待たせないよう、AI の重みを前もって取得する（インストール時に実行）。

python prefetch.py
どれかが失敗しても、残りは続ける（最初の採譜のときに改めて取得される）。
"""
import logging
import sys

logging.getLogger().setLevel(logging.ERROR)
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def step(label, fn):
    print(f"  - {label} ...", flush=True)
    try:
        fn()
        print("    OK", flush=True)
    except Exception as e:  # 取得に失敗しても全体は止めない
        print(f"    取得できませんでした（最初の採譜のときに再挑戦します）: {e}", flush=True)


def piano():
    from app.transcription import bytedance

    bytedance.ensure_checkpoint()


def demucs():
    from demucs.pretrained import get_model

    get_model("htdemucs_6s")


def beats():
    from beat_this.inference import load_model

    load_model("final0", "cpu")


def basic_pitch():
    from app.transcription import basicpitch

    if not basicpitch.available():
        raise RuntimeError("basic-pitch / onnxruntime が入っていません")


if __name__ == "__main__":
    step("ピアノ採譜AI（ByteDance, 約165MB）", piano)
    step("音源分離AI（Demucs htdemucs_6s, 約80MB）", demucs)
    step("拍・小節頭のAI（Beat This!, 約80MB）", beats)
    step("楽器を問わない採譜AI（Basic Pitch）", basic_pitch)
