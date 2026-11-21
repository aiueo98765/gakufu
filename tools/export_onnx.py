"""ブラウザ版エンジン用に、採譜・音源分離・拍検出の AI モデルを ONNX に書き出す。

使い方（backend の venv で）:
    python tools/export_onnx.py bytedance   # ピアノ採譜（音符＋ペダル）
    python tools/export_onnx.py demucs      # 音源分離 htdemucs_6s（STFT は外で行う形）
    python tools/export_onnx.py beatthis    # 拍・小節頭（Beat This!）
    python tools/export_onnx.py basicpitch  # 楽器を問わない採譜（同梱 ONNX を写すだけ）
    python tools/export_onnx.py all

書き出し先は tools/onnx/。重みは fp16 で保存し、読み込み時に fp32 へ戻す
（計算は fp32 のまま。ダウンロード量だけ半分になる）。
各モデルは PyTorch の出力と突き合わせて誤差を表示する。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "tools" / "onnx"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT / "backend"))


# --------------------------------------------------------------------------- 共通
def fp16_storage(src: Path, dst: Path, min_elems: int = 1024, keep=lambda name: False) -> None:
    """大きな fp32 の重みを fp16 で保存し、直後に Cast(fp32) を挟む。

    計算グラフは fp32 のまま。ORT はセッション作成時に Cast を定数畳み込みするので、
    推論の精度は fp16 に丸めた重みの分だけしか変わらない。
    keep(name) が真の重み（STFT の基底など、丸めに弱いもの）は fp32 のまま残す。
    """
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    m = onnx.load(str(src))
    g = m.graph
    new_inits = []
    casts = []
    for init in g.initializer:
        if init.data_type == TensorProto.FLOAT and np.prod(init.dims) >= min_elems and not keep(init.name):
            arr = numpy_helper.to_array(init).astype(np.float16)
            name16 = init.name + "__fp16"
            new_inits.append(numpy_helper.from_array(arr, name16))
            casts.append(helper.make_node("Cast", [name16], [init.name], to=TensorProto.FLOAT, name=init.name + "__cast"))
        else:
            new_inits.append(init)
    del g.initializer[:]
    g.initializer.extend(new_inits)
    nodes = list(g.node)
    del g.node[:]
    g.node.extend(casts + nodes)
    onnx.checker.check_model(m)
    onnx.save(m, str(dst))


def ort_session(path: Path):
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])


def report(name: str, ref: np.ndarray, got: np.ndarray) -> None:
    d = np.abs(ref.astype(np.float64) - got.astype(np.float64))
    print(f"  {name:28s} max|Δ|={d.max():.2e}  mean|Δ|={d.mean():.2e}  (ref max {np.abs(ref).max():.3g})")


def test_audio(sr: int, seconds: float, stereo: bool = False) -> np.ndarray:
    """手元のジョブの音源（無ければ合成音）を使う。"""
    import soundfile as sf

    jobs = ROOT / "backend" / "data" / "jobs"
    for jd in sorted(jobs.glob("*")) if jobs.exists() else []:
        p = jd / "mix.wav"
        if p.exists():
            y, s = sf.read(str(p), dtype="float32", always_2d=True)
            y = y.T
            if s != sr:
                import librosa

                y = np.stack([librosa.resample(c, orig_sr=s, target_sr=sr) for c in y])
            n = int(seconds * sr)
            off = min(int(5 * sr), max(0, y.shape[1] - n))
            y = y[:, off : off + n]
            if y.shape[1] < n:
                y = np.pad(y, ((0, 0), (0, n - y.shape[1])))
            if stereo:
                return (y if y.shape[0] == 2 else np.repeat(y[:1], 2, 0)).astype(np.float32)
            return y.mean(0).astype(np.float32)
    t = np.arange(int(seconds * sr)) / sr
    y = 0.2 * np.sin(2 * np.pi * 440 * t) * (np.sin(2 * np.pi * 2 * t) > 0)
    return np.stack([y, y]).astype(np.float32) if stereo else y.astype(np.float32)


# --------------------------------------------------------------------------- ByteDance
def export_bytedance() -> None:
    import torch
    from piano_transcription_inference.models import Note_pedal

    from app.transcription.bytedance import CKPT_PATH, ensure_checkpoint

    ensure_checkpoint()
    model = Note_pedal(frames_per_second=100, classes_num=88)
    ck = torch.load(str(CKPT_PATH), map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model"], strict=False)
    model.eval()

    names = [
        "reg_onset_output",
        "reg_offset_output",
        "frame_output",
        "velocity_output",
        "reg_pedal_onset_output",
        "reg_pedal_offset_output",
        "pedal_frame_output",
    ]

    class Wrap(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, x):
            o = self.m(x)
            # (1, T, 88) / (1, T, 1) → (T, 88) / (T,) にして返す
            return tuple(o[k][0] if o[k].shape[-1] == 88 else o[k][0, :, 0] for k in names)

    w = Wrap(model).eval()
    x = torch.from_numpy(test_audio(16000, 10.0)[None, :])
    raw = OUT / "bytedance_fp32.onnx"
    with torch.no_grad():
        ref = [t.numpy() for t in w(x)]
        torch.onnx.export(w, (x,), str(raw), input_names=["audio"], output_names=names, opset_version=17, dynamo=False)
    dst = OUT / "piano_transcription.onnx"
    # STFT の基底（cos/sin×窓）は fp16 に丸めると出力が最大 0.45 もずれる。ここだけ fp32 で持つ。
    # 40 秒の曲で音符が 196/196 一致することを確認済み（int8 だと 193/196 に落ちる）。
    fp16_storage(raw, dst, keep=lambda n: "spectrogram" in n or "logmel" in n)
    for path in (raw, dst):
        print(f"{path.name}: {path.stat().st_size / 1e6:.1f} MB")
        sess = ort_session(path)
        got = sess.run(None, {"audio": x.numpy()})
        for n, r, g in zip(names, ref, got):
            report(n, r, g)


# --------------------------------------------------------------------------- ByteDance（ブラウザ用の分割版）
# onnxruntime-web の WebGPU は GRU を「1時刻ずつ2回の GPU 呼び出し」で動かすので、
# 1区間（1001フレーム）で約13万回の呼び出しになり 14 秒かかる。そこで
#   piano_frontend.onnx : 波形 → 7つの音響モデルそれぞれの GRU 入力（畳み込み・fc5・bn5・relu まで）
#   piano_rnn.bin/json  : GRU と最後の全結合の重み（fp16）
# に分け、GRU はブラウザ側の自前の WebGPU シェーダ（1時刻1回で14系列まとめて計算）で回す。
PIANO_AMS = [
    ("note", "frame_model"),
    ("note", "reg_onset_model"),
    ("note", "reg_offset_model"),
    ("note", "velocity_model"),
    ("pedal", "reg_pedal_onset_model"),
    ("pedal", "reg_pedal_offset_model"),
    ("pedal", "reg_pedal_frame_model"),
]


def _load_note_pedal():
    import torch
    from piano_transcription_inference.models import Note_pedal

    from app.transcription.bytedance import CKPT_PATH, ensure_checkpoint

    ensure_checkpoint()
    model = Note_pedal(frames_per_second=100, classes_num=88)
    ck = torch.load(str(CKPT_PATH), map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model"], strict=False)
    return model.eval()


def np_gru_layer(x: np.ndarray, p: dict, pre: str) -> np.ndarray:
    """PyTorch の双方向 GRU 1層（numpy・検証用）。x: (T, I) → (T, 512)"""

    def run(x, wih, whh, bih, bhh):
        T = x.shape[0]
        H = whh.shape[1]
        xp = x @ wih.T + bih
        h = np.zeros(H)
        out = np.zeros((T, H))
        for t in range(T):
            hp = whh @ h + bhh
            r = 1 / (1 + np.exp(-(xp[t, :H] + hp[:H])))
            z = 1 / (1 + np.exp(-(xp[t, H : 2 * H] + hp[H : 2 * H])))
            n = np.tanh(xp[t, 2 * H :] + r * hp[2 * H :])
            h = (1 - z) * n + z * h
            out[t] = h
        return out

    f = run(x, p[pre + "weight_ih"], p[pre + "weight_hh"], p[pre + "bias_ih"], p[pre + "bias_hh"])
    b = run(x[::-1], p[pre + "weight_ih_r"], p[pre + "weight_hh_r"], p[pre + "bias_ih_r"], p[pre + "bias_hh_r"])[::-1]
    return np.concatenate([f, b], axis=1)


def export_piano_split() -> None:
    import json

    import torch
    import torch.nn.functional as F

    model = _load_note_pedal()
    nm, pm = model.note_model, model.pedal_model

    class Frontend(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.nm, self.pm = nm, pm

        @staticmethod
        def feats(am, x):
            x = am.conv_block1(x, pool_size=(1, 2), pool_type="avg")
            x = am.conv_block2(x, pool_size=(1, 2), pool_type="avg")
            x = am.conv_block3(x, pool_size=(1, 2), pool_type="avg")
            x = am.conv_block4(x, pool_size=(1, 2), pool_type="avg")
            x = x.transpose(1, 2).flatten(2)
            return F.relu(am.bn5(am.fc5(x).transpose(1, 2)).transpose(1, 2))

        def forward(self, audio):
            x = self.nm.spectrogram_extractor(audio)
            x = self.nm.logmel_extractor(x)
            xn = self.nm.bn0(x.transpose(1, 3)).transpose(1, 3)
            xp = self.pm.bn0(x.transpose(1, 3)).transpose(1, 3)
            outs = []
            for part, name in PIANO_AMS:
                src = xn if part == "note" else xp
                am = getattr(self.nm if part == "note" else self.pm, name)
                outs.append(self.feats(am, src)[0])  # (T, 768)
            return tuple(outs)

    fe = Frontend().eval()
    audio = torch.from_numpy(test_audio(16000, 10.0)[None, :])
    names = [f"{n}" for _, n in PIANO_AMS]
    raw = OUT / "piano_frontend_fp32.onnx"
    with torch.no_grad():
        ref_feats = [t.numpy() for t in fe(audio)]
        ref_full = model(audio)
        torch.onnx.export(fe, (audio,), str(raw), input_names=["audio"], output_names=names, opset_version=17, dynamo=False)
    dst = OUT / "piano_frontend.onnx"
    fp16_storage(raw, dst, keep=lambda n: "spectrogram" in n or "logmel" in n)
    print(f"  {dst.name}: {dst.stat().st_size / 1e6:.1f} MB")

    # GRU・全結合の重みを fp16 で1つのファイルに並べる
    tensors: dict[str, np.ndarray] = {}

    def add_gru(prefix: str, gru, layers: int):
        for l in range(layers):
            for suf, tag in (("", ""), ("_reverse", "_r")):
                for w in ("weight_ih", "weight_hh", "bias_ih", "bias_hh"):
                    tensors[f"{prefix}.l{l}.{w}{tag}"] = getattr(gru, f"{w}_l{l}{suf}").detach().numpy()

    for part, name in PIANO_AMS:
        am = getattr(nm if part == "note" else pm, name)
        add_gru(name, am.gru, 2)
        tensors[f"{name}.fc.weight"] = am.fc.weight.detach().numpy()
        tensors[f"{name}.fc.bias"] = am.fc.bias.detach().numpy()
    add_gru("reg_onset_gru", nm.reg_onset_gru, 1)
    tensors["reg_onset_fc.weight"] = nm.reg_onset_fc.weight.detach().numpy()
    tensors["reg_onset_fc.bias"] = nm.reg_onset_fc.bias.detach().numpy()
    add_gru("frame_gru", nm.frame_gru, 1)
    tensors["frame_fc.weight"] = nm.frame_fc.weight.detach().numpy()
    tensors["frame_fc.bias"] = nm.frame_fc.bias.detach().numpy()
    # GPU が無いときの後半（GRU・全結合）は ONNX のまま WASM で動かす（自前の JS より速い）
    class Rnn(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.nm, self.pm = nm, pm

        def forward(self, *feats):
            outs = {}
            for (part, name), f in zip(PIANO_AMS, feats):
                am = getattr(self.nm if part == "note" else self.pm, name)
                x, _ = am.gru(f[None])
                outs[name] = torch.sigmoid(am.fc(x))
            ro, vel = outs["reg_onset_model"], outs["velocity_model"]
            x, _ = self.nm.reg_onset_gru(torch.cat((ro, (ro**0.5) * vel), dim=2))
            reg_onset = torch.sigmoid(self.nm.reg_onset_fc(x))
            x, _ = self.nm.frame_gru(torch.cat((outs["frame_model"], reg_onset, outs["reg_offset_model"]), dim=2))
            frame = torch.sigmoid(self.nm.frame_fc(x))
            return (
                reg_onset[0],
                outs["reg_offset_model"][0],
                frame[0],
                vel[0],
                outs["reg_pedal_onset_model"][0, :, 0],
                outs["reg_pedal_offset_model"][0, :, 0],
                outs["reg_pedal_frame_model"][0, :, 0],
            )

    rnn_names = ["reg_onset_output", "reg_offset_output", "frame_output", "velocity_output", "reg_pedal_onset_output", "reg_pedal_offset_output", "pedal_frame_output"]
    feats_t = tuple(torch.from_numpy(f) for f in ref_feats)
    raw_rnn = OUT / "piano_rnn_fp32.onnx"
    with torch.no_grad():
        torch.onnx.export(Rnn().eval(), feats_t, str(raw_rnn), input_names=names, output_names=rnn_names, opset_version=17, dynamo=False)
    dst_rnn = OUT / "piano_rnn.onnx"
    fp16_storage(raw_rnn, dst_rnn)
    print(f"  {dst_rnn.name}: {dst_rnn.stat().st_size / 1e6:.1f} MB")
    got_rnn = ort_session(dst_rnn).run(None, {n: f for n, f in zip(names, ref_feats)})
    for k, v in zip(rnn_names, got_rnn):
        r = ref_full[k][0].numpy()
        report(f"rnn.onnx {k}", r if r.ndim == 2 and r.shape[1] == 88 else r.reshape(-1), v)

    manifest = {"dtype": "float16", "tensors": {}}
    blob = bytearray()
    for k, v in tensors.items():
        a = np.ascontiguousarray(v, dtype=np.float16)
        manifest["tensors"][k] = {"offset": len(blob), "shape": list(a.shape)}
        blob += a.tobytes()
    (OUT / "piano_rnn.bin").write_bytes(bytes(blob))
    (OUT / "piano_rnn.json").write_text(json.dumps(manifest), encoding="utf-8")
    print(f"  piano_rnn.bin: {len(blob) / 1e6:.1f} MB ({len(tensors)} tensors)")

    # 分けた形（ONNX の前半 + numpy の GRU、重みは fp16 に丸めたもの）が元のモデルと同じ出力か
    sess = ort_session(dst)
    feats = sess.run(None, {"audio": audio.numpy()})
    for n, r, g in zip(names, ref_feats, feats):
        report(f"feat {n}", r, g)
    P = {k: v.astype(np.float16).astype(np.float64) for k, v in tensors.items()}
    sig = lambda v: 1 / (1 + np.exp(-v))

    def am_out(name, x):
        h = np_gru_layer(x.astype(np.float64), P, f"{name}.l0.")
        h = np_gru_layer(h, P, f"{name}.l1.")
        return sig(h @ P[f"{name}.fc.weight"].T + P[f"{name}.fc.bias"])

    o = {n: am_out(n, f) for n, f in zip(names, feats)}
    ro, vel = o["reg_onset_model"], o["velocity_model"]
    x = np.concatenate([ro, np.sqrt(ro) * vel], axis=1)
    h = np_gru_layer(x, P, "reg_onset_gru.l0.")
    reg_onset = sig(h @ P["reg_onset_fc.weight"].T + P["reg_onset_fc.bias"])
    x = np.concatenate([o["frame_model"], reg_onset, o["reg_offset_model"]], axis=1)
    h = np_gru_layer(x, P, "frame_gru.l0.")
    frame = sig(h @ P["frame_fc.weight"].T + P["frame_fc.bias"])
    got = {
        "reg_onset_output": reg_onset,
        "reg_offset_output": o["reg_offset_model"],
        "frame_output": frame,
        "velocity_output": vel,
        "reg_pedal_onset_output": o["reg_pedal_onset_model"][:, 0],
        "reg_pedal_offset_output": o["reg_pedal_offset_model"][:, 0],
        "pedal_frame_output": o["reg_pedal_frame_model"][:, 0],
    }
    for k, v in got.items():
        r = ref_full[k][0].numpy()
        report(k, r if r.ndim == 2 and r.shape[1] == 88 else r.reshape(-1), v)


# --------------------------------------------------------------------------- Demucs
# HTDemucs は STFT/iSTFT（複素数）を含むので ONNX にできない。
# 「STFT 済みの複素スペクトル（実部・虚部をチャンネルに並べた CaC 形式）と波形」を入力に、
# 「周波数側の出力スペクトル」と「時間側の出力波形」を返す中身だけを書き出し、
# STFT/iSTFT はブラウザ側（JS）で行う。ここでは同じ計算の numpy 版で突き合わせる。
DEMUCS_NFFT = 4096
DEMUCS_HOP = 1024


def np_spectro(x: np.ndarray, n_fft: int, hop: int) -> np.ndarray:
    """torch.stft(center=True, pad_mode=reflect, window=hann, normalized=True) と同じ。x: (..., L)"""
    pad = n_fft // 2
    xp = np.pad(x, [(0, 0)] * (x.ndim - 1) + [(pad, pad)], mode="reflect")
    n_frames = 1 + (xp.shape[-1] - n_fft) // hop
    win = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n_fft) / n_fft)
    idx = np.arange(n_fft)[None, :] + hop * np.arange(n_frames)[:, None]
    frames = xp[..., idx] * win
    z = np.fft.rfft(frames, axis=-1) / np.sqrt(n_fft)
    return np.swapaxes(z, -1, -2)  # (..., F, T)


def np_ispectro(z: np.ndarray, hop: int, length: int) -> np.ndarray:
    """torch.istft(center=True, window=hann, normalized=True, length=length) と同じ。z: (..., F, T)"""
    n_fft = 2 * (z.shape[-2] - 1)
    win = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n_fft) / n_fft)
    frames = np.fft.irfft(np.swapaxes(z, -1, -2) * np.sqrt(n_fft), n=n_fft, axis=-1) * win
    T = frames.shape[-2]
    total = n_fft + hop * (T - 1)
    out = np.zeros(frames.shape[:-2] + (total,))
    wsum = np.zeros(total)
    for t in range(T):
        out[..., t * hop : t * hop + n_fft] += frames[..., t, :]
        wsum[t * hop : t * hop + n_fft] += win**2
    start = n_fft // 2
    out = out[..., start : start + length]
    w = wsum[start : start + length]
    return out / np.where(w > 1e-11, w, 1.0)


def np_demucs_spec(x: np.ndarray) -> np.ndarray:
    """HTDemucs._spec と同じ。x: (C, L) → (C, 2048, le) 複素"""
    hl = DEMUCS_HOP
    L = x.shape[-1]
    le = int(np.ceil(L / hl))
    pad = hl // 2 * 3
    xp = np.pad(x, ((0, 0), (pad, pad + le * hl - L)), mode="reflect")
    z = np_spectro(xp, DEMUCS_NFFT, hl)[..., :-1, :]
    return z[..., 2 : 2 + le]


def np_demucs_ispec(z: np.ndarray, length: int) -> np.ndarray:
    """HTDemucs._ispec と同じ。z: (..., 2048, le) 複素 → (..., length)"""
    hl = DEMUCS_HOP
    z = np.pad(z, [(0, 0)] * (z.ndim - 2) + [(0, 1), (2, 2)])
    pad = hl // 2 * 3
    le = hl * int(np.ceil(length / hl)) + 2 * pad
    x = np_ispectro(z, hl, le)
    return x[..., pad : pad + length]


def export_demucs() -> None:
    import torch
    from demucs.pretrained import get_model

    bag = get_model("htdemucs_6s")
    model = bag.models[0] if hasattr(bag, "models") else bag
    model.eval()
    L = int(model.segment * model.samplerate)
    print(f"  sources={model.sources} sr={model.samplerate} segment={float(model.segment)} → {L} samples, cac={model.cac}")

    class Core(torch.nn.Module):
        """HTDemucs.forward のうち、STFT の後から iSTFT の前まで。"""

        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, mix, mag):
            m = self.m
            x = mag
            B, C, Fq, T = x.shape
            mean = x.mean(dim=(1, 2, 3), keepdim=True)
            std = x.std(dim=(1, 2, 3), keepdim=True)
            x = (x - mean) / (1e-5 + std)
            xt = mix
            meant = xt.mean(dim=(1, 2), keepdim=True)
            stdt = xt.std(dim=(1, 2), keepdim=True)
            xt = (xt - meant) / (1e-5 + stdt)
            saved, saved_t, lengths, lengths_t = [], [], [], []
            for idx, encode in enumerate(m.encoder):
                lengths.append(x.shape[-1])
                inject = None
                if idx < len(m.tencoder):
                    lengths_t.append(xt.shape[-1])
                    tenc = m.tencoder[idx]
                    xt = tenc(xt)
                    if not tenc.empty:
                        saved_t.append(xt)
                    else:
                        inject = xt
                x = encode(x, inject)
                if idx == 0 and m.freq_emb is not None:
                    frs = torch.arange(x.shape[-2], device=x.device)
                    emb = m.freq_emb(frs).t()[None, :, :, None].expand_as(x)
                    x = x + m.freq_emb_scale * emb
                saved.append(x)
            if m.crosstransformer:
                if m.bottom_channels:
                    b, c, f, t = x.shape
                    x = x.reshape(b, c, f * t)
                    x = m.channel_upsampler(x)
                    x = x.reshape(b, x.shape[1], f, t)
                    xt = m.channel_upsampler_t(xt)
                x, xt = m.crosstransformer(x, xt)
                if m.bottom_channels:
                    b, c, f, t = x.shape
                    x = x.reshape(b, c, f * t)
                    x = m.channel_downsampler(x)
                    x = x.reshape(b, x.shape[1], f, t)
                    xt = m.channel_downsampler_t(xt)
            for idx, decode in enumerate(m.decoder):
                skip = saved.pop(-1)
                x, pre = decode(x, skip, lengths.pop(-1))
                offset = m.depth - len(m.tdecoder)
                if idx >= offset:
                    tdec = m.tdecoder[idx - offset]
                    length_t = lengths_t.pop(-1)
                    if tdec.empty:
                        pre = pre[:, :, 0]
                        xt, _ = tdec(pre, None, length_t)
                    else:
                        skip = saved_t.pop(-1)
                        xt, _ = tdec(xt, skip, length_t)
            S = len(m.sources)
            x = x.view(B, S, -1, Fq, T)
            x = x * std[:, None] + mean[:, None]
            xt = xt.view(B, S, -1, mix.shape[-1])
            xt = xt * stdt[:, None] + meant[:, None]
            return x, xt

    core = Core(model).eval()
    wav = test_audio(model.samplerate, L / model.samplerate, stereo=True)
    z = np_demucs_spec(wav.astype(np.float64))  # (2, 2048, T)
    mag = np.stack([z.real, z.imag], 1).reshape(1, 4, z.shape[-2], z.shape[-1]).astype(np.float32)
    mix_t = torch.from_numpy(wav[None])
    with torch.no_grad():
        ref_full = model(mix_t)[0].numpy()  # (S, 2, L) 公式の forward
        # 自前の STFT が torch と同じか
        z_t = model._spec(mix_t)[0].numpy()
        report("STFT (numpy vs torch)", z_t.real, z.real)
        torch_core = [t.numpy() for t in core(mix_t, torch.from_numpy(mag))]
    raw = OUT / "demucs_fp32.onnx"
    torch.onnx.export(core, (mix_t, torch.from_numpy(mag)), str(raw), input_names=["mix", "spec"], output_names=["spec_out", "wave_out"], opset_version=17, dynamo=False)
    dst = OUT / "htdemucs_6s.onnx"
    fp16_storage(raw, dst)

    def finish(spec_out, wave_out):
        S = spec_out.shape[1]
        so = spec_out[0].reshape(S, 2, 2, spec_out.shape[-2], spec_out.shape[-1])  # (S, C, re/im, F, T)
        zc = so[:, :, 0] + 1j * so[:, :, 1]
        return wave_out[0] + np_demucs_ispec(zc.astype(np.complex128), L)

    report("core torch → full", ref_full, finish(*torch_core))
    for path in (raw, dst):
        print(f"{path.name}: {path.stat().st_size / 1e6:.1f} MB")
        sess = ort_session(path)
        got = sess.run(None, {"mix": wav[None], "spec": mag})
        report("onnx full", ref_full, finish(*got))
        for i, name in enumerate(model.sources):
            r = ref_full[i]
            g = finish(*got)[i]
            sdr = 10 * np.log10((r**2).sum() / (((r - g) ** 2).sum() + 1e-20) + 1e-20)
            print(f"    {name:7s} SDR vs torch {sdr:6.1f} dB")


# --------------------------------------------------------------------------- Beat This!
# 入力はログメルスペクトログラム（22050Hz, n_fft 1024, hop 441 = 50fps, 30〜11000Hz, 128 帯域, slaney, 正規化なし, 振幅）
# の log1p(1000x)。モデルは 1500 フレーム（30 秒）固定で書き出し、短い曲は 0 で埋める。
BT_CHUNK = 1500
BT_BORDER = 6


def bt_mel_fb(sr=22050, n_fft=1024, f_min=30.0, f_max=11000.0, n_mels=128) -> np.ndarray:
    """torchaudio.functional.melscale_fbanks(norm=None, mel_scale='slaney') と同じ。(n_freqs, n_mels)"""

    def hz_to_mel(f):
        f = np.asarray(f, dtype=np.float64)
        f_sp = 200.0 / 3
        mels = f / f_sp
        min_log_hz = 1000.0
        min_log_mel = min_log_hz / f_sp
        logstep = np.log(6.4) / 27.0
        return np.where(f >= min_log_hz, min_log_mel + np.log(np.maximum(f, 1e-10) / min_log_hz) / logstep, mels)

    def mel_to_hz(m):
        m = np.asarray(m, dtype=np.float64)
        f_sp = 200.0 / 3
        freqs = f_sp * m
        min_log_hz = 1000.0
        min_log_mel = min_log_hz / f_sp
        logstep = np.log(6.4) / 27.0
        return np.where(m >= min_log_mel, min_log_hz * np.exp(logstep * (m - min_log_mel)), freqs)

    n_freqs = n_fft // 2 + 1
    all_freqs = np.linspace(0, sr // 2, n_freqs)
    m_pts = np.linspace(hz_to_mel(f_min), hz_to_mel(f_max), n_mels + 2)
    f_pts = mel_to_hz(m_pts)
    f_diff = f_pts[1:] - f_pts[:-1]
    slopes = f_pts[None, :] - all_freqs[:, None]
    down = -slopes[:, :-2] / f_diff[:-1]
    up = slopes[:, 2:] / f_diff[1:]
    return np.maximum(0.0, np.minimum(down, up))


def bt_logmel(y: np.ndarray) -> np.ndarray:
    """(L,) 22050Hz → (T, 128)"""
    z = np_spectro(y.astype(np.float64)[None], 1024, 441)[0]  # (513, T)
    mag = np.abs(z)
    mel = bt_mel_fb().T @ mag  # (128, T)
    return np.log1p(1000.0 * mel).T.astype(np.float32)


def bt_frames(sess, spect: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """beat_this.inference.split_predict_aggregate と同じ分割・統合（keep_first）。"""
    n = len(spect)
    starts = np.arange(-BT_BORDER, n - BT_BORDER, BT_CHUNK - 2 * BT_BORDER)
    if n > BT_CHUNK - 2 * BT_BORDER:
        starts[-1] = n - (BT_CHUNK - BT_BORDER)
    beat = np.full(n, -1000.0, np.float32)
    down = np.full(n, -1000.0, np.float32)
    for s in starts[::-1]:  # 前の区間の予測で後の区間を上書きする
        chunk = np.zeros((BT_CHUNK, spect.shape[1]), np.float32)
        a, b = max(s, 0), min(s + BT_CHUNK, n)
        chunk[a - s : b - s] = spect[a:b]
        ob, od = sess.run(None, {"spect": chunk[None]})
        lo, hi = s + BT_BORDER, min(s + BT_CHUNK - BT_BORDER, n)
        beat[lo:hi] = ob[0, BT_BORDER : BT_BORDER + hi - lo]
        down[lo:hi] = od[0, BT_BORDER : BT_BORDER + hi - lo]
    return beat, down


def bt_peaks(beat: np.ndarray, down: np.ndarray, fps: int = 50) -> tuple[np.ndarray, np.ndarray]:
    """Postprocessor(type='minimal') と同じ。"""

    def pick(x):
        pad = np.pad(x, 3, constant_values=-np.inf)
        mx = np.max(np.stack([pad[i : i + len(x)] for i in range(7)]), 0)
        fr = np.nonzero((x == mx) & (x > 0))[0]
        out = []
        for f in fr:  # 隣り合うピークは平均にまとめる（deduplicate_peaks width=1）
            if out and f - out[-1][-1] <= 1:
                out[-1].append(f)
            else:
                out.append([f])
        return np.array([np.mean(g) for g in out])

    bt = pick(beat) / fps
    dt = pick(down) / fps
    if len(bt):
        dt = np.array([bt[np.argmin(np.abs(bt - d))] for d in dt])
    return bt, np.unique(dt)


def export_beatthis() -> None:
    import soundfile as sf
    import torch
    import librosa
    from beat_this.inference import Audio2Beats, load_model
    from beat_this.preprocessing import LogMelSpect

    model = load_model("final0", "cpu")
    x = torch.from_numpy(bt_logmel(test_audio(22050, 30.0))[None, :BT_CHUNK])
    with torch.no_grad():
        y = test_audio(22050, 30.0)
        ref_spec = LogMelSpect()(torch.from_numpy(y)).numpy()
        report("logmel (numpy vs torch)", ref_spec, bt_logmel(y))

        class Wrap(torch.nn.Module):
            def __init__(self, m):
                super().__init__()
                self.m = m

            def forward(self, s):
                o = self.m(s)
                return o["beat"], o["downbeat"]

        w = Wrap(model).eval()
        ref = [t.numpy() for t in w(x)]
        raw = OUT / "beatthis_fp32.onnx"
        torch.onnx.export(w, (x,), str(raw), input_names=["spect"], output_names=["beat", "downbeat"], opset_version=17, dynamo=False)
    dst = OUT / "beat_this.onnx"
    fp16_storage(raw, dst)
    for path in (raw, dst):
        print(f"{path.name}: {path.stat().st_size / 1e6:.1f} MB")
        got = ort_session(path).run(None, {"spect": x.numpy()})
        report("beat logits", ref[0], got[0])
        report("downbeat logits", ref[1], got[1])

    # 曲全体で、公式の Audio2Beats と拍・小節頭が一致するか（長い曲と 30 秒未満の曲）
    a2b = Audio2Beats("final0", "cpu")
    sess = ort_session(dst)
    jobs = ROOT / "backend" / "data" / "jobs"
    tried = 0
    for jd in sorted(jobs.glob("*"), key=lambda p: -(p / "mix.wav").stat().st_size if (p / "mix.wav").exists() else 0):
        p = jd / "mix.wav"
        if not p.exists():
            continue
        y, sr = sf.read(str(p), dtype="float32", always_2d=True)
        dur = y.shape[0] / sr
        if tried == 1 and dur > 30:
            continue
        y = librosa.resample(y.mean(1), orig_sr=sr, target_sr=22050)
        rb, rd = a2b(y, 22050)
        gb, gd = bt_peaks(*bt_frames(sess, bt_logmel(y)))
        hit_b = sum(np.min(np.abs(gb - t)) < 0.021 for t in rb) if len(gb) else 0
        hit_d = sum(np.min(np.abs(gd - t)) < 0.021 for t in rd) if len(gd) else 0
        print(f"  {jd.name} {dur:5.1f}s  beats {hit_b}/{len(rb)} (ours {len(gb)})  downbeats {hit_d}/{len(rd)} (ours {len(gd)})")
        tried += 1
        if tried == 2:
            break


# --------------------------------------------------------------------------- 入口
def main() -> None:
    what = sys.argv[1:] or ["all"]
    jobs = {"bytedance": export_bytedance, "piano": export_piano_split, "demucs": export_demucs, "beatthis": export_beatthis}
    for k in jobs if "all" in what else what:
        print(f"== {k}")
        jobs[k]()


if __name__ == "__main__":
    main()
