"use client";

import { useState } from "react";
import { Check, Copy, Cpu, Download, LoaderCircle, Play, RefreshCw, Settings2, ShieldCheck } from "lucide-react";
import { DEFAULT_ENGINE, engineUrl, setEngineUrl } from "@/lib/api";

const REPO = process.env.NEXT_PUBLIC_REPO_URL || "https://github.com/aiueo98765/gakufu";
// インストールスクリプトは GitHub の生ファイル（UTF-8 のテキスト）として配る
const RAW = process.env.NEXT_PUBLIC_RAW_URL || "https://raw.githubusercontent.com/aiueo98765/gakufu/main";

type OS = "win" | "mac" | "linux";

function detectOS(): OS {
  if (typeof navigator === "undefined") return "win";
  const ua = navigator.userAgent;
  if (/Mac/i.test(ua)) return "mac";
  if (/Linux/i.test(ua) && !/Android/i.test(ua)) return "linux";
  return "win";
}

function CopyLine({ text }: { text: string }) {
  const [done, setDone] = useState(false);
  return (
    <div className="flex items-stretch gap-2">
      <code className="block flex-1 overflow-x-auto whitespace-nowrap rounded-lg bg-black/50 px-3 py-2 font-mono text-[12px] text-emerald-200">{text}</code>
      <button
        className="btn btn-ghost !px-2.5"
        onClick={async () => {
          try {
            await navigator.clipboard.writeText(text);
            setDone(true);
            setTimeout(() => setDone(false), 1500);
          } catch {
            /* クリップボードが使えない環境 */
          }
        }}
        title="コピー"
      >
        {done ? <Check className="size-4 text-emerald-300" /> : <Copy className="size-4" />}
      </button>
    </div>
  );
}

/**
 * 楽譜AIエンジン（各自の PC で動く採譜サーバー）につながらないときの案内。
 * 処理はすべて使う人の PC（の GPU）で行い、このサイトのサーバーは使わない。
 */
export function EngineGate({ onRetry, checking }: { onRetry: () => void; checking?: boolean }) {
  const [os, setOs] = useState<OS>(detectOS());
  const [showCustom, setShowCustom] = useState(false);
  const [url, setUrl] = useState(engineUrl());
  const install =
    os === "win"
      ? `irm ${RAW}/install.ps1 | iex`
      : `curl -fsSL ${RAW}/install.sh | bash`;
  const start = os === "win" ? "デスクトップの「楽譜AIエンジン」をダブルクリック" : "~/.gakufu/start.sh";
  return (
    <div className="glass mx-auto max-w-3xl rounded-3xl p-6 sm:p-8">
      <div className="flex items-start gap-3">
        <div className="grid size-11 shrink-0 place-items-center rounded-xl bg-gradient-to-br from-indigo-500 to-fuchsia-500">
          <Cpu className="size-6 text-white" />
        </div>
        <div>
          <h2 className="text-xl font-black">あなたのPCで「楽譜AIエンジン」を動かしてください</h2>
          <p className="mt-1 text-sm text-ink-300">
            音源分離と採譜AIは、使う人それぞれのPC（GPU）で動きます。動画の取得・解析・楽譜はすべてあなたのPCの中で完結し、
            このサイトにはアップロードされません。最初に1回だけ入れれば、あとは起動するだけです。
          </p>
        </div>
      </div>

      <div className="mt-5 flex gap-1.5">
        {(
          [
            ["win", "Windows"],
            ["mac", "Mac"],
            ["linux", "Linux"],
          ] as const
        ).map(([k, l]) => (
          <button key={k} className={`btn btn-ghost !py-1 text-xs ${os === k ? "btn-active" : ""}`} onClick={() => setOs(k)}>
            {l}
          </button>
        ))}
      </div>

      <ol className="mt-4 space-y-5">
        <li>
          <div className="mb-1.5 flex items-center gap-2 text-sm font-bold">
            <span className="grid size-6 place-items-center rounded-full bg-white/10 text-xs">1</span>
            必要なもの
          </div>
          <ul className="ml-8 list-disc space-y-1 text-xs text-ink-300">
            <li>
              Python 3.11〜3.13（
              <a className="text-indigo-300 underline" href="https://www.python.org/downloads/" target="_blank" rel="noreferrer">
                python.org
              </a>
              {os === "win" ? "、または PowerShell で winget install Python.Python.3.13" : os === "mac" ? "、または brew install python@3.13" : ""}）
            </li>
            <li>空き容量 約8GB（AIの重み・PyTorch を含む）</li>
            <li>{os === "mac" ? "Mac は CPU で動きます（数分かかります）" : "NVIDIA の GPU があると速い（無ければ CPU で動きますが数分かかります）"}</li>
          </ul>
        </li>
        <li>
          <div className="mb-1.5 flex items-center gap-2 text-sm font-bold">
            <span className="grid size-6 place-items-center rounded-full bg-white/10 text-xs">2</span>
            <Download className="size-4" /> インストール（{os === "win" ? "PowerShell" : "ターミナル"}に貼り付けて実行・10〜20分）
          </div>
          <div className="ml-8">
            <CopyLine text={install} />
            <p className="mt-1.5 text-[11px] text-ink-400">
              中身を確かめてから実行したい場合は{" "}
              <a className="text-indigo-300 underline" href={`${REPO}/blob/main/install.${os === "win" ? "ps1" : "sh"}`} target="_blank" rel="noreferrer">
                スクリプト
              </a>{" "}
              と{" "}
              <a className="text-indigo-300 underline" href={REPO} target="_blank" rel="noreferrer">
                ソースコード
              </a>{" "}
              を見られます。
            </p>
          </div>
        </li>
        <li>
          <div className="mb-1.5 flex items-center gap-2 text-sm font-bold">
            <span className="grid size-6 place-items-center rounded-full bg-white/10 text-xs">3</span>
            <Play className="size-4" /> 起動
          </div>
          <div className="ml-8 space-y-1.5">
            {os === "win" ? <p className="text-sm text-ink-200">{start}</p> : <CopyLine text={start} />}
            <p className="text-[11px] text-ink-400">
              黒い窓（エンジン）が開いている間だけ使えます。ブラウザが「このデバイス上のアプリへのアクセス」の許可を求めたら「許可」してください。
            </p>
          </div>
        </li>
      </ol>

      <div className="mt-6 flex flex-wrap items-center gap-2">
        <button className="btn btn-primary" onClick={onRetry} disabled={checking}>
          {checking ? <LoaderCircle className="size-4 animate-spin" /> : <RefreshCw className="size-4" />}
          エンジンにつなぐ
        </button>
        <button className="btn btn-ghost text-xs" onClick={() => setShowCustom((v) => !v)}>
          <Settings2 className="size-3.5" /> 接続先を変える
        </button>
        <span className="flex items-center gap-1 text-[11px] text-ink-400">
          <ShieldCheck className="size-3.5" /> エンジンはこのサイトと、あなたのPCで開いた画面からしか操作できません
        </span>
      </div>
      {showCustom && (
        <div className="mt-3 flex gap-2">
          <input className="field flex-1 !py-1.5 text-sm" value={url} onChange={(e) => setUrl(e.target.value)} placeholder={DEFAULT_ENGINE} />
          <button
            className="btn btn-ghost"
            onClick={() => {
              setEngineUrl(url.trim());
              onRetry();
            }}
          >
            保存してつなぐ
          </button>
        </div>
      )}
    </div>
  );
}
