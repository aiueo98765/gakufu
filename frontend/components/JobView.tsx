"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { ArrowLeft, LoaderCircle, RotateCcw } from "lucide-react";
import { api, engineMode } from "@/lib/api";
import { rerunLocalJob } from "@/lib/localEngine";
import type { Job } from "@/lib/types";
import { JobProgress } from "./JobProgress";
import { Workspace } from "./Workspace";
import { EngineGate } from "./EngineGate";

/**
 * 1 曲の画面（処理中は進捗、完成したら楽譜）。/job/?id=… のページと、
 * ブラウザ版で採譜を始めた直後のトップページ（ページを移動しない）の両方で使う。
 */
export function JobView({ id, onBack, onRerun }: { id: string; onBack: () => void; onRerun?: (newId: string) => void }) {
  const [job, setJob] = useState<Job | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [ver, setVer] = useState(0);
  const [rerunning, setRerunning] = useState(false);

  // 採譜のやり直しなどで状態が変わったら、もう一度見に行く
  useEffect(() => {
    const on = () => setVer((v) => v + 1);
    window.addEventListener("gakufu:job-reload", on);
    return () => window.removeEventListener("gakufu:job-reload", on);
  }, []);

  // ブラウザ版はこのタブの中で処理しているので、処理中にタブを閉じる・再読み込みする前に確かめる
  const running = !!job && (job.status === "queued" || job.status === "running");
  useEffect(() => {
    if (!running || engineMode() !== "browser") return;
    const warn = (e: BeforeUnloadEvent) => {
      e.preventDefault();
      e.returnValue = "";
    };
    window.addEventListener("beforeunload", warn);
    return () => window.removeEventListener("beforeunload", warn);
  }, [running]);

  useEffect(() => {
    if (!id) return;
    let stop = false;
    let timer: ReturnType<typeof setTimeout>;
    setJob(null);
    const tick = async () => {
      try {
        const j = await api.job(id);
        if (stop) return;
        setJob(j);
        setErr(null);
        if (j.status === "queued" || j.status === "running") timer = setTimeout(tick, 900);
      } catch (e) {
        if (stop) return;
        setErr((e as Error).message);
        timer = setTimeout(tick, 3000);
      }
    };
    tick();
    return () => {
      stop = true;
      clearTimeout(timer);
    };
  }, [id, ver]);

  // ブラウザ版：中断した曲を、保存してある元の音声からもう一度実行する
  const rerun = async () => {
    if (!job) return;
    setRerunning(true);
    try {
      const j = await rerunLocalJob(job);
      onRerun?.(j.id);
    } catch (e) {
      alert((e as Error).message);
      setRerunning(false);
    }
  };

  if (!job) {
    if (err && err.includes("エンジン")) {
      return (
        <main className="bg-aurora min-h-screen px-4 py-10">
          <EngineGate onRetry={() => window.location.reload()} />
        </main>
      );
    }
    return <Center>{err ? <span className="text-rose-300">{err}</span> : <LoaderCircle className="size-6 animate-spin" />}</Center>;
  }
  if (job.status === "done") return <Workspace job={job} />;
  const canRerun = engineMode() === "browser" && (job.status === "error" || job.status === "canceled");
  return (
    <main className="bg-aurora min-h-screen">
      <div className="mx-auto max-w-3xl px-4 py-8">
        <button onClick={onBack} className="btn btn-ghost mb-6">
          <ArrowLeft className="size-4" /> トップへ
        </button>
        <JobProgress job={job} onCancel={async () => setJob(await api.cancel(job.id).then(() => api.job(job.id)))} />
        {canRerun && (
          <div className="mt-4 flex justify-end">
            <button className="btn btn-primary" onClick={rerun} disabled={rerunning}>
              {rerunning ? <LoaderCircle className="size-4 animate-spin" /> : <RotateCcw className="size-4" />}
              同じ音声でもう一度実行
            </button>
          </div>
        )}
      </div>
    </main>
  );
}

export function Center({ children }: { children: React.ReactNode }) {
  return <main className="bg-aurora grid min-h-screen place-items-center text-ink-200">{children}</main>;
}

export function NoJob() {
  return (
    <Center>
      ジョブが指定されていません。<Link href="/" className="text-indigo-300 underline">トップへ</Link>
    </Center>
  );
}
