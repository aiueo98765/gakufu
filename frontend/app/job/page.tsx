"use client";

import { Suspense, useEffect, useState } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import Link from "next/link";
import { ArrowLeft, LoaderCircle } from "lucide-react";
import { api, engineMode } from "@/lib/api";
import type { Job } from "@/lib/types";
import { JobProgress } from "@/components/JobProgress";
import { Workspace } from "@/components/Workspace";
import { EngineGate } from "@/components/EngineGate";

function JobView() {
  const params = useSearchParams();
  const router = useRouter();
  const id = params.get("id") || "";
  const [job, setJob] = useState<Job | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [ver, setVer] = useState(0);

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

  if (!id) {
    return (
      <Center>
        ジョブが指定されていません。<Link href="/" className="text-indigo-300 underline">トップへ</Link>
      </Center>
    );
  }
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
  return (
    <main className="bg-aurora min-h-screen">
      <div className="mx-auto max-w-3xl px-4 py-8">
        <button onClick={() => router.push("/")} className="btn btn-ghost mb-6">
          <ArrowLeft className="size-4" /> トップへ
        </button>
        <JobProgress job={job} onCancel={async () => setJob(await api.cancel(job.id).then(() => api.job(job.id)))} />
      </div>
    </main>
  );
}

function Center({ children }: { children: React.ReactNode }) {
  return <main className="bg-aurora grid min-h-screen place-items-center text-ink-200">{children}</main>;
}

export default function JobPage() {
  return (
    <Suspense fallback={<Center><LoaderCircle className="size-6 animate-spin" /></Center>}>
      <JobView />
    </Suspense>
  );
}
