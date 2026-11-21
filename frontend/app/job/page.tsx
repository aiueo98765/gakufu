"use client";

import { Suspense } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import { LoaderCircle } from "lucide-react";
import { Center, JobView, NoJob } from "@/components/JobView";

function JobPageInner() {
  const params = useSearchParams();
  const router = useRouter();
  const id = params.get("id") || "";
  if (!id) return <NoJob />;
  return (
    <JobView
      id={id}
      onBack={() => router.push("/")}
      // ページは読み込み直さず URL だけ替える（ブラウザ版はこのタブの中で処理しているため）
      onRerun={(nid) => window.history.replaceState(null, "", `?id=${nid}`)}
    />
  );
}

export default function JobPage() {
  return (
    <Suspense
      fallback={
        <Center>
          <LoaderCircle className="size-6 animate-spin" />
        </Center>
      }
    >
      <JobPageInner />
    </Suspense>
  );
}
