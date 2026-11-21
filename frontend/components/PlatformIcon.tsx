import { FileAudio } from "lucide-react";

/** サービスのロゴ（簡略化した図形。各社の商標ロゴの複製ではない） */
export function PlatformIcon({ platform, className = "size-5" }: { platform?: string; className?: string }) {
  if (platform === "youtube") {
    return (
      <svg viewBox="0 0 24 24" className={className} aria-label="YouTube">
        <rect x="1.5" y="5" width="21" height="14" rx="4" fill="#ff3d3d" />
        <path d="M10 9v6l5-3z" fill="#fff" />
      </svg>
    );
  }
  if (platform === "x") {
    return (
      <svg viewBox="0 0 24 24" className={className} aria-label="X">
        <rect x="1.5" y="1.5" width="21" height="21" rx="5" fill="#0b0b0b" stroke="#3a3a3a" />
        <path d="M7 6.5h2.6l7.4 11H14.4zM16.6 6.5 7.4 17.5" stroke="#fff" strokeWidth="1.6" fill="none" strokeLinecap="round" />
      </svg>
    );
  }
  if (platform === "instagram") {
    return (
      <svg viewBox="0 0 24 24" className={className} aria-label="Instagram">
        <defs>
          <linearGradient id="ig-g" x1="0" y1="1" x2="1" y2="0">
            <stop offset="0" stopColor="#feda75" />
            <stop offset="0.35" stopColor="#fa7e1e" />
            <stop offset="0.6" stopColor="#d62976" />
            <stop offset="1" stopColor="#4f5bd5" />
          </linearGradient>
        </defs>
        <rect x="2" y="2" width="20" height="20" rx="6" fill="url(#ig-g)" />
        <circle cx="12" cy="12" r="4.2" stroke="#fff" strokeWidth="1.8" fill="none" />
        <circle cx="17.3" cy="6.8" r="1.2" fill="#fff" />
      </svg>
    );
  }
  return <FileAudio className={className} />;
}

export function detectPlatform(url: string): "youtube" | "x" | "instagram" | null {
  let u = url.trim();
  if (!u) return null;
  if (!/^https?:\/\//i.test(u)) u = "https://" + u;
  try {
    const h = new URL(u).hostname.toLowerCase().replace(/\.$/, "");
    const is = (root: string) => h === root || h.endsWith("." + root);
    if (is("youtube.com") || is("youtu.be") || is("youtube-nocookie.com")) return "youtube";
    if (is("x.com") || is("twitter.com")) return "x";
    if (is("instagram.com")) return "instagram";
  } catch {
    return null;
  }
  return null;
}

export const PLATFORM_LABEL: Record<string, string> = {
  youtube: "YouTube",
  x: "X (Twitter)",
  instagram: "Instagram",
  upload: "ファイル",
};
