import type { Metadata, Viewport } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "楽譜AI — 動画URLからピアノ譜へ",
  description: "YouTube・X・Instagram の動画から音声を取り出し、AI でピアノ譜・ギターのタブ譜に採譜。簡単〜超プロ (Virtuoso) まで難易度別にアレンジし、再生・編集・運指・PDF/MIDI/TAB 書き出しまで。",
};

export const viewport: Viewport = {
  themeColor: "#07080d",
  width: "device-width",
  initialScale: 1,
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="ja">
      <body className="min-h-screen">{children}</body>
    </html>
  );
}
