/** @type {import('next').NextConfig} */
// GitHub Pages のプロジェクトサイト（https://<user>.github.io/gakufu/）では NEXT_PUBLIC_BASE_PATH=/gakufu で組む
const basePath = process.env.NEXT_PUBLIC_BASE_PATH || "";
// ブラウザ版エンジン（public/engine/）の版。公開のたびに変わり、古い部品のキャッシュと混ざらないようにする
const buildId = (process.env.GITHUB_SHA || "").slice(0, 10) || `local-${Date.now().toString(36)}`;

const nextConfig = {
  env: { NEXT_PUBLIC_BUILD_ID: buildId },
  // 静的書き出し（npm run build → out/）。FastAPI でも GitHub Pages でもそのまま配れる。
  output: "export",
  basePath,
  assetPrefix: basePath || undefined,
  images: { unoptimized: true },
  trailingSlash: true,
  reactStrictMode: true,
};

export default nextConfig;
