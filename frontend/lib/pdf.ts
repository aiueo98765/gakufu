"use client";

/**
 * MusicXML → A4 縦の PDF。
 * 画面外で OpenSheetMusicDisplay に A4 ページ割りで浄書させ、各ページの SVG を svg2pdf で
 * ベクターのまま PDF に描く（拡大しても音符がにじまない）。
 * jsPDF の標準フォントは日本語を持たないので、日本語を含む文字（題名など）だけは
 * 高解像度の画像にして同じ位置に貼る。
 */
export async function exportPdf(
  xml: string,
  filename: string,
  onProgress?: (msg: string) => void,
): Promise<void> {
  const [{ OpenSheetMusicDisplay }, { jsPDF }, { svg2pdf }] = await Promise.all([
    import("opensheetmusicdisplay"),
    import("jspdf"),
    import("svg2pdf.js"),
  ]);
  onProgress?.("ページを組んでいます…");
  const host = document.createElement("div");
  host.style.cssText = "position:fixed;left:-20000px;top:0;width:1200px;background:#fff;color:#000;";
  document.body.appendChild(host);
  try {
    const osmd = new OpenSheetMusicDisplay(host, {
      backend: "svg",
      autoResize: false,
      pageFormat: "A4_P",
      pageBackgroundColor: "#FFFFFF",
      drawTitle: true,
      drawSubtitle: true,
      drawComposer: true,
      drawCredits: true,
      drawPartNames: false,
      drawMeasureNumbers: true,
      drawFingerings: true,
      fingeringPosition: "auto",
      defaultColorMusic: "#000000",
    });
    osmd.EngravingRules.PageLeftMargin = 6;
    osmd.EngravingRules.PageRightMargin = 6;
    osmd.EngravingRules.PageTopMargin = 6;
    osmd.EngravingRules.PageBottomMargin = 6;
    const { fitTitle } = await import("@/components/ScoreView");
    fitTitle(osmd, xml);
    await osmd.load(xml);
    osmd.zoom = 1.0;
    osmd.render();

    const svgs = Array.from(host.querySelectorAll("svg"));
    if (!svgs.length) throw new Error("楽譜を描画できませんでした");
    const pdf = new jsPDF({ orientation: "portrait", unit: "pt", format: "a4", compress: true });
    const pageW = pdf.internal.pageSize.getWidth();
    const pageH = pdf.internal.pageSize.getHeight();

    for (let i = 0; i < svgs.length; i++) {
      onProgress?.(`PDF を書き出しています… ${i + 1}/${svgs.length} ページ`);
      const svg = svgs[i] as SVGSVGElement;
      if (i > 0) pdf.addPage();
      const vb = svg.viewBox?.baseVal;
      const svgW = vb && vb.width ? vb.width : svg.width.baseVal.value;
      const svgH = vb && vb.height ? vb.height : svg.height.baseVal.value;
      const scale = Math.min(pageW / svgW, pageH / svgH);
      const offX = (pageW - svgW * scale) / 2;
      const offY = 0;

      // 日本語などの文字を画像で描き、SVG からは取り除く
      const texts = Array.from(svg.querySelectorAll("text")).filter((t) => /[^\u0000-ÿ]/.test(t.textContent || ""));
      const overlays: { data: string; x: number; y: number; w: number; h: number }[] = [];
      for (const t of texts) {
        const img = textToImage(t);
        if (img) {
          overlays.push({
            data: img.data,
            x: offX + img.x * scale,
            y: offY + img.y * scale,
            w: img.w * scale,
            h: img.h * scale,
          });
        }
        t.remove();
      }
      await svg2pdf(svg, pdf, { x: offX, y: offY, width: svgW * scale, height: svgH * scale });
      for (const o of overlays) pdf.addImage(o.data, "PNG", o.x, o.y, o.w, o.h, undefined, "FAST");
      pdf.setFontSize(8);
      pdf.setTextColor(120);
      pdf.text(`${i + 1} / ${svgs.length}`, pageW / 2, pageH - 14, { align: "center" });
    }
    onProgress?.("保存しています…");
    pdf.save(filename);
  } finally {
    host.remove();
  }
}

/** SVG の <text> を同じ見た目の PNG にする（座標は SVG のユーザー単位）。 */
function textToImage(t: SVGTextElement): { data: string; x: number; y: number; w: number; h: number } | null {
  let box: DOMRect;
  try {
    box = t.getBBox();
  } catch {
    return null;
  }
  if (!box.width || !box.height) return null;
  const cs = window.getComputedStyle(t);
  const fontSize = parseFloat(t.getAttribute("font-size") || cs.fontSize || "16");
  const family = t.getAttribute("font-family") || cs.fontFamily || "serif";
  const weight = t.getAttribute("font-weight") || cs.fontWeight || "normal";
  const style = t.getAttribute("font-style") || cs.fontStyle || "normal";
  const fill = t.getAttribute("fill") || "#000";
  const k = 4; // 4倍で描いておけば印刷でも粗くならない
  const pad = fontSize * 0.25;
  const canvas = document.createElement("canvas");
  canvas.width = Math.ceil((box.width + pad * 2) * k);
  canvas.height = Math.ceil((box.height + pad * 2) * k);
  const ctx = canvas.getContext("2d");
  if (!ctx) return null;
  ctx.scale(k, k);
  ctx.fillStyle = fill;
  ctx.textBaseline = "alphabetic";
  ctx.font = `${style} ${weight} ${fontSize}px ${family}, "Hiragino Mincho ProN", "Yu Mincho", "Noto Serif JP", serif`;
  // getBBox の y は文字の上端。ベースラインは x/y 属性から取る
  const baseY = parseFloat(t.getAttribute("y") || "0");
  const startX = box.x;
  ctx.fillText(t.textContent || "", pad, baseY - box.y + pad);
  return { data: canvas.toDataURL("image/png"), x: startX - pad, y: box.y - pad, w: box.width + pad * 2, h: box.height + pad * 2 };
}
