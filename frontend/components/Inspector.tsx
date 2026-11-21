"use client";

import { ArrowDown, ArrowUp, ChevronsDown, ChevronsUp, Hand, Trash2 } from "lucide-react";
import { formatPos, noteName, noteNameJa } from "@/lib/music";
import { STRING_NAMES, fretOn } from "@/lib/guitar";
import { STRING_COLORS } from "./PianoRoll";
import type { Note, Score } from "@/lib/types";

const DUR_LABEL: [number, string][] = [
  [4, "全音符"],
  [3, "付点2分"],
  [2, "2分"],
  [1.5, "付点4分"],
  [1, "4分"],
  [0.75, "付点8分"],
  [2 / 3, "3連4分"],
  [0.5, "8分"],
  [1 / 3, "3連8分"],
  [0.375, "付点16分"],
  [0.25, "16分"],
  [1 / 6, "6連16分"],
  [0.125, "32分"],
];

function durLabel(d: number) {
  const hit = DUR_LABEL.find(([v]) => Math.abs(v - d) < 1e-3);
  return hit ? hit[1] : `${+d.toFixed(3)}拍`;
}

export function Inspector({
  score,
  selection,
  onEdit,
  onDelete,
}: {
  score: Score;
  selection: Set<string>;
  onEdit: (fn: (notes: Note[]) => Note[], label: string) => void;
  onDelete: () => void;
}) {
  const sel = score.notes.filter((n) => selection.has(n.id));
  const k = score.key;
  const mapSel = (f: (n: Note) => Note, label: string) => onEdit((ns) => ns.map((n) => (selection.has(n.id) ? f(n) : n)), label);
  const clampP = (p: number) => Math.min(108, Math.max(21, p));

  if (!sel.length) {
    return (
      <div className="space-y-3 text-sm text-ink-300">
        <p>五線譜の小節をクリックするとそこから再生、ピアノロールの音符をクリックすると選択して直せます。</p>
        <div className="grid grid-cols-[auto_1fr] gap-x-3 gap-y-1.5 text-xs">
          <span className="kbd">Space</span> 再生 / 一時停止
          <span className="kbd">↑ ↓</span> 半音ずらす
          <span className="kbd">Shift+↑↓</span> オクターブ
          <span className="kbd">← →</span> 位置をずらす
          <span className="kbd">[ ]</span> 短く / 長く
          <span className="kbd">H</span> 右手 ⇄ 左手
          <span className="kbd">1〜5</span> 指番号（0 で自動）
          <span className="kbd">Del</span> 削除
          <span className="kbd">Ctrl+Z / Y</span> 元に戻す / やり直し
          <span className="kbd">Ctrl+A</span> 全選択
          <span className="kbd">L / M</span> ループ / クリック
        </div>
      </div>
    );
  }

  const one = sel.length === 1 ? sel[0] : null;
  const guitar = score.instrument === "guitar";
  const commonHand = sel.every((n) => n.hand === sel[0].hand) ? sel[0].hand : null;
  const commonFinger = sel.every((n) => n.finger === sel[0].finger) ? sel[0].finger : undefined;
  const avgVel = Math.round(sel.reduce((a, n) => a + n.vel, 0) / sel.length);

  return (
    <div className="space-y-4">
      <div>
        {one ? (
          <>
            <div className="flex items-baseline gap-2">
              <span className="text-2xl font-black">{noteName(one.pitch, k.fifths, k.minor)}</span>
              <span className="text-ink-300">{noteNameJa(one.pitch, k.fifths, k.minor)}</span>
            </div>
            <div className="text-xs text-ink-300">
              {formatPos(one.start, score)} ・ {durLabel(one.dur)}
              {guitar && one.string ? ` ・ ${STRING_NAMES[one.string - 1]} ${one.fret}フレット` : ""}
              {one.ornament === "trill" && " ・ トリル"}
              {one.arpeggiate && " ・ アルペジオ"}
            </div>
          </>
        ) : (
          <div className="text-lg font-bold">{sel.length} 音を選択中</div>
        )}
      </div>

      <div>
        <div className="label mb-1.5">音の高さ</div>
        <div className="grid grid-cols-4 gap-1.5">
          <button className="btn btn-ghost !px-1" onClick={() => mapSel((n) => ({ ...n, pitch: clampP(n.pitch - 12) }), "オクターブ下げ")} title="Shift+↓">
            <ChevronsDown className="size-4" /> 8度
          </button>
          <button className="btn btn-ghost !px-1" onClick={() => mapSel((n) => ({ ...n, pitch: clampP(n.pitch - 1) }), "半音下げ")} title="↓">
            <ArrowDown className="size-4" /> 半音
          </button>
          <button className="btn btn-ghost !px-1" onClick={() => mapSel((n) => ({ ...n, pitch: clampP(n.pitch + 1) }), "半音上げ")} title="↑">
            <ArrowUp className="size-4" /> 半音
          </button>
          <button className="btn btn-ghost !px-1" onClick={() => mapSel((n) => ({ ...n, pitch: clampP(n.pitch + 12) }), "オクターブ上げ")} title="Shift+↑">
            <ChevronsUp className="size-4" /> 8度
          </button>
        </div>
      </div>

      {guitar && (
        <div>
          <div className="label mb-1.5">弦（タブ譜の位置）</div>
          <div className="grid grid-cols-6 gap-1.5">
            {[1, 2, 3, 4, 5, 6].map((s) => {
              const ok = sel.every((n) => fretOn(score, n.pitch, s) !== null);
              const active = sel.every((n) => n.string === s);
              return (
                <button
                  key={s}
                  disabled={!ok}
                  title={one ? (ok ? `${fretOn(score, one.pitch, s)}フレット` : "この弦では出せない音") : STRING_NAMES[s - 1]}
                  className={`btn btn-ghost !px-0 ${active ? "btn-active" : ""}`}
                  onClick={() =>
                    mapSel((n) => ({ ...n, string: s, fret: fretOn(score, n.pitch, s) ?? n.fret, stringLocked: true }), `${STRING_NAMES[s - 1]}に`)
                  }
                >
                  <span className="size-2 rounded-full" style={{ background: STRING_COLORS[s - 1] }} />
                  {s}
                </button>
              );
            })}
          </div>
          {one && one.string && (
            <div className="mt-1 text-[11px] text-ink-400">
              {[1, 2, 3, 4, 5, 6]
                .map((s) => [s, fretOn(score, one.pitch, s)] as const)
                .filter(([, f]) => f !== null)
                .map(([s, f]) => `${s}弦${f}F`)
                .join(" / ")}
            </div>
          )}
          <button className="btn btn-ghost mt-1.5 w-full !py-1 text-xs" onClick={() => mapSel((n) => ({ ...n, stringLocked: false }), "弦を自動に")}>
            弦の固定を外す（再計算で自動）
          </button>
        </div>
      )}

      <div>
        <div className="label mb-1.5">{guitar ? "声部" : "手"}</div>
        <div className="grid grid-cols-2 gap-1.5">
          <button className={`btn btn-ghost ${commonHand === "R" ? "btn-active" : ""}`} onClick={() => mapSel((n) => ({ ...n, hand: "R" }), "右手に")}>
            <span className="size-2 rounded-full bg-rh" /> {guitar ? "メロディ" : "右手"}
          </button>
          <button className={`btn btn-ghost ${commonHand === "L" ? "btn-active" : ""}`} onClick={() => mapSel((n) => ({ ...n, hand: "L" }), "左手に")}>
            <span className="size-2 rounded-full bg-lh" /> {guitar ? "伴奏" : "左手"}
          </button>
        </div>
      </div>

      <div className={guitar ? "hidden" : ""}>
        <div className="label mb-1.5 flex items-center gap-1">
          <Hand className="size-3.5" /> 指番号
        </div>
        <div className="grid grid-cols-6 gap-1.5">
          {[1, 2, 3, 4, 5].map((f) => (
            <button
              key={f}
              className={`btn btn-ghost !px-0 ${commonFinger === f ? "btn-active" : ""}`}
              onClick={() => mapSel((n) => ({ ...n, finger: f, fingerLocked: true }), `指番号 ${f}`)}
            >
              {f}
            </button>
          ))}
          <button
            className="btn btn-ghost !px-0 text-xs"
            title="自動（次の再計算で AI が決める）"
            onClick={() => mapSel((n) => ({ ...n, fingerLocked: false }), "指番号を自動に")}
          >
            自動
          </button>
        </div>
        {sel.some((n) => n.fingerLocked) && <div className="mt-1 text-[11px] text-amber-200/80">黄色の数字は手で固定した指（再計算しても変わりません）</div>}
      </div>

      <div>
        <div className="label mb-1.5">強さ（ベロシティ） {avgVel}</div>
        <input
          type="range"
          className="range w-full"
          min={1}
          max={127}
          value={avgVel}
          style={{ ["--fill" as string]: `${(avgVel / 127) * 100}%` }}
          onChange={(e) => {
            const v = Number(e.target.value);
            mapSel((n) => ({ ...n, vel: v }), "強さを変更");
          }}
        />
      </div>

      <button className="btn btn-ghost w-full text-rose-300" onClick={onDelete}>
        <Trash2 className="size-4" /> 削除
      </button>
    </div>
  );
}
