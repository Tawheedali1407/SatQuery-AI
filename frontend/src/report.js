import { fmtArea } from "./api.js";

const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);

// Opens a print-ready analysis report in a new window; the browser's "Save as PDF" produces the file.
export function exportReport(result, primary, comparison) {
  const w = window.open("", "_blank");
  if (!w) return alert("Allow pop-ups to export the report.");
  const now = new Date().toLocaleString();
  const scene = (s) =>
    s
      ? `<tr><td>${esc(s.name)}</td><td>${s.modality.toUpperCase()}</td><td>${s.width}×${s.height}</td><td>${
          s.gsd_m ? s.gsd_m.toFixed(2) + " m" : "—"
        }</td><td>${esc(s.crs || "—")}</td></tr>`
      : "";
  const regions = result.evidence.regions
    .slice(0, 10)
    .map(
      (r, i) =>
        `<tr><td>${i + 1}</td><td>${esc(r.label)}</td><td>${fmtArea(r.area)}</td><td>${r.centroid_norm
          .map((v) => v.toFixed(2))
          .join(", ")}</td><td>${r.confidence?.toFixed(2) ?? "—"}</td></tr>`
    )
    .join("");
  w.document.write(`<!doctype html><html><head><meta charset="utf-8"><title>SatQuery AI report</title>
<style>
body{font:13px/1.5 "IBM Plex Sans",system-ui,sans-serif;color:#111;margin:32px;max-width:900px}
h1{font-size:20px;margin:0}h2{font-size:14px;margin:24px 0 8px;text-transform:uppercase;letter-spacing:.06em;color:#334}
.sub{color:#556;margin-bottom:16px}.answer{font-size:16px;padding:12px 16px;border-left:4px solid #0e7490;background:#f1f8fa}
table{border-collapse:collapse;width:100%}td,th{border-bottom:1px solid #ddd;padding:4px 8px;text-align:left;font-size:12px}
.imgs{display:grid;grid-template-columns:repeat(2,1fr);gap:12px}.imgs figure{margin:0}.imgs img{width:100%;border:1px solid #ccc}
figcaption{font-size:11px;color:#556}.mono{font-family:"IBM Plex Mono",monospace;font-size:11px}.warn{color:#8a5a00}
@media print{body{margin:12mm}}
</style></head><body>
<h1>SatQuery AI — analysis report</h1>
<div class="sub">${esc(now)} · intent: ${esc(result.intent)} · confidence ${Math.round(result.confidence * 100)}% · ${result.total_ms} ms</div>
<h2>Query</h2><p>${esc(result.query)}</p>
<h2>Answer</h2><p class="answer">${esc(result.answer)}</p>
${result.warnings.map((x) => `<p class="warn">⚠ ${esc(x)}</p>`).join("")}
<p class="mono">method: ${esc(result.method)}</p>
<h2>Scenes</h2><table><tr><th>File</th><th>Modality</th><th>Pixels</th><th>GSD</th><th>CRS</th></tr>${scene(primary)}${scene(comparison)}</table>
<h2>Evidence</h2><div class="imgs">${result.evidence.images
    .map((im) => `<figure><img src="${im.src}"><figcaption>${esc(im.label)}</figcaption></figure>`)
    .join("")}</div>
${regions ? `<h2>Grounded regions</h2><table><tr><th>#</th><th>Label</th><th>Area</th><th>Centre</th><th>Conf.</th></tr>${regions}</table>` : ""}
<h2>Tool trace</h2><table>${result.trace
    .map((t) => `<tr><td class="mono">${esc(t.step)}</td><td class="mono">${esc(t.tool || "")}</td><td>${esc(t.detail)}</td><td class="mono">${t.ms} ms</td></tr>`)
    .join("")}</table>
<p class="mono" style="margin-top:24px;color:#667">Confidence values are heuristic (classification margins), not calibrated probabilities.</p>
<script>window.onload=()=>setTimeout(()=>window.print(),300)</script>
</body></html>`);
  w.document.close();
}
