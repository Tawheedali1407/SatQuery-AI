// Thin client for the FastAPI backend.
const BASE = (import.meta.env.VITE_API_URL || "").replace(/\/$/, "");

async function req(path, opts = {}) {
  const r = await fetch(`${BASE}${path}`, opts);
  if (!r.ok) {
    let msg = `${r.status} ${r.statusText}`;
    try {
      const body = await r.json();
      msg = body.detail || msg;
    } catch {
      /* non-JSON error */
    }
    throw new Error(msg);
  }
  return r.json();
}

export const api = {
  health: () => req("/api/health"),
  scenes: () => req("/api/scenes"),
  samples: () => req("/api/samples"),
  loadSample: (id) => req(`/api/samples/${id}/load`, { method: "POST" }),
  upload: (file, { modality, gsd } = {}) => {
    const fd = new FormData();
    fd.append("file", file);
    if (modality) fd.append("modality", modality);
    if (gsd) fd.append("gsd_m", gsd);
    return req("/api/scenes", { method: "POST", body: fd });
  },
  remove: (id) => req(`/api/scenes/${id}`, { method: "DELETE" }),
  query: (query, primary_id, comparison_id) =>
    req("/api/query", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query, primary_id, comparison_id: comparison_id || null }),
    }),
  previewUrl: (id) => `${BASE}/api/scenes/${id}/preview`,
};

export function fmtBytes(n) {
  if (!n) return "—";
  const u = ["B", "KB", "MB", "GB"];
  let i = 0;
  while (n >= 1024 && i < u.length - 1) {
    n /= 1024;
    i++;
  }
  return `${n.toFixed(i ? 1 : 0)} ${u[i]}`;
}

export function fmtArea(a) {
  if (!a) return "—";
  if (a.km2 !== undefined) return a.km2 >= 0.01 ? `${a.km2.toFixed(3)} km²` : `${Math.round(a.m2)} m²`;
  return `${a.pixels.toLocaleString()} px`;
}
