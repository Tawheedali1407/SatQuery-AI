import { useEffect, useRef, useState } from "react";
import { api, fmtArea } from "./api.js";
import Evidence from "./Evidence.jsx";
import { exportReport } from "./report.js";

const EXAMPLES = [
  { label: "Land cover types", q: "What are the land cover types in this scene?" },
  { label: "Locate water bodies", q: "Locate the water bodies" },
  { label: "Detect change", q: "What changed between the two dates?" },
  { label: "Fuse optical + SAR", q: "Fuse optical and SAR to map flood water" },
];

function SceneSlot({ title, hint, scenes, value, onChange, onUpload, disabled, busy }) {
  const fileRef = useRef(null);
  const [modality, setModality] = useState("");
  const [gsd, setGsd] = useState("");
  const scene = scenes.find((s) => s.id === value);
  return (
    <div className="slot">
      <div className="slot-head">
        <span className="label">{title}</span>
        <span className="hint">{hint}</span>
      </div>
      <div className="slot-body">
        <div className="thumb">
          {scene ? <img src={api.previewUrl(scene.id)} alt={scene.name} /> : <span className="thumb-empty">no scene</span>}
        </div>
        <div className="slot-controls">
          <select value={value} onChange={(e) => onChange(e.target.value)} disabled={disabled}>
            <option value="">— select registered scene —</option>
            {scenes.map((s) => (
              <option key={s.id} value={s.id}>
                {s.name} · {s.modality.toUpperCase()}
              </option>
            ))}
          </select>
          <div className="upload-row">
            <select value={modality} onChange={(e) => setModality(e.target.value)} title="Sensor modality">
              <option value="">auto-detect</option>
              <option value="optical">optical</option>
              <option value="sar">SAR</option>
            </select>
            <input
              className="gsd"
              type="number"
              min="0"
              step="0.1"
              placeholder="GSD m"
              value={gsd}
              onChange={(e) => setGsd(e.target.value)}
              title="Ground sampling distance (m/pixel). GeoTIFFs carry this already; set it for PNG/JPG to get areas in km²."
            />
            <button className="btn ghost" disabled={disabled || busy} onClick={() => fileRef.current?.click()}>
              {busy ? "Uploading…" : "Upload"}
            </button>
            <input
              ref={fileRef}
              type="file"
              hidden
              accept=".tif,.tiff,.png,.jpg,.jpeg"
              onChange={(e) => {
                const f = e.target.files?.[0];
                if (f) onUpload(f, { modality, gsd });
                e.target.value = "";
              }}
            />
          </div>
          {scene && (
            <div className="meta mono">
              {scene.width}×{scene.height}px · {scene.gsd_m ? `${scene.gsd_m.toFixed(1)} m/px` : "GSD unknown"} ·{" "}
              {scene.crs || "no CRS"}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default function Console({ scenes, samples, primaryId, comparisonId, setPrimaryId, setComparisonId, addScenes, online }) {
  const [query, setQuery] = useState("");
  const [running, setRunning] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [uploading, setUploading] = useState("");
  const [sampleId, setSampleId] = useState("");
  const [loadingSample, setLoadingSample] = useState(false);

  const primary = scenes.find((s) => s.id === primaryId);
  const comparison = scenes.find((s) => s.id === comparisonId);
  const sample = samples.find((s) => s.id === sampleId);

  useEffect(() => {
    if (!sampleId && samples.length) setSampleId(samples.find((s) => s.available)?.id || "");
  }, [samples, sampleId]);

  const upload = async (slot, file, opts) => {
    setUploading(slot);
    setError(null);
    try {
      const s = await api.upload(file, opts);
      addScenes([s]);
      (slot === "primary" ? setPrimaryId : setComparisonId)(s.id);
    } catch (e) {
      setError(`Upload failed: ${e.message}`);
    } finally {
      setUploading("");
    }
  };

  const loadSample = async () => {
    if (!sample) return;
    setLoadingSample(true);
    setError(null);
    try {
      const list = await api.loadSample(sample.id);
      addScenes(list);
      const p = list.find((s) => s.role === "primary") || list[0];
      const c = list.find((s) => s.role === "comparison");
      setPrimaryId(p?.id || "");
      setComparisonId(c?.id || "");
      setResult(null);
      if (sample.suggested_queries?.length) setQuery(sample.suggested_queries[0]);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoadingSample(false);
    }
  };

  const run = async (q = query) => {
    if (!primaryId || !q.trim()) return;
    setRunning(true);
    setError(null);
    try {
      setResult(await api.query(q.trim(), primaryId, comparisonId));
    } catch (e) {
      setError(e.message);
      setResult(null);
    } finally {
      setRunning(false);
    }
  };

  const reset = () => {
    setQuery("");
    setResult(null);
    setError(null);
    setComparisonId("");
  };

  const norm = (s) => s.toLowerCase().replace(/^(the|a)\s+/, "").replace(/[?.]/g, "").replace(/\bthe\s+/g, "");
  const chips = [...(sample?.suggested_queries || []).map((q) => ({ label: q, q })), ...EXAMPLES].filter(
    (c, i, arr) => arr.findIndex((x) => x.q === c.q || norm(x.label) === norm(c.label) || norm(x.q) === norm(c.q)) === i
  );

  return (
    <div className="console">
      <section className="panel register">
        <div className="panel-head">
          <h2>Scene register &amp; query</h2>
          <span className="tag">local session only</span>
        </div>

        {samples.length > 0 && (
          <div className="sample-row">
            <select value={sampleId} onChange={(e) => setSampleId(e.target.value)}>
              {samples.map((s) => (
                <option key={s.id} value={s.id} disabled={!s.available}>
                  {s.title}
                  {s.available ? "" : " (not downloaded)"}
                </option>
              ))}
            </select>
            <button className="btn ghost" onClick={loadSample} disabled={!online || !sample?.available || loadingSample}>
              {loadingSample ? "Loading…" : "Load demo scene"}
            </button>
          </div>
        )}
        {sample && <p className="sample-desc">{sample.description} <span className="src">Source: {sample.source}</span></p>}

        <SceneSlot
          title="Primary scene"
          hint="required"
          scenes={scenes}
          value={primaryId}
          onChange={setPrimaryId}
          onUpload={(f, o) => upload("primary", f, o)}
          disabled={!online}
          busy={uploading === "primary"}
        />
        <SceneSlot
          title="Comparison scene"
          hint="optional · later date for change, or SAR/optical for fusion"
          scenes={scenes.filter((s) => s.id !== primaryId)}
          value={comparisonId}
          onChange={setComparisonId}
          onUpload={(f, o) => upload("comparison", f, o)}
          disabled={!online}
          busy={uploading === "comparison"}
        />

        <div className="query-box">
          <textarea
            value={query}
            maxLength={500}
            placeholder={primary ? "Ask about the scene — e.g. “How much of the area is flooded?”" : "Register a primary scene first"}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) run();
            }}
          />
          <div className="query-foot">
            <span className="mono counter">{query.length} / 500</span>
            <div className="actions">
              <button className="btn ghost" onClick={reset}>
                Reset
              </button>
              <button className="btn primary" onClick={() => run()} disabled={!primaryId || !query.trim() || running || !online}>
                {running ? "Analysing…" : "Run analysis"}
              </button>
            </div>
          </div>
        </div>

        <div className="examples">
          <span className="label">Try an example query</span>
          <div className="chips">
            {chips.map((c) => (
              <button
                key={c.q}
                className="chip-btn"
                disabled={!primaryId || running}
                onClick={() => {
                  setQuery(c.q);
                  run(c.q);
                }}
              >
                {c.label}
              </button>
            ))}
          </div>
        </div>
      </section>

      <section className="panel interpret">
        <div className="panel-head">
          <h2>Interpretation</h2>
          <span className="tag">{running ? "analysing…" : result ? `${result.intent} · ${result.total_ms} ms` : "awaiting query"}</span>
        </div>

        {error && <div className="banner error">{error}</div>}

        {!result && !error && !running && (
          <div className="empty">
            Register a primary scene, ask a question, and the routed tool trace will appear here.
            <div className="empty-sub">Every answer is computed from the pixels of your scene — nothing is pre-scripted.</div>
          </div>
        )}
        {running && <div className="scanning"><span />Routing query to specialist tools…</div>}

        {result && !running && (
          <div className="result">
            <div className="answer-card">
              <div className="answer-meta">
                <span className={`intent intent-${result.intent}`}>{result.intent.toUpperCase()}</span>
                <Confidence value={result.confidence} />
                {result.vqa_backend && <span className="tag">vqa/{result.vqa_backend}</span>}
              </div>
              <p className="answer">{result.answer}</p>
              <p className="method mono">method: {result.method}</p>
              {result.warnings?.map((w) => (
                <div key={w} className="banner warn">
                  {w}
                </div>
              ))}
            </div>

            <Evidence images={result.evidence.images} regions={result.evidence.regions} />

            {result.evidence.regions.length > 0 && (
              <div className="regions">
                <div className="label">Grounded regions</div>
                <table>
                  <thead>
                    <tr>
                      <th>#</th>
                      <th>Label</th>
                      <th>Area</th>
                      <th>Centre (x, y)</th>
                      <th>Conf.</th>
                    </tr>
                  </thead>
                  <tbody>
                    {result.evidence.regions.slice(0, 8).map((r, i) => (
                      <tr key={i}>
                        <td className="mono">{i + 1}</td>
                        <td>{r.label}</td>
                        <td className="mono">{fmtArea(r.area)}</td>
                        <td className="mono">
                          {r.centroid_norm[0].toFixed(2)}, {r.centroid_norm[1].toFixed(2)}
                        </td>
                        <td className="mono">{r.confidence !== undefined ? r.confidence.toFixed(2) : "—"}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}

            <div className="trace">
              <div className="label">Routed tool trace</div>
              <ol>
                {result.trace.map((t, i) => (
                  <li key={i}>
                    <span className="step mono">{t.step}</span>
                    {t.tool && <span className="tool mono">{t.tool}</span>}
                    <span className="detail">{t.detail}</span>
                    <span className="ms mono">{t.ms} ms</span>
                  </li>
                ))}
              </ol>
            </div>

            <div className="result-actions">
              <button className="btn ghost" onClick={() => exportReport(result, primary, comparison)}>
                Export report (PDF)
              </button>
            </div>
          </div>
        )}
      </section>
    </div>
  );
}

function Confidence({ value }) {
  const pct = Math.round((value || 0) * 100);
  const level = pct >= 70 ? "high" : pct >= 45 ? "med" : "low";
  return (
    <span className={`conf conf-${level}`} title="Heuristic confidence from classification margins — not a calibrated probability">
      <span className="conf-bar">
        <span style={{ width: `${pct}%` }} />
      </span>
      <span className="mono">{pct}% conf.</span>
    </span>
  );
}
