import { useCallback, useEffect, useState } from "react";
import { api } from "./api.js";
import Console from "./Console.jsx";
import Library from "./Library.jsx";
import Globe from "./Globe.jsx";

const VIEWS = [
  { id: "console", label: "Analysis console", short: "Console", icon: "◎" },
  { id: "library", label: "Scene library", short: "Library", icon: "▤" },
  { id: "globe", label: "Global view", short: "Globe", icon: "◍" },
];

export default function App() {
  const [view, setView] = useState("console");
  const [health, setHealth] = useState(null);
  const [healthErr, setHealthErr] = useState(null);
  const [scenes, setScenes] = useState([]);
  const [samples, setSamples] = useState([]);
  const [primaryId, setPrimaryId] = useState("");
  const [comparisonId, setComparisonId] = useState("");

  const refresh = useCallback(async () => {
    try {
      const [h, s] = await Promise.all([api.health(), api.scenes()]);
      setHealth(h);
      setScenes(s);
      setHealthErr(null);
    } catch (e) {
      setHealthErr(e.message);
    }
  }, []);

  useEffect(() => {
    refresh();
    api.samples().then(setSamples).catch(() => setSamples([]));
    const t = setInterval(refresh, 15000);
    return () => clearInterval(t);
  }, [refresh]);

  const addScenes = (list) => {
    setScenes((prev) => {
      const ids = new Set(prev.map((s) => s.id));
      return [...prev, ...list.filter((s) => !ids.has(s.id))];
    });
  };

  const removeScene = async (id) => {
    await api.remove(id);
    setScenes((prev) => prev.filter((s) => s.id !== id));
    if (primaryId === id) setPrimaryId("");
    if (comparisonId === id) setComparisonId("");
  };

  const openInConsole = (id) => {
    setPrimaryId(id);
    setView("console");
  };

  const online = health && !healthErr;
  const regionLabel = (() => {
    const p = scenes.find((s) => s.id === primaryId);
    if (!p) return "NO SCENE";
    if (p.bounds_wgs84) {
      const [w, s, e, n] = p.bounds_wgs84;
      return `${((s + n) / 2).toFixed(2)}°, ${((w + e) / 2).toFixed(2)}°`;
    }
    return p.location ? p.location.toUpperCase().slice(0, 28) : "UNREFERENCED";
  })();

  return (
    <div className="shell">
      <aside className="rail">
        <div className="brand">
          <span className="brand-mark" aria-hidden>
            <svg viewBox="0 0 32 32" width="26" height="26">
              <circle cx="16" cy="16" r="5" fill="currentColor" />
              <ellipse cx="16" cy="16" rx="13" ry="5" fill="none" stroke="currentColor" strokeWidth="1.6" transform="rotate(-30 16 16)" />
            </svg>
          </span>
          <div>
            <div className="brand-name">SATQUERY AI</div>
            <div className="brand-sub">Ground station console for optical / SAR scene analysis</div>
          </div>
        </div>
        <nav className="nav">
          {VIEWS.map((v) => (
            <button key={v.id} className={`nav-item ${view === v.id ? "active" : ""}`} onClick={() => setView(v.id)}>
              <span className="nav-icon">{v.icon}</span>
              {v.label}
            </button>
          ))}
        </nav>
        <div className="rail-foot">
          <div className={`status ${online ? "ok" : "down"}`}>
            <span className="dot" />
            {online ? "BACKEND LIVE" : "BACKEND OFFLINE"}
          </div>
          {online && (
            <div className="rail-meta">
              v{health.version} · change: {health.change_backend}
              <br />
              VQA: {health.vqa_backend}
              <br />
              GeoTIFF: {health.geotiff_support ? "enabled" : "PNG/JPG only"}
            </div>
          )}
        </div>
      </aside>

      <main className="main">
        <header className="topbar">
          <div className="topbar-left">
            <span className="chip">REGION {regionLabel}</span>
            <span className="chip muted">SESSION LOCAL · NO EGRESS</span>
          </div>
          <div className="topbar-right mono">{scenes.length} scene{scenes.length === 1 ? "" : "s"} registered</div>
        </header>

        {healthErr && (
          <div className="banner error">
            Can't reach the analysis backend ({healthErr}). Start it with <code>uvicorn app.main:app --reload</code> in{" "}
            <code>backend/</code>, or set <code>VITE_API_URL</code>.
          </div>
        )}

        <div className="view">
          {view === "console" && (
            <Console
              scenes={scenes}
              samples={samples}
              primaryId={primaryId}
              comparisonId={comparisonId}
              setPrimaryId={setPrimaryId}
              setComparisonId={setComparisonId}
              addScenes={addScenes}
              online={online}
            />
          )}
          {view === "library" && <Library scenes={scenes} onRemove={removeScene} onOpen={openInConsole} />}
          {view === "globe" && <Globe scenes={scenes} onOpen={openInConsole} />}
        </div>
      </main>

      <nav className="tabbar">
        {VIEWS.map((v) => (
          <button key={v.id} className={view === v.id ? "active" : ""} onClick={() => setView(v.id)}>
            <span>{v.icon}</span>
            {v.short}
          </button>
        ))}
      </nav>
    </div>
  );
}
