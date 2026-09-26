import { useEffect, useRef, useState } from "react";

// CesiumJS is loaded from a CDN on first visit so the main bundle stays small.
const CESIUM_VER = "1.121.0";
const CESIUM_BASE = `https://cdn.jsdelivr.net/npm/cesium@${CESIUM_VER}/Build/Cesium/`;
let cesiumPromise;

function loadCesium() {
  if (window.Cesium) return Promise.resolve(window.Cesium);
  if (!cesiumPromise) {
    cesiumPromise = new Promise((resolve, reject) => {
      window.CESIUM_BASE_URL = CESIUM_BASE;
      const css = document.createElement("link");
      css.rel = "stylesheet";
      css.href = `${CESIUM_BASE}Widgets/widgets.css`;
      document.head.appendChild(css);
      const s = document.createElement("script");
      s.src = `${CESIUM_BASE}Cesium.js`;
      s.onload = () => resolve(window.Cesium);
      s.onerror = () => reject(new Error("Could not load CesiumJS (offline?)"));
      document.head.appendChild(s);
    });
  }
  return cesiumPromise;
}

export default function Globe({ scenes, onOpen }) {
  const el = useRef(null);
  const viewer = useRef(null);
  const [status, setStatus] = useState("Loading globe assets…");
  const geo = scenes.filter((s) => s.bounds_wgs84);
  const unref = scenes.filter((s) => !s.bounds_wgs84);

  useEffect(() => {
    let cancelled = false;
    loadCesium()
      .then((Cesium) => {
        if (cancelled || !el.current || viewer.current) return;
        Cesium.Ion.defaultAccessToken = ""; // no Ion token: open OSM imagery + ellipsoid terrain only
        viewer.current = new Cesium.Viewer(el.current, {
          baseLayer: new Cesium.ImageryLayer(new Cesium.OpenStreetMapImageryProvider({ url: "https://tile.openstreetmap.org/" })),
          baseLayerPicker: false,
          geocoder: false,
          timeline: false,
          animation: false,
          homeButton: true,
          sceneModePicker: true,
          navigationHelpButton: false,
          fullscreenButton: false,
          infoBox: false,
          selectionIndicator: false,
        });
        viewer.current.camera.setView({ destination: Cesium.Cartesian3.fromDegrees(80, 20, 9_000_000) }); // India
        const handler = new Cesium.ScreenSpaceEventHandler(viewer.current.scene.canvas);
        handler.setInputAction((click) => {
          const picked = viewer.current.scene.pick(click.position);
          const id = picked?.id?.properties?.sceneId?.getValue();
          if (id) onOpenRef.current(id);
        }, Cesium.ScreenSpaceEventType.LEFT_CLICK);
        setStatus("");
      })
      .catch((e) => setStatus(e.message));
    return () => {
      cancelled = true;
      viewer.current?.destroy();
      viewer.current = null;
    };
  }, []);

  const onOpenRef = useRef(onOpen);
  onOpenRef.current = onOpen;

  useEffect(() => {
    const v = viewer.current;
    const Cesium = window.Cesium;
    if (!v || !Cesium) return;
    v.entities.removeAll();
    geo.forEach((s) => {
      const [w, so, e, n] = s.bounds_wgs84;
      const color = s.modality === "sar" ? Cesium.Color.ORANGE : Cesium.Color.CYAN;
      v.entities.add({
        rectangle: {
          coordinates: Cesium.Rectangle.fromDegrees(w, so, e, n),
          material: color.withAlpha(0.25),
          outline: true,
          outlineColor: color,
          height: 0,
        },
        position: Cesium.Cartesian3.fromDegrees((w + e) / 2, (so + n) / 2),
        label: {
          text: s.name,
          font: "12px IBM Plex Mono, monospace",
          fillColor: Cesium.Color.WHITE,
          showBackground: true,
          backgroundColor: Cesium.Color.fromCssColorString("#0b1220").withAlpha(0.8),
          pixelOffset: new Cesium.Cartesian2(0, -14),
          disableDepthTestDistance: Number.POSITIVE_INFINITY,
        },
        properties: { sceneId: s.id },
      });
    });
    if (geo.length) v.flyTo(v.entities, { duration: 1.2 });
  }, [status, scenes]); // eslint-disable-line react-hooks/exhaustive-deps

  return (
    <section className="panel globe">
      <div className="panel-head">
        <h2>Global view</h2>
        <span className="tag">CesiumJS · open imagery, no token</span>
      </div>
      <div className="globe-wrap">
        <div ref={el} className="globe-canvas" />
        {status && <div className="globe-status">{status}</div>}
      </div>
      <p className="globe-note">
        Footprints mark registered scenes with an embedded CRS (cyan optical, orange SAR). Click a footprint to open it in the
        console.
        {unref.length > 0 && (
          <>
            {" "}
            Not georeferenced (PNG/JPG without CRS): {unref.map((s) => s.name).join(", ")}.
          </>
        )}
      </p>
    </section>
  );
}
