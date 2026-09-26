import { useEffect, useState } from "react";

const LEGEND = [
  ["Water", "#1e78ff"],
  ["Vegetation", "#28be50"],
  ["Built-up", "#f04646"],
  ["Bare soil", "#dcaa5a"],
  ["Cloud", "#ebebf5"],
];

// Main evidence viewer: one large image with optional bounding boxes, and a thumbnail strip.
export default function Evidence({ images, regions }) {
  const [idx, setIdx] = useState(images.length - 1);
  const [showBoxes, setShowBoxes] = useState(true);
  useEffect(() => setIdx(images.length - 1), [images]);
  const img = images[Math.min(idx, images.length - 1)];
  if (!img) return null;
  return (
    <div className="evidence">
      <div className="evidence-head">
        <span className="label">Evidence · {img.label}</span>
        {img.boxes && regions.length > 0 && (
          <label className="toggle">
            <input type="checkbox" checked={showBoxes} onChange={(e) => setShowBoxes(e.target.checked)} /> boxes
          </label>
        )}
      </div>
      <div className="evidence-main">
        <div className="evidence-frame">
          <img src={img.src} alt={img.label} />
          {img.boxes && showBoxes && (
            <svg className="boxes" viewBox="0 0 1 1" preserveAspectRatio="none">
              {regions.slice(0, 12).map((r, i) => {
                const [x0, y0, x1, y1] = r.bbox_norm;
                return (
                  <g key={i}>
                    <rect x={x0} y={y0} width={x1 - x0} height={y1 - y0} vectorEffect="non-scaling-stroke" />
                  </g>
                );
              })}
            </svg>
          )}
          {img.boxes && showBoxes && (
            <div className="box-labels">
              {regions.slice(0, 12).map((r, i) => (
                <span key={i} style={{ left: `${r.bbox_norm[0] * 100}%`, top: `${r.bbox_norm[1] * 100}%` }}>
                  {i + 1}
                </span>
              ))}
            </div>
          )}
        </div>
        {img.legend === "landcover" && (
          <div className="legend">
            {LEGEND.map(([n, c]) => (
              <span key={n}>
                <i style={{ background: c }} />
                {n}
              </span>
            ))}
          </div>
        )}
      </div>
      {images.length > 1 && (
        <div className="thumbs">
          {images.map((im, i) => (
            <button key={i} className={i === idx ? "active" : ""} onClick={() => setIdx(i)} title={im.label}>
              <img src={im.src} alt={im.label} />
              <span>{im.label}</span>
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
