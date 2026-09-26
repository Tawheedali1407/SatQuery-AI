import { api, fmtBytes } from "./api.js";

export default function Library({ scenes, onRemove, onOpen }) {
  return (
    <section className="panel library">
      <div className="panel-head">
        <h2>Scene library</h2>
        <span className="tag">this session</span>
      </div>
      {scenes.length === 0 ? (
        <div className="empty">No scenes registered yet — add one from the Analysis console.</div>
      ) : (
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th />
                <th>File</th>
                <th>Modality</th>
                <th>Size</th>
                <th>Pixels</th>
                <th>GSD</th>
                <th>CRS</th>
                <th>Role</th>
                <th />
              </tr>
            </thead>
            <tbody>
              {scenes.map((s) => (
                <tr key={s.id}>
                  <td>
                    <img className="lib-thumb" src={api.previewUrl(s.id)} alt="" />
                  </td>
                  <td>
                    <div>{s.name}</div>
                    {s.location && <div className="sub">{s.location}</div>}
                  </td>
                  <td>
                    <span className={`mod mod-${s.modality}`}>{s.modality.toUpperCase()}</span>
                  </td>
                  <td className="mono">{fmtBytes(s.size_bytes)}</td>
                  <td className="mono">
                    {s.original_width}×{s.original_height}
                    {s.original_width !== s.width && <div className="sub">analysed at {s.width}×{s.height}</div>}
                  </td>
                  <td className="mono">{s.gsd_m ? `${s.gsd_m.toFixed(1)} m` : "—"}</td>
                  <td className="mono">{s.crs || "—"}</td>
                  <td>{s.role || s.date || "—"}</td>
                  <td className="row-actions">
                    <button className="btn ghost sm" onClick={() => onOpen(s.id)}>
                      Open
                    </button>
                    <button className="btn ghost sm danger" onClick={() => onRemove(s.id)}>
                      Remove
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </section>
  );
}
