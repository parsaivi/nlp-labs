import { useState, useMemo, useCallback, useRef, useEffect } from "react";

// Shared by every lesson (the build prepends this file to each one).
// Theme: Ink notebook. Dark ruled page, violet ink (accent) and apricot marks (highlight).
// Every colour-coded meaning also carries a second cue: a label, a marker shape or a dash style.

const THEME = {
  bg: "#15121f",
  surface: "#1e1a2e",
  text: "#f0ebfa",
  muted: "#b4abc9",
  accent: "#a594f9",
  highlight: "#ffb26b",
  third: "#6fd3c7",
  grid: "#3d3560",
  warn: "#ff7d93",
  ok: "#7fe0a6",
};
const MONO = "'IBM Plex Mono', ui-monospace, SFMono-Regular, Menlo, Consolas, monospace";
const SANS = "'IBM Plex Sans', system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif";
const DISPLAY = "'IBM Plex Serif', Georgia, 'Times New Roman', serif";
const FA = "'Vazirmatn', 'Noto Naskh Arabic', Tahoma, sans-serif";
const SCREENS = ["Puzzle", "Explore", "Name", "Challenge"];
const REDUCED =
  typeof window !== "undefined" && !!window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;

// @math-start
function clamp(v, a, b) {
  return v < a ? a : v > b ? b : v;
}
function fmt(v, d) {
  if (d === undefined) d = 2;
  if (v === Infinity) return "∞";
  if (v === -Infinity) return "−∞";
  if (!isFinite(v)) return "NaN";
  if (Math.abs(v) >= 1e5) return v.toExponential(1).replace("-", "−");
  const s = v.toFixed(d);
  if (/^-0\.?0*$/.test(s)) return s.slice(1);
  return s.replace("-", "−");
}
function sgn(v, d) {
  return (v >= 0 ? "+" : "−") + fmt(Math.abs(v), d);
}
function dot(a, b) {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}
function norm(a) {
  return Math.sqrt(dot(a, a));
}
function cosine(a, b) {
  const n = norm(a) * norm(b);
  return n > 1e-12 ? dot(a, b) / n : 0;
}
function vadd(a, b) {
  return a.map((v, i) => v + b[i]);
}
function vsub(a, b) {
  return a.map((v, i) => v - b[i]);
}
function vscale(a, k) {
  return a.map((v) => v * k);
}
function vsum(list, dim) {
  const out = new Array(dim).fill(0);
  for (const v of list) for (let i = 0; i < dim; i++) out[i] += v[i];
  return out;
}
function softmax(arr) {
  let m = -Infinity;
  for (const v of arr) if (v > m) m = v;
  if (m === -Infinity) return arr.map(() => NaN);
  const e = arr.map((v) => (v === -Infinity ? 0 : Math.exp(v - m)));
  const s = e.reduce((a, b) => a + b, 0);
  return e.map((v) => v / s);
}
function sigmoid(x) {
  return 1 / (1 + Math.exp(-x));
}
function argmax(arr) {
  let k = 0;
  for (let i = 1; i < arr.length; i++) if (arr[i] > arr[k]) k = i;
  return k;
}
function mean(arr) {
  return arr.reduce((a, b) => a + b, 0) / arr.length;
}
function stdev(arr) {
  const m = mean(arr);
  return Math.sqrt(mean(arr.map((v) => (v - m) * (v - m))));
}
// seeded random numbers so every learner sees the same "random" draw
function mulberry32(seed) {
  let a = seed >>> 0;
  return function () {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
function randn(rng) {
  const u = Math.max(rng(), 1e-12);
  const v = rng();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}
function hashStr(s) {
  let h = 2166136261;
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return h >>> 0;
}
// rows-of-matrix times vector
function matVec(M, v) {
  return M.map((r) => dot(r, v));
}
function rot2(deg) {
  const t = (deg * Math.PI) / 180;
  return [
    [Math.cos(t), -Math.sin(t)],
    [Math.sin(t), Math.cos(t)],
  ];
}
// @math-end

function rgba(hex, a) {
  const n = parseInt(hex.slice(1), 16);
  return "rgba(" + ((n >> 16) & 255) + "," + ((n >> 8) & 255) + "," + (n & 255) + "," + a + ")";
}
function svgPoint(svg, e) {
  const pt = svg.createSVGPoint();
  pt.x = e.clientX;
  pt.y = e.clientY;
  return pt.matrixTransform(svg.getScreenCTM().inverse());
}

// ---------- small UI pieces ----------
function Fa({ children, color }) {
  return (
    <bdi lang="fa" style={{ fontFamily: FA, fontSize: "1.04em", color: color || "inherit" }}>
      {children}
    </bdi>
  );
}
function Panel({ children, edge, style }) {
  return (
    <div style={Object.assign({ background: THEME.surface, border: "1px solid " + (edge || THEME.grid), borderRadius: 6, padding: "10px 12px" }, style || {})}>
      {children}
    </div>
  );
}
function Note({ children }) {
  return <div style={{ color: THEME.muted, fontSize: 14, lineHeight: 1.5, margin: "0 0 8px" }}>{children}</div>;
}
function Label({ children }) {
  return (
    <div style={{ fontFamily: MONO, fontSize: 11, letterSpacing: 1, textTransform: "uppercase", color: THEME.muted, margin: "2px 0 6px" }}>{children}</div>
  );
}
function Pill({ value, active, onPick, children, title }) {
  const click = useCallback(() => onPick(value), [onPick, value]);
  return (
    <button
      onClick={click}
      title={title}
      aria-pressed={!!active}
      style={{
        fontFamily: MONO,
        fontSize: 12.5,
        padding: "5px 10px",
        borderRadius: 14,
        cursor: "pointer",
        border: "1px solid " + (active ? THEME.highlight : THEME.grid),
        background: active ? rgba(THEME.highlight, 0.16) : THEME.bg,
        color: THEME.text,
      }}
    >
      {children}
    </button>
  );
}
function Pills({ options, value, onPick }) {
  return (
    <div style={{ display: "flex", gap: 6, flexWrap: "wrap", margin: "2px 0 8px" }}>
      {options.map((o) => (
        <Pill key={o[0]} value={o[0]} active={value === o[0]} onPick={onPick}>
          {o[1]}
        </Pill>
      ))}
    </div>
  );
}
function Btn({ onClick, children, disabled, strong }) {
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      style={{
        fontFamily: MONO,
        fontSize: 12.5,
        padding: "6px 11px",
        borderRadius: 4,
        cursor: disabled ? "default" : "pointer",
        border: "1px solid " + (disabled ? THEME.grid : THEME.accent),
        background: strong && !disabled ? rgba(THEME.accent, 0.18) : "transparent",
        color: disabled ? THEME.muted : THEME.accent,
      }}
    >
      {children}
    </button>
  );
}
function Slider({ id, label, value, min, max, stepSize, onChange, display, width }) {
  const change = useCallback((e) => onChange(parseFloat(e.target.value)), [onChange]);
  return (
    <label htmlFor={id} style={{ display: "block", margin: "4px 0 8px", minWidth: 0, width: width || "auto" }}>
      <div style={{ display: "flex", justifyContent: "space-between", gap: 8, fontSize: 13, color: THEME.muted }}>
        <span>{label}</span>
        <span style={{ fontFamily: MONO, color: THEME.text, fontVariantNumeric: "tabular-nums" }}>{display !== undefined ? display : fmt(value)}</span>
      </div>
      <input id={id} type="range" min={min} max={max} step={stepSize} value={value} onChange={change} style={{ width: "100%" }} />
    </label>
  );
}
function Toggle({ label, value, onChange }) {
  const flip = useCallback(() => onChange(!value), [onChange, value]);
  return (
    <button
      onClick={flip}
      aria-pressed={value}
      style={{
        fontFamily: MONO,
        fontSize: 12.5,
        padding: "5px 10px",
        borderRadius: 4,
        cursor: "pointer",
        border: "1px solid " + (value ? THEME.highlight : THEME.grid),
        background: value ? rgba(THEME.highlight, 0.16) : THEME.bg,
        color: THEME.text,
        margin: "2px 6px 6px 0",
      }}
    >
      {value ? "■ " : "□ "}
      {label}
    </button>
  );
}
// a clickable word token; `on` marks it as chosen
function Chip({ value, on, onPick, children, mark, disabled }) {
  const click = useCallback(() => onPick(value), [onPick, value]);
  return (
    <button
      onClick={click}
      disabled={disabled}
      aria-pressed={!!on}
      style={{
        fontFamily: FA,
        fontSize: 15,
        padding: "3px 10px 4px",
        borderRadius: 4,
        cursor: disabled ? "default" : "pointer",
        border: "1px " + (on ? "solid " + THEME.highlight : "dashed " + THEME.grid),
        background: on ? rgba(THEME.highlight, 0.16) : THEME.bg,
        color: disabled ? THEME.muted : THEME.text,
        lineHeight: 1.5,
      }}
    >
      {mark ? <span style={{ fontFamily: MONO, fontSize: 11, color: THEME.muted, marginLeft: 4 }}>{mark} </span> : null}
      <span lang="fa">{children}</span>
    </button>
  );
}
function Hint({ hints }) {
  const [n, setN] = useState(0);
  const more = useCallback(() => setN((v) => Math.min(v + 1, hints.length)), [hints.length]);
  return (
    <div style={{ marginTop: 8 }}>
      {hints.slice(0, n).map((h, i) => (
        <div key={i} style={{ color: THEME.text, fontSize: 13.5, margin: "4px 0 6px", paddingLeft: 10, borderLeft: "2px dashed " + THEME.accent, lineHeight: 1.45 }}>
          <span style={{ fontFamily: MONO, color: THEME.accent }}>hint {i + 1} </span>
          {h}
        </div>
      ))}
      {n < hints.length && (
        <button
          onClick={more}
          style={{ fontFamily: MONO, fontSize: 12.5, background: "transparent", color: THEME.accent, border: "1px dashed " + THEME.accent, borderRadius: 4, padding: "4px 10px", cursor: "pointer" }}
        >
          {n === 0 ? "Stuck?" : "Another hint"}
        </button>
      )}
    </div>
  );
}
function Eq({ children, size }) {
  return (
    <div
      style={{
        fontFamily: MONO,
        fontSize: size || 14,
        color: THEME.text,
        background: THEME.bg,
        border: "1px solid " + THEME.grid,
        borderLeft: "3px solid " + THEME.accent,
        borderRadius: 4,
        padding: "8px 10px",
        margin: "6px 0",
        lineHeight: 1.7,
        overflowX: "auto",
        whiteSpace: "nowrap",
      }}
    >
      {children}
    </div>
  );
}
function Readout({ rows }) {
  return (
    <div style={{ display: "grid", gridTemplateColumns: "auto auto", justifyContent: "space-between", gap: "2px 12px", fontFamily: MONO, fontSize: 13, lineHeight: 1.6 }}>
      {rows.map((r, i) => (
        <React.Fragment key={i}>
          <span style={{ color: THEME.muted }}>{r[0]}</span>
          <span style={{ color: r[2] || THEME.text, textAlign: "right", fontVariantNumeric: "tabular-nums" }}>{r[1]}</span>
        </React.Fragment>
      ))}
    </div>
  );
}
// a goal line: ◆ when met, ◇ when not, so the state never depends on colour alone
function Goal({ met, children }) {
  return (
    <div style={{ fontFamily: MONO, fontSize: 13, color: met ? THEME.ok : THEME.muted, margin: "3px 0", lineHeight: 1.5 }}>
      {met ? "◆ " : "◇ "}
      {children}
    </div>
  );
}

// ---------- lesson frame: the four screens ----------
function ScreenTab({ i, active, done, onPick }) {
  const click = useCallback(() => onPick(i), [onPick, i]);
  return (
    <button
      onClick={click}
      aria-current={active ? "step" : undefined}
      style={{
        fontFamily: MONO,
        fontSize: 12.5,
        padding: "5px 10px",
        cursor: "pointer",
        borderRadius: 4,
        border: "1px solid " + (active ? THEME.accent : THEME.grid),
        background: active ? rgba(THEME.accent, 0.15) : "transparent",
        color: active || done ? THEME.text : THEME.muted,
      }}
    >
      {i + 1} {SCREENS[i]}
    </button>
  );
}
function LessonFrame({ screen, onScreen, title, question, stage, side, below }) {
  const back = useCallback(() => onScreen(Math.max(0, screen - 1)), [onScreen, screen]);
  const next = useCallback(() => onScreen(Math.min(3, screen + 1)), [onScreen, screen]);
  return (
    <div style={{ fontFamily: SANS, color: THEME.text, background: rgba(THEME.surface, 0.6), border: "1px solid " + THEME.grid, borderRadius: 6, padding: "12px 14px 14px" }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", gap: 10, flexWrap: "wrap", marginBottom: 8 }}>
        <div style={{ fontFamily: DISPLAY, fontSize: 23, fontWeight: 600, letterSpacing: 0.1 }}>{title}</div>
        <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
          {SCREENS.map((s, i) => (
            <ScreenTab key={s} i={i} active={i === screen} done={i < screen} onPick={onScreen} />
          ))}
        </div>
      </div>
      <div style={{ fontSize: 16, lineHeight: 1.5, margin: "4px 0 12px", paddingLeft: 12, borderLeft: "3px solid " + THEME.highlight, maxWidth: 920 }}>
        <span style={{ fontFamily: MONO, fontSize: 12, color: THEME.highlight, marginRight: 8 }}>{SCREENS[screen].toUpperCase()}</span>
        {question}
      </div>
      <div style={{ display: "flex", flexWrap: "wrap", gap: 16, alignItems: "flex-start" }}>
        <div style={{ flex: "1 1 460px", minWidth: 0, maxWidth: 680 }}>{stage}</div>
        <div style={{ flex: "1 1 300px", minWidth: 0, display: "flex", flexDirection: "column", gap: 10 }}>{side}</div>
      </div>
      {below}
      <div style={{ display: "flex", justifyContent: "space-between", marginTop: 14 }}>
        <Btn onClick={back} disabled={screen === 0}>
          ← Back
        </Btn>
        <Btn onClick={next} disabled={screen === 3} strong>
          {screen === 3 ? "Last screen" : "Next: " + SCREENS[screen + 1] + " →"}
        </Btn>
      </div>
    </div>
  );
}

// ---------- markers and arrows ----------
function Marker({ x, y, shape, color, r, hollow }) {
  const s = r || 6;
  const fill = hollow ? THEME.bg : color;
  const stroke = hollow ? color : THEME.bg;
  const sw = hollow ? 2 : 1.5;
  // no JSX spread here: the build's wrapper breaks Babel's _extends helper
  if (shape === "square") return <rect x={x - s * 0.85} y={y - s * 0.85} width={s * 1.7} height={s * 1.7} fill={fill} stroke={stroke} strokeWidth={sw} />;
  if (shape === "diamond") return <polygon points={x + "," + (y - s * 1.15) + " " + (x + s * 1.15) + "," + y + " " + x + "," + (y + s * 1.15) + " " + (x - s * 1.15) + "," + y} fill={fill} stroke={stroke} strokeWidth={sw} />;
  if (shape === "triangle") return <polygon points={x + "," + (y - s * 1.2) + " " + (x + s * 1.1) + "," + (y + s * 0.8) + " " + (x - s * 1.1) + "," + (y + s * 0.8)} fill={fill} stroke={stroke} strokeWidth={sw} />;
  return <circle cx={x} cy={y} r={s} fill={fill} stroke={stroke} strokeWidth={sw} />;
}
function Arrow({ x1, y1, x2, y2, color, width, dash, head }) {
  const dx = x2 - x1;
  const dy = y2 - y1;
  const L = Math.hypot(dx, dy);
  if (L < 1e-6) return null;
  const ux = dx / L;
  const uy = dy / L;
  const h = Math.min(head || 9, L * 0.6);
  const bx = x2 - ux * h;
  const by = y2 - uy * h;
  return (
    <g>
      <line x1={x1} y1={y1} x2={bx} y2={by} stroke={color} strokeWidth={width || 2} strokeDasharray={dash} strokeLinecap="round" />
      <polygon points={x2 + "," + y2 + " " + (bx - uy * h * 0.45) + "," + (by + ux * h * 0.45) + " " + (bx + uy * h * 0.45) + "," + (by - ux * h * 0.45)} fill={color} />
    </g>
  );
}
function SvgLabel({ x, y, children, color, size, anchor, fa, weight }) {
  return (
    <text
      x={x}
      y={y}
      textAnchor={anchor || "start"}
      fontFamily={fa ? FA : MONO}
      fontSize={size || (fa ? 15 : 12)}
      fontWeight={weight || 400}
      fill={color || THEME.text}
      stroke={THEME.bg}
      strokeWidth={3.5}
      paintOrder="stroke"
      style={{ userSelect: "none" }}
    >
      {children}
    </text>
  );
}

// ---------- a 2-D vector plane: words as points, drag the ones marked drag ----------
// points: [{ id, label, x, y, color, shape, fa, drag, hollow, r, lx, ly, arrow }]
// arrows: [{ x1, y1, x2, y2, color, dash, width, label, fa }]   (data coordinates)
// extra(P): extra SVG drawn under the points, P(x, y) -> [px, py]
function WordPlane({ xr, yr, points, arrows, onDrag, extra, top, W, H, ariaLabel, origin, ticks, axisLabels }) {
  const w = W || 560;
  const h = H || 420;
  const pad = 26;
  const ref = useRef(null);
  const [drag, setDrag] = useState(null);
  const X = (x) => pad + ((x - xr[0]) / (xr[1] - xr[0])) * (w - 2 * pad);
  const Y = (y) => h - pad - ((y - yr[0]) / (yr[1] - yr[0])) * (h - 2 * pad);
  const P = (x, y) => [X(x), Y(y)];
  const toData = useCallback(
    (e) => {
      const p = svgPoint(ref.current, e);
      const x = xr[0] + ((p.x - pad) / (w - 2 * pad)) * (xr[1] - xr[0]);
      const y = yr[0] + ((h - pad - p.y) / (h - 2 * pad)) * (yr[1] - yr[0]);
      return [clamp(x, xr[0], xr[1]), clamp(y, yr[0], yr[1])];
    },
    [xr, yr, w, h]
  );
  const down = useCallback(
    (e) => {
      const id = e.target.getAttribute("data-id");
      if (!id || !onDrag) return;
      e.preventDefault();
      try {
        ref.current.setPointerCapture(e.pointerId);
      } catch (err) {
        // capture is a nicety: dragging still works without it
      }
      setDrag(id);
      const d = toData(e);
      onDrag(id, d[0], d[1]);
    },
    [onDrag, toData]
  );
  const move = useCallback(
    (e) => {
      if (!drag) return;
      const d = toData(e);
      onDrag(drag, d[0], d[1]);
    },
    [drag, onDrag, toData]
  );
  const up = useCallback(() => setDrag(null), []);
  const step = ticks || 1;
  const gx = [];
  for (let v = Math.ceil(xr[0] / step) * step; v <= xr[1] + 1e-9; v += step) gx.push(v);
  const gy = [];
  for (let v = Math.ceil(yr[0] / step) * step; v <= yr[1] + 1e-9; v += step) gy.push(v);
  const anyDrag = points.some((p) => p.drag);
  return (
    <svg
      ref={ref}
      viewBox={"0 0 " + w + " " + h}
      role="img"
      aria-label={ariaLabel || "vector plane"}
      onPointerDown={down}
      onPointerMove={move}
      onPointerUp={up}
      onPointerCancel={up}
      style={{ width: "100%", display: "block", background: THEME.bg, border: "1px solid " + THEME.grid, borderRadius: 6, touchAction: anyDrag ? "none" : "auto", cursor: drag ? "grabbing" : "default" }}
    >
      {gx.map((v) => (
        <line key={"x" + v} x1={X(v)} x2={X(v)} y1={pad} y2={h - pad} stroke={THEME.grid} strokeWidth={Math.abs(v) < 1e-9 && origin !== false ? 1.2 : 0.5} opacity={Math.abs(v) < 1e-9 && origin !== false ? 1 : 0.55} />
      ))}
      {gy.map((v) => (
        <line key={"y" + v} y1={Y(v)} y2={Y(v)} x1={pad} x2={w - pad} stroke={THEME.grid} strokeWidth={Math.abs(v) < 1e-9 && origin !== false ? 1.2 : 0.5} opacity={Math.abs(v) < 1e-9 && origin !== false ? 1 : 0.55} />
      ))}
      {axisLabels ? (
        <g>
          <SvgLabel x={w - pad} y={h - 8} anchor="end" color={THEME.muted} size={11}>
            {axisLabels[0]}
          </SvgLabel>
          <SvgLabel x={8} y={16} color={THEME.muted} size={11}>
            {axisLabels[1]}
          </SvgLabel>
        </g>
      ) : null}
      {extra ? extra(P) : null}
      {(arrows || []).map((a, i) => {
        const p1 = P(a.x1, a.y1);
        const p2 = P(a.x2, a.y2);
        return (
          <g key={"a" + i}>
            <Arrow x1={p1[0]} y1={p1[1]} x2={p2[0]} y2={p2[1]} color={a.color || THEME.muted} dash={a.dash} width={a.width} />
            {a.label ? (
              <SvgLabel x={(p1[0] + p2[0]) / 2 + 6} y={(p1[1] + p2[1]) / 2 - 6} color={a.color || THEME.muted} fa={a.fa} size={a.fa ? 13 : 11}>
                {a.label}
              </SvgLabel>
            ) : null}
          </g>
        );
      })}
      {points.map((p) => {
        const c = P(p.x, p.y);
        const col = p.color || THEME.accent;
        return (
          <g key={p.id} opacity={p.dim ? 0.38 : 1}>
            {p.arrow ? <Arrow x1={X(0)} y1={Y(0)} x2={c[0]} y2={c[1]} color={col} width={1.4} dash="4 3" head={7} /> : null}
            <Marker x={c[0]} y={c[1]} shape={p.shape} color={col} r={p.r || 6} hollow={p.hollow} />
            {p.label ? (
              <SvgLabel x={c[0] + (p.lx === undefined ? 9 : p.lx)} y={c[1] + (p.ly === undefined ? -8 : p.ly)} color={p.labelColor || THEME.text} fa={p.fa} anchor={p.anchor}>
                {p.label}
              </SvgLabel>
            ) : null}
            {p.drag ? <circle data-id={p.id} cx={c[0]} cy={c[1]} r={17} fill="transparent" stroke={col} strokeDasharray="3 3" strokeWidth={1} style={{ cursor: "grab" }} /> : null}
          </g>
        );
      })}
      {top ? top(P) : null}
    </svg>
  );
}

// ---------- horizontal bars: one row per item ----------
// rows: [{ label, value, color, fa, note, mark }]; range [lo, hi]; lo < 0 draws a centre line at 0
function Bars({ rows, range, digits, labelW, ariaLabel, valueFmt, rowH }) {
  const W = 600;
  const lw = labelW || 120;
  const rh = rowH || 24;
  const x0 = lw + 8;
  const x1 = W - 74;
  const lo = range ? range[0] : 0;
  const hi = range ? range[1] : 1;
  const X = (v) => x0 + ((clamp(v, lo, hi) - lo) / (hi - lo)) * (x1 - x0);
  const zero = X(Math.max(lo, Math.min(0, hi)));
  return (
    <svg viewBox={"0 0 " + W + " " + (rows.length * rh + 8)} style={{ width: "100%", display: "block" }} role="img" aria-label={ariaLabel || "bar chart"}>
      {rows.map((r, i) => {
        const y = 4 + i * rh;
        const v = r.value;
        const ok = isFinite(v);
        const a = ok ? X(v) : zero;
        return (
          <g key={i}>
            <text x={lw} y={y + rh / 2 + 5} textAnchor="end" fontFamily={r.fa ? FA : MONO} fontSize={r.fa ? 14.5 : 12} fill={THEME.text}>
              {r.label}
            </text>
            <rect x={x0} y={y + 3} width={x1 - x0} height={rh - 6} fill={THEME.bg} stroke={THEME.grid} strokeWidth={0.8} />
            <rect x={Math.min(a, zero)} y={y + 5} width={Math.abs(a - zero)} height={rh - 10} fill={r.color || THEME.accent} opacity={0.9} />
            {lo < 0 ? <line x1={zero} x2={zero} y1={y + 1} y2={y + rh - 1} stroke={THEME.muted} strokeWidth={1} /> : null}
            <text x={x1 + 8} y={y + rh / 2 + 5} fontFamily={MONO} fontSize={12} fill={THEME.text}>
              {ok ? (valueFmt ? valueFmt(v) : fmt(v, digits === undefined ? 2 : digits)) : "—"}
              {r.mark ? " " + r.mark : ""}
            </text>
          </g>
        );
      })}
    </svg>
  );
}

// ---------- a grid of numbers: co-occurrence counts, attention weights, masks ----------
// values[i][j]; shade(v) -> 0..1 opacity of the accent fill; text(v, i, j) -> cell text; onCell(i, j)
function Matrix({ rowLabels, colLabels, values, shade, text, onCell, hiRow, hiCol, cell, faRows, faCols, colorFor, ariaLabel, rowW, colH, mark }) {
  const cw = cell || 46;
  const rw = rowW || 86;
  const ch = colH || 30;
  const R = rowLabels.length;
  const C = colLabels.length;
  const W = rw + C * cw + 4;
  const H = ch + R * cw + 4;
  return (
    <svg viewBox={"0 0 " + W + " " + H} style={{ width: "100%", maxWidth: W * 1.25, display: "block" }} role="img" aria-label={ariaLabel || "matrix"}>
      {colLabels.map((c, j) => (
        <text key={"c" + j} x={rw + j * cw + cw / 2} y={ch - 9} textAnchor="middle" fontFamily={faCols ? FA : MONO} fontSize={faCols ? 13.5 : 11.5} fill={hiCol === j ? THEME.highlight : THEME.text}>
          {c}
        </text>
      ))}
      {rowLabels.map((r, i) => (
        <text key={"r" + i} x={rw - 8} y={ch + i * cw + cw / 2 + 5} textAnchor="end" fontFamily={faRows ? FA : MONO} fontSize={faRows ? 13.5 : 11.5} fill={hiRow === i ? THEME.highlight : THEME.text}>
          {r}
        </text>
      ))}
      {values.map((row, i) =>
        row.map((v, j) => {
          const x = rw + j * cw;
          const y = ch + i * cw;
          const op = shade ? shade(v, i, j) : 0;
          const col = colorFor ? colorFor(v, i, j) : THEME.accent;
          const hi = hiRow === i || hiCol === j;
          const m = mark ? mark(v, i, j) : null;
          return (
            <g key={i + "-" + j} onClick={onCell ? () => onCell(i, j) : undefined} style={{ cursor: onCell ? "pointer" : "default" }}>
              <rect x={x + 1} y={y + 1} width={cw - 2} height={cw - 2} rx={3} fill={THEME.bg} stroke={hi ? THEME.highlight : THEME.grid} strokeWidth={hi ? 1.5 : 0.8} />
              <rect x={x + 2} y={y + 2} width={cw - 4} height={cw - 4} rx={2} fill={col} opacity={clamp(op, 0, 1)} />
              <text x={x + cw / 2} y={y + cw / 2 + 4.5} textAnchor="middle" fontFamily={MONO} fontSize={cw < 40 ? 10.5 : 12} fill={THEME.text} stroke={THEME.bg} strokeWidth={op > 0.55 ? 3 : 0} paintOrder="stroke">
                {text ? text(v, i, j) : fmt(v, 2)}
              </text>
              {m ? (
                <text x={x + cw - 6} y={y + 12} textAnchor="end" fontFamily={MONO} fontSize={10} fill={THEME.highlight}>
                  {m}
                </text>
              ) : null}
            </g>
          );
        })
      )}
    </svg>
  );
}
// @common-end
