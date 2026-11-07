import { useState, useMemo, useCallback } from "react";
import { THEME, MONO } from "./common";

// Lesson 5 - Why divide by √d_k (Attention, slides 40-43, 62).
// Aha: a dot product of two random d_k-dimensional vectors has standard deviation √d_k. Big scores push the
// softmax to a one-hot, where its gradient is nearly zero. Dividing by √d_k puts the scores back on a unit
// scale, whatever the dimension.

// @sim-start
const KEYS = 8;
// raw dot products q·k for random q, k with independent N(0, 1) components
function rawScores(d, draws, seed) {
  const r = mulberry32(seed);
  const out = [];
  for (let t = 0; t < draws; t++) {
    const q = new Float64Array(d);
    for (let i = 0; i < d; i++) q[i] = randn(r);
    const row = [];
    for (let k = 0; k < KEYS; k++) {
      let s = 0;
      for (let i = 0; i < d; i++) s += q[i] * randn(r);
      row.push(s);
    }
    out.push(row);
  }
  return out;
}
function avgTop(rows, c) {
  let t = 0;
  for (const row of rows) t += Math.max.apply(null, softmax(row.map((s) => s / c)));
  return t / rows.length;
}
// @sim-end

const TOP_LO = 0.32;
const TOP_HI = 0.41;
const DRAWS = 250;

function Hist({ vals, range, label }) {
  const W = 600;
  const H = 150;
  const B = 41;
  const lo = range[0];
  const hi = range[1];
  const counts = new Array(B).fill(0);
  let out = 0;
  for (const v of vals) {
    if (v < lo || v > hi) out++;
    counts[clamp(Math.floor(((v - lo) / (hi - lo)) * B), 0, B - 1)]++;
  }
  const mx = Math.max(1, ...counts);
  const bw = (W - 40) / B;
  const X = (v) => 20 + ((v - lo) / (hi - lo)) * (W - 40);
  return (
    <svg viewBox={"0 0 " + W + " " + H} style={{ width: "100%", display: "block" }} role="img" aria-label={label}>
      {counts.map((c, i) => (
        <rect key={i} x={20 + i * bw + 1} y={H - 26 - (c / mx) * (H - 44)} width={bw - 2} height={(c / mx) * (H - 44)} fill={THEME.accent} opacity={0.75} />
      ))}
      <line x1={20} x2={W - 20} y1={H - 26} y2={H - 26} stroke={THEME.grid} />
      {[lo, lo / 2, 0, hi / 2, hi].map((t) => (
        <SvgLabel key={t} x={X(t)} y={H - 9} anchor="middle" color={THEME.muted} size={10.5}>
          {fmt(t, Math.abs(hi) < 10 ? 1 : 0)}
        </SvgLabel>
      ))}
      <SvgLabel x={W - 20} y={14} anchor="end" color={THEME.muted} size={11}>
        {label + (out ? " · " + out + " off the chart" : "")}
      </SvgLabel>
    </svg>
  );
}

export default function ScaleLesson({ initialScreen }) {
  const [screen, setScreen] = useState(initialScreen || 0);
  const [lg, setLg] = useState(2);
  const [scaled, setScaled] = useState(false);
  const [heads, setHeads] = useState(3);
  const [dk, setDk] = useState(64);
  const [c64, setC64] = useState(1);
  const [c256, setC256] = useState(1);

  const d = Math.pow(2, lg);
  const rows = useMemo(() => rawScores(d, DRAWS, 11 + lg), [d, lg]);
  const flat = useMemo(() => rows.reduce((a, r) => a.concat(r), []), [rows]);
  const div = screen === 1 && scaled ? Math.sqrt(d) : 1;
  const shown = flat.map((s) => s / div);
  const sd = stdev(shown);
  const w0 = softmax(rows[0].map((s) => s / div));
  const top = Math.max.apply(null, w0);
  const rng = Math.max(4, Math.ceil(3.2 * sd));

  const r64 = useMemo(() => rawScores(64, DRAWS, 900), []);
  const r256 = useMemo(() => rawScores(256, DRAWS, 901), []);

  const weightsBars = (w) => <Bars rows={w.map((v, i) => ({ label: "key " + (i + 1), value: v, color: v === Math.max.apply(null, w) ? THEME.highlight : THEME.accent }))} range={[0, 1]} labelW={60} rowH={19} ariaLabel="softmax weights over 8 keys" />;

  let question, stage, side;
  if (screen === 0) {
    question = (
      <>
        <b>Where we are:</b> lesson 4 built self-attention from dot products q·k. In the real model, q and k have d_k = 64 numbers each (slide 41). <b>The puzzle:</b> take random q and k whose entries have mean 0 and variance 1, as at initialization. Raise d_k from 4 to 256 and watch two things: how spread out the scores get, and what the softmax over 8 keys does with them. Why would that hurt training?
      </>
    );
    stage = (
      <>
        <Hist vals={shown} range={[-rng, rng]} label={"q·k for " + DRAWS * KEYS + " random pairs, d_k = " + d} />
        <Label>softmax over 8 keys for one query</Label>
        {weightsBars(w0)}
      </>
    );
    side = (
      <>
        <Panel>
          <Slider id="a5-d" label="key dimension d_k" value={lg} min={0} max={9} stepSize={1} onChange={setLg} display={"d_k = " + d} />
          <Readout
            rows={[
              ["standard deviation of q·k", fmt(sd, 2)],
              ["√d_k", fmt(Math.sqrt(d), 2)],
              ["largest weight", fmt(top, 4)],
              ["its gradient α(1 − α)", fmt(top * (1 - top), 4)],
            ]}
          />
        </Panel>
        <Note>The gradient of a softmax weight with respect to its own score is α(1 − α). It peaks at 0.25 and drops to nearly 0 when α is close to 1.</Note>
      </>
    );
  } else if (screen === 1) {
    question = (
      <>
        <b>Claim to test:</b> if q and k have independent entries with mean 0 and variance 1, then Var(q·k) = d_k, so the scores' standard deviation is √d_k. Dividing by √d_k brings it back to 1 (slide 40). Turn the division on and sweep d_k again. Does the softmax still saturate? Then check slide 40's two-key example below.
      </>
    );
    stage = (
      <>
        <Hist vals={shown} range={[-rng, rng]} label={(scaled ? "q·k / √d_k" : "q·k") + ", d_k = " + d} />
        <Label>softmax over 8 keys for one query</Label>
        {weightsBars(w0)}
      </>
    );
    side = (
      <>
        <Panel>
          <Slider id="a5-d2" label="key dimension d_k" value={lg} min={0} max={9} stepSize={1} onChange={setLg} display={"d_k = " + d} />
          <Toggle label="divide scores by √d_k" value={scaled} onChange={setScaled} />
          <Readout
            rows={[
              ["standard deviation of scores", fmt(sd, 2)],
              ["largest weight", fmt(top, 3)],
              ["its gradient α(1 − α)", fmt(top * (1 - top), 4)],
            ]}
          />
        </Panel>
        <Panel>
          <Label>slide 40, d_k = 64, two keys</Label>
          <Readout
            rows={[
              ["raw scores [8, 0] → softmax", "[" + softmax([8, 0]).map((v) => fmt(v, 4)).join(", ") + "]"],
              ["÷ 8 → [1, 0] → softmax", "[" + softmax([1, 0]).map((v) => fmt(v, 3)).join(", ") + "]"],
              ["gradient α(1 − α), raw", fmt(softmax([8, 0])[0] * softmax([8, 0])[1], 4)],
              ["gradient α(1 − α), ÷ 8", fmt(softmax([1, 0])[0] * softmax([1, 0])[1], 3)],
            ]}
          />
        </Panel>
      </>
    );
  } else if (screen === 2) {
    const h = Math.pow(2, heads);
    const dh = 512 / h;
    question = (
      <>
        <b>What you found:</b> unscaled dot products grow like √d_k, so the softmax saturates and its gradients vanish. Dividing by √d_k keeps the scores near unit scale for any d_k. Transformer Base also splits its 512 dimensions into 8 heads of 64, each with its own W_Q, W_K, W_V, so different heads can learn different patterns (slide 41). Change the number of heads.
      </>
    );
    stage = (
      <>
        <Eq size={12.5}>
          Var(q·k) = d_k, Std(q·k) = √d_k
          <br />
          Attention(Q, K, V) = softmax(QKᵀ / √d_k) V
          <br />
          head_i = Attention(XW_Q,i, XW_K,i, XW_V,i), MultiHead = Concat(head_1 … head_h) W_O
          <br />
          encoder layer: y = LN(x + MultiHead(x)), out = LN(y + FFN(y)) (post-norm, slide 42)
          <br />
          LN(u)_j = γ_j (u_j − μ) / √(σ² + ε) + β_j, over the d features of one token (slide 43)
        </Eq>
        <Bars
          rows={[
            { label: "W_Q, W_K, W_V, all heads", value: 3 * 512 * dh * h, color: THEME.accent },
            { label: "W_O (512 × 512)", value: 512 * 512, color: THEME.highlight },
          ]}
          range={[0, 800000]}
          valueFmt={(v) => v.toLocaleString("en-US")}
          labelW={200}
          ariaLabel="parameter count per attention layer"
        />
      </>
    );
    side = (
      <>
        <Panel>
          <Slider id="a5-h" label="number of heads h" value={heads} min={0} max={5} stepSize={1} onChange={setHeads} display={"h = " + h} />
          <Readout
            rows={[
              ["d_k = d_v per head", "512 / " + h + " = " + dh],
              ["scores divided by", "√" + dh + " = " + fmt(Math.sqrt(dh), 2)],
              ["projection parameters", (3 * 512 * dh * h).toLocaleString("en-US") + " (same for every h)"],
            ]}
          />
        </Panel>
        <Note>Slides 40–43 and 62. More heads do not add parameters: each head is narrower. Transformer Base uses 8 heads, model width 512 and FFN width 2048 (65M parameters). Big uses 16 heads and width 1024 (213M).</Note>
      </>
    );
  } else {
    const t64 = avgTop(r64, c64);
    const t256 = avgTop(r256, c256);
    const ok64 = t64 >= TOP_LO && t64 <= TOP_HI;
    const ok256 = t256 >= TOP_LO && t256 <= TOP_HI;
    const rowsShow = dk === 64 ? r64 : r256;
    const cShow = dk === 64 ? c64 : c256;
    const wShow = softmax(rowsShow[0].map((s) => s / cShow));
    question = (
      <>
        Forget the formula for a moment. For two key sizes, d_k = 64 and d_k = 256, choose the number every score is divided by. Aim for a softmax over 8 keys that is neither flat nor one-hot: the largest weight should average between {TOP_LO} and {TOP_HI} over {DRAWS} random queries. What do your two divisors have in common?
      </>
    );
    stage = (
      <>
        <Hist vals={rowsShow.reduce((a, r) => a.concat(r.map((s) => s / cShow)), [])} range={[-6, 6]} label={"scores ÷ " + fmt(cShow, 1) + ", d_k = " + dk} />
        <Label>softmax over 8 keys, one query</Label>
        {weightsBars(wShow)}
      </>
    );
    side = (
      <>
        <Panel edge={ok64 && ok256 ? THEME.ok : THEME.grid}>
          <Pills
            options={[
              [64, "show d_k = 64"],
              [256, "show d_k = 256"],
            ]}
            value={dk}
            onPick={setDk}
          />
          <Slider id="a5-c64" label="divisor for d_k = 64" value={c64} min={1} max={32} stepSize={0.1} onChange={setC64} display={"÷ " + fmt(c64, 1)} />
          <Slider id="a5-c256" label="divisor for d_k = 256" value={c256} min={1} max={32} stepSize={0.1} onChange={setC256} display={"÷ " + fmt(c256, 1)} />
          <Readout
            rows={[
              ["average largest weight, d_k = 64", fmt(t64, 3), ok64 ? THEME.ok : THEME.text],
              ["average largest weight, d_k = 256", fmt(t256, 3), ok256 ? THEME.ok : THEME.text],
            ]}
          />
          <Goal met={ok64}>d_k = 64 in the useful range</Goal>
          <Goal met={ok256}>d_k = 256 in the useful range</Goal>
          <Hint
            hints={[
              "A flat softmax has largest weight 1/8 = 0.125, a one-hot has 1. The useful range is what scores with a standard deviation of about 1 give.",
              "The raw scores' spread is √d_k. Try divisors near √64 and √256.",
            ]}
          />
        </Panel>
        {ok64 && ok256 ? <Note>Your divisors: {fmt(c64, 1)} and {fmt(c256, 1)}. Their ratio is {fmt(c256 / c64, 2)}, close to √(256/64) = 2: the divisor grows with √d_k.</Note> : null}
      </>
    );
  }

  return <LessonFrame screen={screen} onScreen={setScreen} title="Why divide by √d_k" question={question} stage={stage} side={side} />;
}
