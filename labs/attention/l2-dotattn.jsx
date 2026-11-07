import { useState, useCallback } from "react";
import { THEME, MONO } from "./common";

// Lesson 2 - Score, normalize, mix (Attention, slides 20-21, 23).
// Aha: attention is three steps. Score each key against the query (here a dot product), turn the scores
// into weights that are positive and sum to 1 (softmax), and return the weighted sum of the values.
// Because the weights are positive and sum to 1, the output always lies inside the values' convex hull.

const K = [
  [1, 0],
  [0, 1],
  [1, 1],
];
const KN = ["k₁", "k₂", "k₃"];
const QR = [-3.5, 3.5];
const OR = [-0.6, 1.6];
const TARGET = [0.3, 0.75];
const TOL = 0.05;

function attend(q, mode, beta) {
  const s = K.map((k) => beta * dot(q, k));
  let w;
  if (mode === "sum") {
    const z = s.reduce((a, b) => a + b, 0);
    w = s.map((v) => (Math.abs(z) < 1e-9 ? NaN : v / z));
  } else w = softmax(s);
  const o = [w[0] * K[0][0] + w[1] * K[1][0] + w[2] * K[2][0], w[0] * K[0][1] + w[1] * K[1][1] + w[2] * K[2][1]];
  return { s, w, o };
}
function hull(P) {
  const pts = K.map((k) => P(k[0], k[1]));
  return <polygon points={pts.map((p) => p[0] + "," + p[1]).join(" ")} fill={rgba(THEME.accent, 0.12)} stroke={THEME.accent} strokeDasharray="4 3" />;
}
function keyPoints() {
  return K.map((k, i) => ({ id: "k" + i, label: KN[i] + " = v" + (i + 1) + " = [" + k.join(", ") + "]", x: k[0], y: k[1], color: THEME.accent, shape: "square", r: 6 }));
}
function outPlane(r, extraPts, ariaLabel) {
  const ok = isFinite(r.o[0]) && isFinite(r.o[1]);
  const inside = ok && r.o[0] >= OR[0] && r.o[0] <= OR[1] && r.o[1] >= OR[0] && r.o[1] <= OR[1];
  const pts = keyPoints().concat(extraPts || []);
  if (inside) pts.push({ id: "out", label: "output", x: r.o[0], y: r.o[1], color: THEME.highlight, shape: "diamond", r: 8 });
  return (
    <div>
      <WordPlane xr={OR} yr={OR} ticks={0.5} points={pts} extra={hull} W={420} H={360} ariaLabel={ariaLabel || "Output of attention and the triangle of values"} />
      {!inside ? <div style={{ fontFamily: MONO, fontSize: 12.5, color: THEME.warn, marginTop: 4 }}>✕ output {ok ? "(" + fmt(r.o[0], 2) + ", " + fmt(r.o[1], 2) + ") is off this chart" : "is undefined (the scores sum to 0)"}</div> : null}
    </div>
  );
}

export default function DotAttnLesson({ initialScreen }) {
  const [screen, setScreen] = useState(initialScreen || 0);
  const [q, setQ] = useState([1, 2]);
  const [mode, setMode] = useState("softmax");
  const [beta, setBeta] = useState(1);
  const [qc, setQc] = useState([1.5, 1.0]);

  const onDrag = useCallback((id, x, y) => {
    if (id === "q") setQ([x, y]);
  }, []);
  const onDragC = useCallback((id, x, y) => {
    if (id === "q") setQc([x, y]);
  }, []);

  const qPlane = (qq, drag) => (
    <WordPlane
      xr={QR}
      yr={QR}
      points={keyPoints().concat([{ id: "q", label: "q = [" + fmt(qq[0], 2) + ", " + fmt(qq[1], 2) + "]", x: qq[0], y: qq[1], color: THEME.third, shape: "triangle", r: 8, drag: true, arrow: true }])}
      onDrag={drag}
      W={420}
      H={360}
      ariaLabel="Drag the query q"
    />
  );
  const weightBars = (r) => (
    <Bars rows={K.map((k, i) => ({ label: "α" + (i + 1) + " (" + KN[i] + ")", value: r.w[i], color: r.w[i] < 0 ? THEME.warn : THEME.accent, mark: "score " + fmt(r.s[i], 2) }))} range={[-0.5, 1]} labelW={90} ariaLabel="attention weights" />
  );
  const twoUp = (left, right) => (
    <div style={{ display: "flex", flexWrap: "wrap", gap: 8 }}>
      <div style={{ flex: "1 1 260px", minWidth: 0 }}>
        <Label>query plane: drag q</Label>
        {left}
      </div>
      <div style={{ flex: "1 1 260px", minWidth: 0 }}>
        <Label>output = Σ αᵢ vᵢ (zoomed)</Label>
        {right}
      </div>
    </div>
  );

  let question, stage, side;
  if (screen === 0) {
    const r = attend(q, "sum", 1);
    question = (
      <>
        <b>Where we are:</b> lesson 1 showed that each decoder step needs its own weighted mix of the source states. Where do the weights come from? Attention is a function: one query vector attends to a set of vectors and returns a vector (slide 20). <b>The puzzle:</b> this is slide 21's version. Score each key by a dot product with q, divide each score by their sum, and mix. It works for q = [1, 2]. Drag q around: where does it break, and why?
      </>
    );
    stage = (
      <>
        {twoUp(qPlane(q, onDrag), outPlane(r))}
        {weightBars(r)}
      </>
    );
    side = (
      <>
        <Panel>
          <Readout
            rows={[
              ["scores q·kᵢ", r.s.map((v) => fmt(v, 2)).join(", ")],
              ["weights = score ÷ sum", r.w.map((v) => fmt(v, 3)).join(", "), r.w.some((v) => !(v >= 0)) ? THEME.warn : THEME.text],
              ["output", isFinite(r.o[0]) ? "[" + fmt(r.o[0], 3) + ", " + fmt(r.o[1], 3) + "]" : "undefined"],
            ]}
          />
          <Goal met={r.w.every((v) => v >= 0) && isFinite(r.o[0])}>all weights are between 0 and 1</Goal>
        </Panel>
        <Note>At q = [1, 2] the scores are 1, 2, 3, the weights 1/6, 2/6, 3/6, and the output is [2/3, 5/6], as on slide 21. Now drag q into the lower-left quarter.</Note>
      </>
    );
  } else if (screen === 1) {
    const r = attend(q, mode, beta);
    question = (
      <>
        <b>Claim to test:</b> softmax fixes the normalization. Every weight is exp(score) divided by the sum of exps, so weights are always positive and sum to 1, whatever the scores (slide 23). Switch between the two normalizations and drag q to the same bad spot. Then raise the sharpness β, which multiplies every score. What happens to the weights as the scores grow?
      </>
    );
    stage = (
      <>
        {twoUp(qPlane(q, onDrag), outPlane(r))}
        {weightBars(r)}
      </>
    );
    side = (
      <>
        <Panel>
          <Pills
            options={[
              ["sum", "÷ sum (slide 21)"],
              ["softmax", "softmax"],
            ]}
            value={mode}
            onPick={setMode}
          />
          <Slider id="a2-b" label="sharpness β (scores × β)" value={beta} min={0.1} max={6} stepSize={0.05} onChange={setBeta} display={"β = " + fmt(beta, 2)} />
          <Readout
            rows={[
              ["largest weight", fmt(Math.max.apply(null, r.w), 3)],
              ["weights sum to", fmt(r.w.reduce((a, b) => a + b, 0), 3)],
            ]}
          />
        </Panel>
        <Note>With softmax the output can never leave the dashed triangle, because it is an average of v₁, v₂ and v₃ with positive weights. A large β makes the weights nearly one-hot, so the output snaps to one key.</Note>
      </>
    );
  } else if (screen === 2) {
    const r = attend(q, "softmax", 1);
    question = (
      <>
        <b>What you found:</b> attention scores every key against the query, turns the scores into positive weights that sum to 1 with a softmax, and returns the weighted sum of the values. The output is always inside the values' convex hull, so it is a blend of what is there. These are Bahdanau's equations 5–6 for one decoder step (slide 23). Drag q and read every term.
      </>
    );
    stage = (
      <>
        {twoUp(qPlane(q, onDrag), outPlane(r))}
        <Eq size={12.5}>
          1 score: e_t,j = a(s_t−1, h_j) here a(q, k) = q·k → {r.s.map((v) => fmt(v, 2)).join(", ")}
          <br />
          2 normalize: α_t,j = exp(e_t,j) / Σₖ exp(e_t,k) → {r.w.map((v) => fmt(v, 3)).join(", ")}
          <br />
          3 mix: c_t = Σⱼ α_t,j h_j → [{fmt(r.o[0], 3)}, {fmt(r.o[1], 3)}]
        </Eq>
      </>
    );
    side = (
      <>
        <Panel>
          <Readout
            rows={[
              ["t", "target (decoder) step"],
              ["j, k", "source positions"],
              ["s_t−1", "previous decoder state = the query"],
              ["h_j", "encoder states = keys and values"],
            ]}
          />
        </Panel>
        <Note>Slides 20–23. Here keys and values are the same vectors. Next lesson: Bahdanau did not use a plain dot product. He used a small learned network as the scoring function a(·,·).</Note>
      </>
    );
  } else {
    const r = attend(qc, "softmax", 1);
    const err = Math.hypot(r.o[0] - TARGET[0], r.o[1] - TARGET[1]);
    const ok = err <= TOL;
    const tgt = { id: "t", label: "target", x: TARGET[0], y: TARGET[1], color: THEME.ok, shape: "circle", r: 7, hollow: true };
    question = (
      <>
        Use softmax attention over the same three keys. Find a query q that makes the output land within {TOL} of the target point [{TARGET[0]}, {TARGET[1]}] (the hollow circle). You can only move q. The keys and values stay fixed.
      </>
    );
    stage = (
      <>
        {twoUp(qPlane(qc, onDragC), outPlane(r, [tgt], "Output and target"))}
        {weightBars(r)}
      </>
    );
    side = (
      <>
        <Panel edge={ok ? THEME.ok : THEME.grid}>
          <Readout
            rows={[
              ["output", "[" + fmt(r.o[0], 3) + ", " + fmt(r.o[1], 3) + "]"],
              ["distance to target", fmt(err, 3), ok ? THEME.ok : THEME.text],
              ["weights", r.w.map((v) => fmt(v, 2)).join(", ")],
            ]}
          />
          <Goal met={ok}>output within {TOL} of the target</Goal>
          <Hint
            hints={[
              "The output is α₁k₁ + α₂k₂ + α₃k₃ = [α₁ + α₃, α₂ + α₃]. Work out which weights the target needs first.",
              "The target needs α₂ ≈ 0.70, α₁ ≈ 0.25 and α₃ only ≈ 0.05. k₃ = k₁ + k₂ scores q₁ + q₂, so to keep k₃ small both q₁ and q₂ must be negative, with q₂ the less negative.",
            ]}
          />
        </Panel>
        <Note>Getting a key's weight down is as important as getting another's up: the query decides the whole distribution, not just the winner.</Note>
      </>
    );
  }

  return <LessonFrame screen={screen} onScreen={setScreen} title="Score, normalize, mix" question={question} stage={stage} side={side} />;
}
