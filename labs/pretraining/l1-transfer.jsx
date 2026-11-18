import { useState, useMemo, useCallback } from "react";
import { THEME, MONO } from "./common";

// Lesson 1 - Learn once, reuse (Pretraining, slides 7-19).
// Aha: a backbone trained on a rich source task turns inputs into features where a new task is easy.
// With a handful of labels, a new head on frozen features beats training from scratch; with more labels
// (or a shifted domain), fine-tuning the backbone too pulls ahead. Toy: each "leaf image" is a 2-D point,
// and the toy backbone's one feature is h = distance from a centre it learned on the source task.

// @sim-start
const CS = [0, 0]; // centre learned on the source task
const CT = [0.3, -0.2]; // target leaves (lessons 1-3): slightly off-centre
const CH = [0.7, -0.5]; // challenge: a new camera, further off-centre
const RAD = 1.25;
function makeLeaves(n, seed, ct) {
  const rng = mulberry32(seed);
  const out = [];
  for (let i = 0; i < n; i++) {
    const dmg = i % 2 === 0;
    const r = dmg ? Math.sqrt(rng()) * (RAD - 0.08) : RAD + 0.08 + rng() * 1.25;
    const t = rng() * 2 * Math.PI;
    let y = dmg ? 1 : 0;
    if (rng() < 0.04) y = 1 - y;
    out.push({ x: [ct[0] + r * Math.cos(t), ct[1] + r * Math.sin(t)], y });
  }
  return out;
}
function feat(p, c) {
  return Math.hypot(p[0] - c[0], p[1] - c[1]);
}
// head on the backbone feature: p(damaged) = sigmoid(a (tau - h)); learnC = fine-tune the backbone's centre too
function trainHead(data, c0, learnC, steps) {
  let a = 2;
  let tau = 1;
  const c = [c0[0], c0[1]];
  const lr = 0.3;
  for (let s = 0; s < steps; s++) {
    let ga = 0;
    let gt = 0;
    let gc0 = 0;
    let gc1 = 0;
    for (const d of data) {
      const h = feat(d.x, c);
      const e = sigmoid(a * (tau - h)) - d.y;
      ga += e * (tau - h);
      gt += e * a;
      if (learnC && h > 1e-6) {
        gc0 += e * a * ((d.x[0] - c[0]) / h);
        gc1 += e * a * ((d.x[1] - c[1]) / h);
      }
    }
    const n = data.length;
    a -= (lr * ga) / n;
    tau -= (lr * gt) / n;
    if (learnC) {
      c[0] -= (lr * gc0) / n;
      c[1] -= (lr * gc1) / n;
    }
  }
  return { a, tau, c };
}
// scratch: a small MLP (2 -> 10 tanh -> 1) from a seeded random start, trained on the raw input
function trainMLP(data, seed, steps) {
  const H = 10;
  const rng = mulberry32(seed);
  const W1 = [];
  const b1 = [];
  const W2 = [];
  for (let j = 0; j < H; j++) {
    W1.push([randn(rng) * 0.8, randn(rng) * 0.8]);
    b1.push(randn(rng) * 0.3);
    W2.push(randn(rng) * 0.5);
  }
  let b2 = 0;
  const lr = 0.5;
  const n = data.length;
  for (let s = 0; s < steps; s++) {
    const gW1 = W1.map(() => [0, 0]);
    const gb1 = new Array(H).fill(0);
    const gW2 = new Array(H).fill(0);
    let gb2 = 0;
    for (const d of data) {
      const h = W1.map((w, j) => Math.tanh(w[0] * d.x[0] + w[1] * d.x[1] + b1[j]));
      let z = b2;
      for (let j = 0; j < H; j++) z += W2[j] * h[j];
      const e = sigmoid(z) - d.y;
      gb2 += e;
      for (let j = 0; j < H; j++) {
        gW2[j] += e * h[j];
        const g = e * W2[j] * (1 - h[j] * h[j]);
        gW1[j][0] += g * d.x[0];
        gW1[j][1] += g * d.x[1];
        gb1[j] += g;
      }
    }
    for (let j = 0; j < H; j++) {
      W2[j] -= (lr * gW2[j]) / n;
      W1[j][0] -= (lr * gW1[j][0]) / n;
      W1[j][1] -= (lr * gW1[j][1]) / n;
      b1[j] -= (lr * gb1[j]) / n;
    }
    b2 -= (lr * gb2) / n;
  }
  return { W1, b1, W2, b2 };
}
function mlpScore(m, x) {
  let z = m.b2;
  for (let j = 0; j < m.W1.length; j++) z += m.W2[j] * Math.tanh(m.W1[j][0] * x[0] + m.W1[j][1] * x[1] + m.b1[j]);
  return z;
}
function accuracy(pred, data) {
  let k = 0;
  for (const d of data) if (pred(d.x) === d.y) k++;
  return k / data.length;
}
// @sim-end

const NS = [6, 10, 20, 40, 100, 200, 400];
// held-out accuracy (%) on 600 test leaves, averaged over 6 seeded draws, from the functions above
const CURVE = {
  scratch: [65.9, 69.0, 74.4, 83.5, 91.6, 94.6, 95.3],
  frozen: [82.8, 84.3, 87.0, 87.1, 88.1, 88.3, 88.3],
  fine: [78.8, 85.6, 90.2, 90.8, 93.1, 95.1, 95.5],
};
const PUZ = makeLeaves(8, 18, CT);
const TEST = makeLeaves(600, 999, CT);
const CH_TRAIN = makeLeaves(12, 31, CH);
const CH_TEST = makeLeaves(600, 999, CH);
const CH_SHOW = CH_TEST.slice(0, 160);
const XR = [-3, 3];
const YR = [-2.6, 2.6];

function leafPoints(data, prefix, hollow) {
  return data.map((d, i) => ({ id: prefix + i, x: d.x[0], y: d.x[1], shape: d.y ? "triangle" : "circle", color: d.y ? THEME.highlight : THEME.third, hollow: hollow, r: hollow ? 3 : 6, dim: hollow }));
}
// marching squares on f(x, y) > 0, returned as one SVG path in pixel units
function contourPath(f, P, n) {
  const dx = (XR[1] - XR[0]) / n;
  const dy = (YR[1] - YR[0]) / n;
  const v = [];
  for (let i = 0; i <= n; i++) {
    const row = [];
    for (let j = 0; j <= n; j++) row.push(f(XR[0] + j * dx, YR[0] + i * dy));
    v.push(row);
  }
  let d = "";
  const lerp = (a, b) => a / (a - b);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      const c = [v[i][j], v[i][j + 1], v[i + 1][j + 1], v[i + 1][j]];
      const x0 = XR[0] + j * dx;
      const y0 = YR[0] + i * dy;
      const pts = [];
      if (c[0] > 0 !== c[1] > 0) pts.push([x0 + lerp(c[0], c[1]) * dx, y0]);
      if (c[1] > 0 !== c[2] > 0) pts.push([x0 + dx, y0 + lerp(c[1], c[2]) * dy]);
      if (c[3] > 0 !== c[2] > 0) pts.push([x0 + lerp(c[3], c[2]) * dx, y0 + dy]);
      if (c[0] > 0 !== c[3] > 0) pts.push([x0, y0 + lerp(c[0], c[3]) * dy]);
      for (let k = 0; k + 1 < pts.length; k += 2) {
        const a = P(pts[k][0], pts[k][1]);
        const b = P(pts[k + 1][0], pts[k + 1][1]);
        d += "M" + a[0].toFixed(1) + " " + a[1].toFixed(1) + "L" + b[0].toFixed(1) + " " + b[1].toFixed(1);
      }
    }
  }
  return d;
}
function circleExtra(c, r, color, dash) {
  return function (P) {
    const o = P(c[0], c[1]);
    const e = P(c[0] + r, c[1]);
    return (
      <g>
        <circle cx={o[0]} cy={o[1]} r={Math.abs(e[0] - o[0])} fill={rgba(color, 0.07)} stroke={color} strokeWidth={2} strokeDasharray={dash} />
        <line x1={o[0] - 7} x2={o[0] + 7} y1={o[1]} y2={o[1]} stroke={color} strokeWidth={2} />
        <line x1={o[0]} x2={o[0]} y1={o[1] - 7} y2={o[1] + 7} stroke={color} strokeWidth={2} />
      </g>
    );
  };
}
// a 1-D strip: where each leaf lands on the backbone feature h, and the head's threshold
function FeatureStrip({ data, c, tau }) {
  const W = 600;
  const X = (v) => 30 + (clamp(v, 0, 3.2) / 3.2) * (W - 60);
  return (
    <svg viewBox={"0 0 " + W + " 64"} style={{ width: "100%", display: "block", marginTop: 6 }} role="img" aria-label="feature axis h">
      <line x1={30} x2={W - 30} y1={30} y2={30} stroke={THEME.grid} strokeWidth={1.5} />
      {[0, 1, 2, 3].map((t) => (
        <SvgLabel key={t} x={X(t)} y={56} anchor="middle" color={THEME.muted} size={11}>
          {String(t)}
        </SvgLabel>
      ))}
      <rect x={30} y={18} width={X(tau) - 30} height={24} fill={rgba(THEME.highlight, 0.1)} />
      <line x1={X(tau)} x2={X(tau)} y1={10} y2={50} stroke={THEME.accent} strokeWidth={2} />
      <SvgLabel x={X(tau) + 5} y={14} color={THEME.accent} size={11}>
        {"τ = " + fmt(tau, 2)}
      </SvgLabel>
      {data.map((d, i) => (
        <Marker key={i} x={X(feat(d.x, c))} y={30 + (i % 2 ? -5 : 5)} shape={d.y ? "triangle" : "circle"} color={d.y ? THEME.highlight : THEME.third} r={5} />
      ))}
      <SvgLabel x={W - 30} y={14} anchor="end" color={THEME.muted} size={11}>
        feature h = distance from ✚
      </SvgLabel>
    </svg>
  );
}

export default function TransferLesson({ initialScreen }) {
  const [screen, setScreen] = useState(initialScreen || 0);
  const [mode, setMode] = useState("raw");
  const [ang, setAng] = useState(30);
  const [off, setOff] = useState(0.2);
  const [tau, setTau] = useState(0.6);
  const [ni, setNi] = useState(0);
  const [strat, setStrat] = useState("frozen");
  const [classes, setClasses] = useState(2);
  const [freeze, setFreeze] = useState(true);
  const [cc, setCc] = useState([CS[0], CS[1]]);
  const [ctau, setCtau] = useState(1.2);
  const [unfrozen, setUnfrozen] = useState(false);

  const onDragC = useCallback((id, x, y) => setCc([x, y]), []);
  const onUnfreeze = useCallback((v) => {
    setUnfrozen(v);
    if (!v) setCc([CS[0], CS[1]]);
  }, []);
  const N = NS[ni];
  const fitted = useMemo(() => {
    if (screen !== 1) return null;
    const tr = makeLeaves(N, 100 + N, CT);
    if (strat === "scratch") {
      const m = trainMLP(tr, 7, 600);
      return { tr, kind: "mlp", m, acc: accuracy((x) => (mlpScore(m, x) > 0 ? 1 : 0), TEST) };
    }
    const m = trainHead(tr, CS, strat === "fine", 400);
    return { tr, kind: "head", m, acc: accuracy((x) => (sigmoid(m.a * (m.tau - feat(x, m.c))) > 0.5 ? 1 : 0), TEST) };
  }, [screen, N, strat]);

  let question, stage, side;
  if (screen === 0) {
    const t = (ang * Math.PI) / 180;
    const nrm = [Math.cos(t), Math.sin(t)];
    const predRaw = (x) => (nrm[0] * x[0] + nrm[1] * x[1] - off > 0 ? 1 : 0);
    const predBb = (x) => (feat(x, CS) < tau ? 1 : 0);
    const right = PUZ.filter((d) => (mode === "raw" ? predRaw(d.x) : predBb(d.x)) === d.y).length;
    const lineExtra = (P) => {
      const p0 = [nrm[0] * off, nrm[1] * off];
      const dir = [-nrm[1], nrm[0]];
      const a = P(p0[0] - dir[0] * 9, p0[1] - dir[1] * 9);
      const b = P(p0[0] + dir[0] * 9, p0[1] + dir[1] * 9);
      const m = P(p0[0], p0[1]);
      const q = P(p0[0] + nrm[0] * 0.6, p0[1] + nrm[1] * 0.6);
      return (
        <g>
          <line x1={a[0]} y1={a[1]} x2={b[0]} y2={b[1]} stroke={THEME.accent} strokeWidth={2} />
          <Arrow x1={m[0]} y1={m[1]} x2={q[0]} y2={q[1]} color={THEME.accent} width={1.5} />
          <SvgLabel x={q[0] + 4} y={q[1] - 4} color={THEME.accent} size={11}>
            ▲ side
          </SvgLabel>
        </g>
      );
    };
    question = (
      <>
        <b>Where we are:</b> we have a tokenizer, embeddings and attention, but in 2017 every NLP task still trained its own model (slides 6–8). Vision had a way out: train a backbone once on ImageNet, keep it, and replace only the head (slides 9–13). <b>The puzzle:</b> you have 8 labeled leaf photos, healthy ● or damaged ▲. Can a head alone separate them? Try a linear head on the raw input, then reuse a pretrained backbone.
      </>
    );
    stage = (
      <>
        <WordPlane xr={XR} yr={YR} points={leafPoints(PUZ, "p", false)} extra={mode === "raw" ? lineExtra : circleExtra(CS, tau, THEME.accent)} H={380} ariaLabel="eight labeled leaves" />
        {mode === "bb" ? <FeatureStrip data={PUZ} c={CS} tau={tau} /> : null}
      </>
    );
    side = (
      <>
        <Panel>
          <Label>head</Label>
          <Pills
            options={[
              ["raw", "linear head on raw input"],
              ["bb", "pretrained backbone + head"],
            ]}
            value={mode}
            onPick={setMode}
          />
          {mode === "raw" ? (
            <>
              <Slider id="t1-ang" label="line direction (degrees)" value={ang} min={0} max={360} stepSize={1} onChange={setAng} display={ang + "°"} />
              <Slider id="t1-off" label="line offset" value={off} min={-2.5} max={2.5} stepSize={0.01} onChange={setOff} />
            </>
          ) : (
            <Slider id="t1-tau" label="threshold τ on the feature h" value={tau} min={0.2} max={2.6} stepSize={0.01} onChange={setTau} />
          )}
          <Readout rows={[["leaves classified correctly", right + " / 8", right === 8 ? THEME.ok : THEME.text]]} />
          <Goal met={right === 8}>all 8 training leaves correct</Goal>
        </Panel>
        <Note>Toy model: each photo is a point in a 2-D input space. The pretrained backbone computes one feature, h = distance from ✚, a centre it learned on the source task. With the backbone, the head is a single threshold: h &lt; τ means damaged.</Note>
      </>
    );
  } else if (screen === 1) {
    const f = fitted;
    const ex =
      f.kind === "mlp"
        ? (P) => <path d={contourPath((x, y) => mlpScore(f.m, [x, y]), P, 60)} stroke={THEME.accent} strokeWidth={2} fill="none" />
        : circleExtra(f.m.c, f.m.tau, THEME.accent, strat === "fine" ? undefined : "6 4");
    const shown = f.tr.slice(0, 200);
    question = (
      <>
        <b>Claim to test:</b> transfer is a hypothesis, so compare on the same held-out leaves (slide 17). Three ways to train: <b>scratch</b> (random start, learn everything from the target labels), <b>frozen</b> (keep the backbone, train only the head, slide 14) and <b>fine-tune</b> (start from the backbone and update it too, slide 15). The target leaves sit slightly off the source centre. Which strategy wins with 6 labels? With 400? Where do the curves cross?
      </>
    );
    stage = (
      <>
        <WordPlane xr={XR} yr={YR} points={leafPoints(shown, "e", false).map((p) => Object.assign(p, { r: N > 40 ? 3.5 : 6 }))} extra={ex} H={360} ariaLabel="training leaves and learned boundary" />
        <Note>
          The boundary learned from these {N} labels{N > 200 ? " (200 drawn)" : ""}. {strat === "scratch" ? "The scratch MLP draws any shape it likes from 10 tanh units." : strat === "frozen" ? "Frozen: the dashed circle's centre ✚ cannot move; only its radius τ is learned." : "Fine-tuned: the centre ✚ moves with the head."}
        </Note>
      </>
    );
    side = (
      <>
        <Panel>
          <Label>strategy</Label>
          <Pills
            options={[
              ["scratch", "scratch"],
              ["frozen", "frozen + head"],
              ["fine", "fine-tune"],
            ]}
            value={strat}
            onPick={setStrat}
          />
          <Slider id="t1-n" label="labeled target leaves" value={ni} min={0} max={NS.length - 1} stepSize={1} onChange={setNi} display={String(N)} />
          <Readout rows={[["held-out accuracy (this draw)", fmt(f.acc * 100, 1) + "%"]]} />
        </Panel>
        <LinePlot
          series={[
            { label: "scratch", color: THEME.warn, shape: "square", dash: "5 4", pts: NS.map((n, i) => [n, CURVE.scratch[i]]) },
            { label: "frozen + head", color: THEME.third, shape: "circle", pts: NS.map((n, i) => [n, CURVE.frozen[i]]) },
            { label: "fine-tune", color: THEME.accent, shape: "triangle", pts: NS.map((n, i) => [n, CURVE.fine[i]]) },
          ]}
          xr={[5, 450]}
          yr={[60, 100]}
          xlog
          xticks={NS}
          yticks={[60, 70, 80, 90, 100]}
          xlabel="labeled target leaves (log scale)"
          ylabel="held-out accuracy %"
          vline={N}
          H={240}
          legendAt="br"
          ariaLabel="held-out accuracy against number of labels for three strategies"
        />
        <Note>Curves: average of 6 seeded draws, 600 held-out leaves.</Note>
      </>
    );
  } else if (screen === 2) {
    const headParams = 512 * classes + classes;
    question = (
      <>
        <b>What you found:</b> with 6 labels, the frozen backbone wins because the head has almost nothing to learn. Fine-tuning catches up by about 10 labels and wins later, because it can also move the features toward the target domain. Scratch needs about 100 labels to match. In symbols, the backbone maps the input to a feature vector and the head maps that vector to class scores. Change the number of target classes and the update route.
      </>
    );
    stage = (
      <>
        <Eq size={13}>
          h = f_θ(x) ∈ R⁵¹² (backbone) · ŷ = softmax(W h + b), W ∈ R^(C×512) (head)
          <br />
          frozen: update W, b only · fine-tune: update θ, W, b · scratch: θ starts random
        </Eq>
        <svg viewBox="0 0 600 150" style={{ width: "100%", display: "block" }} role="img" aria-label="backbone, feature, head and gradient route">
          <rect x={10} y={40} width={170} height={60} rx={6} fill={THEME.bg} stroke={freeze ? THEME.grid : THEME.highlight} strokeWidth={2} strokeDasharray={freeze ? "5 4" : undefined} />
          <SvgLabel x={95} y={68} anchor="middle">backbone f_θ</SvgLabel>
          <SvgLabel x={95} y={88} anchor="middle" color={freeze ? THEME.muted : THEME.highlight} size={11}>
            {freeze ? "frozen (no update)" : "updated"}
          </SvgLabel>
          <Arrow x1={180} y1={70} x2={250} y2={70} color={THEME.muted} />
          <SvgLabel x={262} y={74}>h ∈ R⁵¹²</SvgLabel>
          <Arrow x1={340} y1={70} x2={390} y2={70} color={THEME.muted} />
          <rect x={390} y={40} width={110} height={60} rx={6} fill={THEME.bg} stroke={THEME.highlight} strokeWidth={2} />
          <SvgLabel x={445} y={68} anchor="middle">head</SvgLabel>
          <SvgLabel x={445} y={88} anchor="middle" color={THEME.highlight} size={11}>
            {"512 → " + classes}
          </SvgLabel>
          <Arrow x1={500} y1={70} x2={560} y2={70} color={THEME.muted} />
          <SvgLabel x={566} y={74} size={11}>
            loss
          </SvgLabel>
          <Arrow x1={560} y1={120} x2={freeze ? 400 : 30} y2={120} color={THEME.highlight} dash="5 3" />
          <SvgLabel x={freeze ? 410 : 40} y={140} color={THEME.highlight} size={11}>
            {freeze ? "gradient stops at the head" : "gradient reaches the backbone"}
          </SvgLabel>
        </svg>
      </>
    );
    side = (
      <>
        <Panel>
          <Toggle label="freeze the backbone" value={freeze} onChange={setFreeze} />
          <Slider id="t1-c" label="target classes C" value={classes} min={2} max={1000} stepSize={1} onChange={setClasses} display={"C = " + classes} />
          <Readout
            rows={[
              ["new head parameters 512·C + C", headParams.toLocaleString("en-US")],
              ["old ImageNet head (C = 1,000)", "513,000"],
            ]}
          />
        </Panel>
        <Note>Slide 13: the 512-D representation is kept and the 1,000-way source head is replaced by a 512 → 2 head (healthy / damaged), only 1,026 numbers. Detection and segmentation need more than a new class layer (slide 16). Slide 18: a supervised label is only one way to pretrain; self-supervised objectives build their targets from the data itself. That is the idea we carry to text next.</Note>
      </>
    );
  } else {
    const predC = (x) => (feat(x, cc) < ctau ? 1 : 0);
    const trainAcc = accuracy(predC, CH_TRAIN);
    const testAcc = accuracy(predC, CH_TEST);
    const ok = testAcc >= 0.93;
    const pts = leafPoints(CH_SHOW, "h", true).concat(leafPoints(CH_TRAIN, "t", false));
    pts.push({ id: "c", x: cc[0], y: cc[1], shape: "diamond", color: THEME.accent, r: 4, drag: unfrozen, label: unfrozen ? "drag ✚" : "✚ frozen", lx: 10, ly: -10, labelColor: THEME.accent });
    question = (
      <>
        A new camera: the farm's photos are framed differently, so damaged leaves now cluster further from the source centre. You have 12 labels (solid marks). Hollow marks are held-out leaves the model never trains on. Reach at least 93% held-out accuracy. Is moving the head enough, or must the backbone change too?
      </>
    );
    stage = <WordPlane xr={XR} yr={YR} points={pts} extra={circleExtra(cc, ctau, THEME.accent, unfrozen ? undefined : "6 4")} onDrag={onDragC} H={400} ariaLabel="shifted target domain; drag the centre when unfrozen" />;
    side = (
      <>
        <Panel edge={ok ? THEME.ok : THEME.grid}>
          <Toggle label="unfreeze the backbone (fine-tune)" value={unfrozen} onChange={onUnfreeze} />
          <Slider id="t1-ct" label="head threshold τ" value={ctau} min={0.2} max={2.6} stepSize={0.01} onChange={setCtau} />
          <Readout
            rows={[
              ["training accuracy (12 labels)", fmt(trainAcc * 100, 0) + "%"],
              ["held-out accuracy (600 leaves)", fmt(testAcc * 100, 1) + "%", ok ? THEME.ok : THEME.text],
              ["backbone centre ✚", "(" + fmt(cc[0], 2) + ", " + fmt(cc[1], 2) + ")"],
            ]}
          />
          <Goal met={ok}>held-out accuracy ≥ 93%</Goal>
          <Hint hints={["With the backbone frozen, try every τ. What is the best held-out accuracy you can get? Where do the triangles cluster compared with ✚?", "Fine-tuning changes what the feature h measures. Unfreeze, drag ✚ into the middle of the triangles, then retune τ."]} />
        </Panel>
        <Note>Domain shift limits a frozen backbone (slide 17): the best τ around the old centre stays near 75%.</Note>
      </>
    );
  }

  return <LessonFrame screen={screen} onScreen={setScreen} title="Learn once, reuse: keep the backbone, swap the head" question={question} stage={stage} side={side} />;
}
