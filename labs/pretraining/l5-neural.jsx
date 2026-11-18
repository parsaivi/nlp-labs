import { useState, useCallback, useEffect, useRef, useMemo } from "react";
import { THEME, MONO } from "./common";

// Lesson 5 - Share strength: from counts to a neural language model (Pretraining, slides 44-60).
// Aha: a count table treats dog and cat as unrelated symbols. Add-one smoothing removes the zeros but gives
// every unseen pair the same share. A neural LM looks words up in one shared table of vectors C and feeds
// the vector to one shared predictor, so a word's vector is what the model knows about it: when training
// pulls cat's vector near dog's, evidence about "dog runs" gives "cat runs" probability too.

// @sim-start
const CTX = ["[BOS]", "the", "a", "dog", "cat", "sleeps", "runs"];
const OUT = ["the", "a", "dog", "cat", "sleeps", "runs", "[EOS]"];
// [context index, next index, count] from the slide-35 corpus
const PAIRS = [
  [0, 0, 5], [0, 1, 3], [1, 2, 3], [1, 3, 2], [2, 2, 2], [2, 3, 1],
  [3, 4, 3], [3, 5, 2], [4, 4, 3], [5, 6, 6], [6, 6, 2],
];
const HID = 6;
const WD = 0.01;
const LR = 0.03;
const STEPS = 1500;
function initNet(seed) {
  const r = mulberry32(seed);
  const C = CTX.map(() => [randn(r) * 0.5, randn(r) * 0.5]);
  const H = [];
  const d = [];
  for (let j = 0; j < HID; j++) {
    H.push([randn(r) * 0.8, randn(r) * 0.8]);
    d.push(randn(r) * 0.2);
  }
  const U = OUT.map(() => {
    const row = [];
    for (let j = 0; j < HID; j++) row.push(randn(r) * 0.5);
    return row;
  });
  const b = OUT.map(() => 0);
  return { C, H, d, U, b };
}
// vector x (a row of C) -> tanh hidden layer -> |V| logits -> softmax
function forward(net, x) {
  const h = net.H.map((w, j) => Math.tanh(w[0] * x[0] + w[1] * x[1] + net.d[j]));
  const y = net.U.map((u, k) => {
    let s = net.b[k];
    for (let j = 0; j < HID; j++) s += u[j] * h[j];
    return s;
  });
  return { h, p: softmax(y) };
}
function lossOf(net) {
  let L = 0;
  let N = 0;
  for (const q of PAIRS) {
    L -= q[2] * Math.log(forward(net, net.C[q[0]]).p[q[1]]);
    N += q[2];
  }
  return L / N;
}
function zerosLike(net) {
  return { C: net.C.map((r) => r.map(() => 0)), H: net.H.map((r) => r.map(() => 0)), d: net.d.map(() => 0), U: net.U.map((r) => r.map(() => 0)), b: net.b.map(() => 0) };
}
function newOpt(net) {
  return { t: 0, m: zerosLike(net), v: zerosLike(net) };
}
// one full-batch step on mean cross-entropy + weight decay, with Adam. The gradient reaches C too.
function trainStep(net, opt) {
  const g = zerosLike(net);
  let N = 0;
  for (const q of PAIRS) {
    const c = q[0];
    const o = q[1];
    const k = q[2];
    N += k;
    const x = net.C[c];
    const f = forward(net, x);
    const dy = f.p.map((p, i) => (p - (i === o ? 1 : 0)) * k);
    const dh = new Array(HID).fill(0);
    for (let i = 0; i < OUT.length; i++) {
      g.b[i] += dy[i];
      for (let j = 0; j < HID; j++) {
        g.U[i][j] += dy[i] * f.h[j];
        dh[j] += dy[i] * net.U[i][j];
      }
    }
    for (let j = 0; j < HID; j++) {
      const da = dh[j] * (1 - f.h[j] * f.h[j]);
      g.d[j] += da;
      g.H[j][0] += da * x[0];
      g.H[j][1] += da * x[1];
      g.C[c][0] += da * net.H[j][0];
      g.C[c][1] += da * net.H[j][1];
    }
  }
  opt.t++;
  const b1 = 0.9;
  const b2 = 0.999;
  const upd = (p, gr, m, v) => {
    const G = gr / N + WD * p;
    const mm = b1 * m + (1 - b1) * G;
    const vv = b2 * v + (1 - b2) * G * G;
    const mh = mm / (1 - Math.pow(b1, opt.t));
    const vh = vv / (1 - Math.pow(b2, opt.t));
    return [p - (LR * mh) / (Math.sqrt(vh) + 1e-8), mm, vv];
  };
  for (const key of ["C", "H", "U"]) {
    for (let i = 0; i < net[key].length; i++) {
      for (let j = 0; j < net[key][i].length; j++) {
        const r = upd(net[key][i][j], g[key][i][j], opt.m[key][i][j], opt.v[key][i][j]);
        net[key][i][j] = r[0];
        opt.m[key][i][j] = r[1];
        opt.v[key][i][j] = r[2];
      }
    }
  }
  for (const key of ["d", "b"]) {
    for (let i = 0; i < net[key].length; i++) {
      const r = upd(net[key][i], g[key][i], opt.m[key][i], opt.v[key][i]);
      net[key][i] = r[0];
      opt.m[key][i] = r[1];
      opt.v[key][i] = r[2];
    }
  }
}
function trained() {
  const net = initNet(1);
  const opt = newOpt(net);
  for (let s = 0; s < STEPS; s++) trainStep(net, opt);
  return net;
}
// @sim-end

const XR = [-2.2, 2.2];
const YR = [-2.2, 2.2];
const SHAPE = { "[BOS]": "square", the: "square", a: "square", dog: "circle", cat: "triangle", sleeps: "diamond", runs: "diamond" };
const COL = { "[BOS]": THEME.muted, the: THEME.muted, a: THEME.muted, dog: THEME.third, cat: THEME.highlight, sleeps: THEME.accent, runs: THEME.accent };
const SHADE = [
  ["runs", "P(runs | word)"],
  ["sleeps", "P(sleeps | word)"],
  ["dog", "P(dog | word)"],
];

// render P(target | x) over the plane into a data URL, drawn as an SVG <image> under the points
function heatURL(net, target) {
  if (typeof document === "undefined") return "";
  const nx = 88;
  const ny = 88;
  const cv = document.createElement("canvas");
  cv.width = nx;
  cv.height = ny;
  const ctx = cv.getContext("2d");
  if (!ctx) return "";
  const img = ctx.createImageData(nx, ny);
  const k = OUT.indexOf(target);
  const n = parseInt(THEME.accent.slice(1), 16);
  for (let i = 0; i < ny; i++) {
    for (let j = 0; j < nx; j++) {
      const x = XR[0] + ((j + 0.5) / nx) * (XR[1] - XR[0]);
      const y = YR[1] - ((i + 0.5) / ny) * (YR[1] - YR[0]);
      const p = forward(net, [x, y]).p[k];
      const o = (i * nx + j) * 4;
      img.data[o] = (n >> 16) & 255;
      img.data[o + 1] = (n >> 8) & 255;
      img.data[o + 2] = n & 255;
      img.data[o + 3] = Math.round(p * 200);
    }
  }
  ctx.putImageData(img, 0, 0);
  return cv.toDataURL();
}
function heatExtra(url) {
  return function (P) {
    const a = P(XR[0], YR[1]);
    const b = P(XR[1], YR[0]);
    return url ? <image href={url} x={a[0]} y={a[1]} width={b[0] - a[0]} height={b[1] - a[1]} preserveAspectRatio="none" style={{ imageRendering: "pixelated" }} /> : null;
  };
}
function netPoints(net, only) {
  return CTX.filter((w) => !only || only.indexOf(w) >= 0).map((w) => {
    const i = CTX.indexOf(w);
    return { id: w, x: clamp(net.C[i][0], XR[0], XR[1]), y: clamp(net.C[i][1], YR[0], YR[1]), label: w, shape: SHAPE[w], color: COL[w], r: w === "cat" || w === "dog" ? 7 : 5 };
  });
}
function DistRow({ title, p }) {
  return (
    <>
      <Label>{title}</Label>
      <Bars rows={OUT.map((w, i) => ({ label: w, value: p[i], color: w === "runs" ? THEME.highlight : THEME.accent }))} range={[0, 1]} digits={3} labelW={64} rowH={19} ariaLabel={title} />
    </>
  );
}

export default function NeuralLesson({ initialScreen }) {
  const [screen, setScreen] = useState(initialScreen || 0);
  const [k, setK] = useState(1);
  const [shade, setShade] = useState("runs");
  const [running, setRunning] = useState(false);
  const [tick, setTick] = useState(0);
  const netRef = useRef(null);
  const optRef = useRef(null);
  const [m, setM] = useState(30);
  const [hid, setHid] = useState(50);
  const [vsz, setVsz] = useState(17000);
  const [pm, setPm] = useState(0.1);
  const [kit, setKit] = useState([1.2, 1.4]);
  const [cshade, setCshade] = useState("runs");

  if (!netRef.current) {
    netRef.current = initNet(1);
    optRef.current = newOpt(netRef.current);
  }
  const fixed = useMemo(() => trained(), []);
  const reset = useCallback(() => {
    netRef.current = initNet(1);
    optRef.current = newOpt(netRef.current);
    setRunning(false);
    setTick((t) => t + 1);
  }, []);
  const toggle = useCallback(() => setRunning((r) => !r), []);
  useEffect(() => {
    if (!running) return undefined;
    let raf = 0;
    let frame = 0;
    const loop = () => {
      const opt = optRef.current;
      const per = REDUCED ? STEPS : 20;
      for (let s = 0; s < per && opt.t < STEPS; s++) trainStep(netRef.current, opt);
      frame++;
      if (frame % 2 === 0 || opt.t >= STEPS) setTick((t) => t + 1);
      if (opt.t >= STEPS) {
        setRunning(false);
        return;
      }
      raf = requestAnimationFrame(loop);
    };
    raf = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(raf);
  }, [running]);
  const onKit = useCallback((id, x, y) => setKit([x, y]), []);

  const net = netRef.current;
  const heat = useMemo(() => (screen === 1 ? heatURL(net, shade) : ""), [screen, shade, tick, net]);
  const cheat = useMemo(() => (screen === 3 ? heatURL(fixed, cshade) : ""), [screen, cshade, fixed]);

  let question, stage, side;
  if (screen === 0) {
    const V = OUT.length;
    const counts = [0, 0, 0, 0, 3, 0, 0];
    const p = counts.map((c) => (c + k) / (3 + k * V));
    question = (
      <>
        <b>Where we are:</b> the bigram table gave "the cat runs" probability 0 because the pair cat → runs never occurred (slide 44). The standard repair is smoothing: add k to every count before dividing (slide 46). <b>The puzzle:</b> after "cat", the corpus saw only "sleeps". Move k. Is there any k for which the model prefers "cat runs" over the nonsense "cat the"?
      </>
    );
    stage = (
      <>
        <Bars rows={OUT.map((w, i) => ({ label: w, value: p[i], color: w === "runs" ? THEME.highlight : w === "the" ? THEME.warn : THEME.accent, mark: counts[i] + " seen" }))} range={[0, 1]} digits={3} labelW={70} ariaLabel="smoothed next-token probabilities after cat" />
        <Eq size={14}>
          P₊ₖ(w | cat) = (count(cat w) + k) / (3 + k·|V|), |V| = 7
          <br />
          k = 1: P(runs | cat) = (0 + 1) / (3 + 7) = 1/10
        </Eq>
      </>
    );
    side = (
      <>
        <Panel>
          <Slider id="n5-k" label="added count k" value={k} min={0} max={3} stepSize={0.01} onChange={setK} />
          <Readout
            rows={[
              ["P(runs | cat)", fmt(p[5], 4), THEME.highlight],
              ["P(the | cat)", fmt(p[0], 4), THEME.warn],
              ["difference", fmt(p[5] - p[0], 4)],
            ]}
          />
        </Panel>
        <Note>Smoothing avoids zero probabilities, but every unseen pair gets the same share. The table still does not know that dog and cat behave similarly (slide 46). Strong count-based models use better smoothing and backoff, with the same blind spot.</Note>
      </>
    );
  } else if (screen === 1) {
    const pc = forward(net, net.C[4]).p;
    const pd = forward(net, net.C[3]).p;
    const step = optRef.current.t;
    question = (
      <>
        <b>Claim to test:</b> Bengio et al. (2003) replace the table with a function: look the previous word up in a shared table of vectors C, then a shared network turns the vector into next-token probabilities (slides 49–54). Train it on the same 8 sentences and watch the 2-D word vectors move. Does "cat runs" get probability, although the corpus never shows it? Where does cat's vector end up?
      </>
    );
    stage = (
      <>
        <WordPlane xr={XR} yr={YR} points={netPoints(net)} extra={heatExtra(heat)} H={400} ariaLabel="context word vectors over a shading of the predicted probability" />
        <Note>Points: rows of C (the vector the model looks up for each previous word). Shading: the probability the predictor gives to the chosen next token if the previous word's vector sat at that spot.</Note>
      </>
    );
    side = (
      <>
        <Panel>
          <div style={{ display: "flex", gap: 6 }}>
            <Btn onClick={toggle} strong disabled={step >= STEPS}>
              {running ? "pause" : step >= STEPS ? "trained" : step ? "continue training" : "train"}
            </Btn>
            <Btn onClick={reset}>reset</Btn>
          </div>
          <Label>shading</Label>
          <Pills options={SHADE} value={shade} onPick={setShade} />
          <Readout
            rows={[
              ["training step", step + " / " + STEPS],
              ["mean cross-entropy", fmt(lossOf(net), 3) + " nats"],
              ["P(runs | cat), counts", "0.000"],
              ["P(runs | cat), neural", fmt(pc[5], 3), THEME.highlight],
              ["P(sleeps | cat), neural", fmt(pc[4], 3)],
              ["P(runs | dog), neural", fmt(pd[5], 3)],
              ["distance cat ↔ dog", fmt(Math.hypot(net.C[3][0] - net.C[4][0], net.C[3][1] - net.C[4][1]), 2)],
            ]}
          />
        </Panel>
        <Note>Toy: 2-D vectors, 6 tanh units, a small weight penalty. "the" and "a" end up together too, since both are followed by dog or cat.</Note>
      </>
    );
  } else if (screen === 2) {
    const n1 = 3;
    const params = vsz * m + n1 * m * hid + hid + hid * vsz + vsz;
    question = (
      <>
        <b>What you found:</b> the network never saw "cat runs", yet gives it about 0.11, because cat and dog are both followed by "sleeps", so training puts their vectors close, and nearby vectors get similar predictions. The prediction loss trains C and the predictor together (slide 55). This is Bengio's model with a 3-word window.
      </>
    );
    stage = (
      <>
        <Eq size={13}>
          x = [C(wₜ₋₃); C(wₜ₋₂); C(wₜ₋₁)] ∈ R^(3m) (concatenate in order, slide 53)
          <br />
          h = tanh(d + H x) · y = b + U h (+ W x, optional direct path)
          <br />
          p = softmax(y) over |V|
          <br />
          loss = −ln p(target): p(milk) = {fmt(pm, 2)} → loss {fmt(-Math.log(pm), 3)}
          <br />
          (slide 55: 0.1 → 2.303, 0.6 → 0.511)
        </Eq>
        <Label>Brown corpus, held-out test perplexity (slide 56, Bengio et al. 2003, Table 1)</Label>
        <Bars
          rows={[
            { label: "class-based n-gram", value: 312, color: THEME.grid },
            { label: "neural LM (MLP1)", value: 268, color: THEME.accent },
            { label: "neural + trigram (MLP10)", value: 252, color: THEME.third },
          ]}
          range={[0, 350]}
          digits={0}
          labelW={200}
          ariaLabel="Brown corpus perplexities"
        />
      </>
    );
    side = (
      <>
        <Panel>
          <Slider id="n5-pm" label="p(milk | the cat drinks)" value={pm} min={0.01} max={1} stepSize={0.01} onChange={setPm} />
          <Slider id="n5-m" label="vector size m" value={m} min={2} max={100} stepSize={1} onChange={setM} display={"m = " + m} />
          <Slider id="n5-h" label="hidden units" value={hid} min={10} max={200} stepSize={1} onChange={setHid} />
          <Slider id="n5-v" label="vocabulary |V|" value={vsz} min={1000} max={50000} stepSize={500} onChange={setVsz} display={vsz.toLocaleString("en-US")} />
          <Readout
            rows={[
              ["C: |V|·m", (vsz * m).toLocaleString("en-US")],
              ["H: 3m·hidden", (n1 * m * hid).toLocaleString("en-US")],
              ["U: hidden·|V|", (hid * vsz).toLocaleString("en-US")],
              ["total (no direct path)", params.toLocaleString("en-US")],
            ]}
          />
        </Panel>
        <Note>Training examples come from sliding a window over text: "the cat drinks milk slowly [EOS]" gives 3 windows of 3 words → next word (slide 52). Costs at the time: AP News took about 3 weeks on 40 CPUs for 5 epochs, and the window still truncates distant history (slide 57). Word2Vec (slide 59) learns vectors too, but it does not score a whole sequence, so it is not a language model.</Note>
      </>
    );
  } else {
    const pk = forward(fixed, kit).p;
    const g1 = pk[4] >= 0.7;
    const g2 = pk[5] >= 0.15;
    const pts = netPoints(fixed, ["the", "a", "dog", "cat", "sleeps"]).concat([{ id: "kitten", x: kit[0], y: kit[1], label: "kitten", shape: "triangle", color: THEME.ok, drag: true, hollow: true, r: 7 }]);
    question = (
      <>
        A new word, "kitten", enters the vocabulary with no training sentences at all. In a neural LM, choosing its vector is choosing what the model believes about it. With the trained predictor fixed, place kitten so that it naps like a cat and plays like a dog: P(sleeps | kitten) ≥ 0.7 and P(runs | kitten) ≥ 0.15. Neither cat nor dog meets both.
      </>
    );
    stage = <WordPlane xr={XR} yr={YR} points={pts} extra={heatExtra(cheat)} onDrag={onKit} H={400} ariaLabel="drag the kitten vector over the predictor's probability map" />;
    side = (
      <>
        <Panel edge={g1 && g2 ? THEME.ok : THEME.grid}>
          <Label>shading</Label>
          <Pills options={SHADE} value={cshade} onPick={setCshade} />
          <Readout
            rows={[
              ["P(sleeps | kitten)", fmt(pk[4], 3), g1 ? THEME.ok : THEME.text],
              ["P(runs | kitten)", fmt(pk[5], 3), g2 ? THEME.ok : THEME.text],
              ["for comparison: cat", fmt(forward(fixed, fixed.C[4]).p[4], 2) + " / " + fmt(forward(fixed, fixed.C[4]).p[5], 2)],
              ["for comparison: dog", fmt(forward(fixed, fixed.C[3]).p[4], 2) + " / " + fmt(forward(fixed, fixed.C[3]).p[5], 2)],
            ]}
          />
          <Goal met={g1}>P(sleeps | kitten) ≥ 0.7</Goal>
          <Goal met={g2}>P(runs | kitten) ≥ 0.15</Goal>
          <Hint hints={["Shade by P(runs) and then by P(sleeps). Where is each one high? Start from cat and move toward dog.", "The answer is a strip that passes between cat and dog and continues up and to the left. Start halfway between the two words."]} />
        </Panel>
        <DistRow title="P(next | kitten)" p={pk} />
      </>
    );
  }

  return <LessonFrame screen={screen} onScreen={setScreen} title="Share strength: from counts to a neural LM" question={question} stage={stage} side={side} />;
}
