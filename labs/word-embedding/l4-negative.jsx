import { useState, useMemo, useCallback } from "react";
import { THEME, MONO } from "./common";

// Lesson 4 - Skip the huge softmax (Word Embedding, slides 23-24).
// Aha: the softmax needs a score for every word in the vocabulary. Negative sampling replaces it with
// k + 1 yes/no questions: push the real pair's score up (σ(s) → 1) and a few random pairs' scores
// down (σ(−s) → 1). One step pulls v toward u_context and pushes it away from the noise words it
// currently scores highest.

// output vectors u (2-D, made up) for a 12-word vocabulary; the center word is تیم, its real context فوتبال
const U = [
  { w: "فوتبال", u: [1.5, 0.5] },
  { w: "بازی", u: [1.2, 1.1] },
  { w: "کتاب", u: [-1.3, 0.8] },
  { w: "چتر", u: [-0.5, -1.4] },
  { w: "ساعت", u: [0.9, -1.2] },
  { w: "ما", u: [-0.2, 1.3] },
  { w: "کرد", u: [0.3, 0.4] },
  { w: "مجلس", u: [-1.5, -0.6] },
  { w: "دولت", u: [-1.2, -1.0] },
  { w: "قانون", u: [-0.8, -1.6] },
  { w: "جدید", u: [0.2, -0.6] },
  { w: "تصویب", u: [-1.6, 0.2] },
];
const POS = 0;
const PLAY = 1;
// made-up corpus counts; noise is drawn from count^0.75 (the paper's choice), never the positive word
const COUNT = [1200, 2500, 900, 120, 400, 6000, 8000, 700, 1500, 1100, 2000, 300];
const XR = [-3, 3];
const ETA = 0.6;
const TOL = 25;

function noiseCdf() {
  const p = COUNT.map((c, i) => (i === POS ? 0 : Math.pow(c, 0.75)));
  const z = p.reduce((a, b) => a + b, 0);
  let acc = 0;
  return p.map((x) => (acc += x / z));
}
const CDF = noiseCdf();
function drawNoise(k, round, forcePlay) {
  const rng = mulberry32(101 + round * 13);
  const out = forcePlay ? [PLAY] : [];
  let guard = 0;
  while (out.length < k && guard++ < 500) {
    const r = rng();
    let j = CDF.findIndex((c) => c >= r);
    if (j < 0) j = CDF.length - 1;
    if (j !== POS && out.indexOf(j) < 0) out.push(j);
  }
  return out;
}
// negative-sampling objective for one (center, context) pair and its gradient in v
function nsTerms(v, upos, unegs) {
  const sp = dot(upos, v);
  const sn = unegs.map((u) => dot(u, v));
  const loss = -Math.log(sigmoid(sp)) - sn.reduce((a, s) => a + Math.log(sigmoid(-s)), 0);
  const pull = 1 - sigmoid(sp);
  const push = sn.map((s) => sigmoid(s));
  const g = [pull * upos[0], pull * upos[1]];
  unegs.forEach((u, i) => {
    g[0] -= push[i] * u[0];
    g[1] -= push[i] * u[1];
  });
  return { sp, sn, loss, pull, push, g };
}
function angleBetween(a, b) {
  const c = clamp(cosine(a, b), -1, 1);
  return (Math.acos(c) * 180) / Math.PI;
}

// challenge rounds: predict which way one step moves v
const ROUNDS = [
  { v: [1.2, 1.0], pos: [1.4, 0.4], negs: [{ w: "کتاب", u: [1.2, 1.2] }, { w: "چتر", u: [-1.0, 0.8] }, { w: "ساعت", u: [0.2, -1.4] }] },
  { v: [-1.0, -0.5], pos: [1.4, 0.4], negs: [{ w: "کتاب", u: [-1.3, 0.8] }, { w: "چتر", u: [-0.5, -1.4] }, { w: "ساعت", u: [0.9, -1.2] }] },
  { v: [0.8, 0.6], pos: [1.5, 0.5], negs: [{ w: "بازی", u: [1.2, 1.1] }, { w: "ساعت", u: [0.9, -1.2] }, { w: "کتاب", u: [-1.3, 0.8] }] },
];

function uPoints(ids, roleOf) {
  return ids.map((j) => {
    const r = roleOf(j);
    return {
      id: "u" + j,
      label: "u " + U[j].w,
      fa: true,
      x: U[j].u[0],
      y: U[j].u[1],
      color: r === "pos" ? THEME.ok : r === "neg" ? THEME.warn : THEME.muted,
      shape: r === "pos" ? "circle" : r === "neg" ? "square" : "diamond",
      r: r === "other" ? 4.5 : 6.5,
      dim: r === "other",
    };
  });
}

export default function NegativeLesson({ initialScreen }) {
  const [screen, setScreen] = useState(initialScreen || 0);
  const [v, setV] = useState([0.4, -0.8]);
  const [logV, setLogV] = useState(6);
  const [k, setK] = useState(3);
  const [round, setRound] = useState(0);
  const [forcePlay, setForcePlay] = useState(false);
  const [kName, setKName] = useState(5);
  const [rIdx, setRIdx] = useState(0);
  const [guess, setGuess] = useState([[1.6, 1.6], [-0.4, -1.0], [1.4, 1.0]]);
  const [stepped, setStepped] = useState([false, false, false]);

  const onDragV = useCallback((id, x, y) => {
    if (id === "v") setV([x, y]);
  }, []);
  const redraw = useCallback(() => setRound((r) => r + 1), []);
  const onDragG = useCallback(
    (id, x, y) => {
      if (id !== "g" || stepped[rIdx]) return;
      setGuess((G) => G.map((p, i) => (i === rIdx ? [x, y] : p)));
    },
    [rIdx, stepped]
  );
  const takeStep = useCallback(() => setStepped((S) => S.map((s, i) => (i === rIdx ? true : s))), [rIdx]);
  const guessAgain = useCallback(() => setStepped((S) => S.map((s, i) => (i === rIdx ? false : s))), [rIdx]);
  const negs = useMemo(() => drawNoise(k, round, forcePlay), [k, round, forcePlay]);
  const T = nsTerms(
    v,
    U[POS].u,
    negs.map((j) => U[j].u)
  );
  const step = useCallback(() => {
    const t = nsTerms(
      v,
      U[POS].u,
      negs.map((j) => U[j].u)
    );
    setV([clamp(v[0] + ETA * t.g[0], XR[0], XR[1]), clamp(v[1] + ETA * t.g[1], XR[0], XR[1])]);
  }, [v, negs]);

  const vPoint = { id: "v", label: "v تیم", fa: true, x: v[0], y: v[1], color: THEME.highlight, shape: "triangle", r: 8, drag: true };
  const nsArrows = (TT, vv, upos, unegs, scale) => {
    const s = scale || 1;
    const out = [{ x1: vv[0], y1: vv[1], x2: vv[0] + s * TT.pull * upos[0], y2: vv[1] + s * TT.pull * upos[1], color: THEME.ok, width: 1.6, dash: "4 3" }];
    unegs.forEach((u, i) => out.push({ x1: vv[0], y1: vv[1], x2: vv[0] - s * TT.push[i] * u[0], y2: vv[1] - s * TT.push[i] * u[1], color: THEME.warn, width: 1.6, dash: "4 3" }));
    out.push({ x1: vv[0], y1: vv[1], x2: vv[0] + s * TT.g[0], y2: vv[1] + s * TT.g[1], color: THEME.highlight, width: 2.8, label: "step" });
    return out;
  };

  let question, stage, side;
  if (screen === 0) {
    const sc = U.map((x) => dot(x.u, v));
    const P = softmax(sc);
    const Vbig = Math.pow(10, logV);
    question = (
      <>
        <b>Where we are:</b> lesson 3 trained Skip-gram with a full softmax: every training pair scores all V words, so that the probabilities sum to 1. Our vocabulary had 12 words. A real one can have millions (slide 23). <b>The puzzle:</b> drag v (the input vector of <Fa>تیم</Fa>) until P(<Fa>فوتبال</Fa> | <Fa>تیم</Fa>) passes 0.5. Watch the bars: every time v moves, every score changes. Which of these scores did you actually need?
      </>
    );
    stage = (
      <>
        <WordPlane xr={XR} yr={XR} points={uPoints(U.map((_, j) => j), (j) => (j === POS ? "pos" : "neg")).map((p) => Object.assign({}, p, { dim: false, r: 5 })).concat([vPoint])} onDrag={onDragV} ariaLabel="Output vectors u of the vocabulary and the draggable input vector v" />
        <Label>softmax over all {U.length} words: P(o | تیم)</Label>
        <Bars rows={U.map((x, j) => ({ label: x.w, fa: true, value: P[j], color: j === POS ? THEME.ok : THEME.warn, mark: j === POS ? "●" : "" }))} range={[0, 1]} labelW={70} rowH={20} ariaLabel="softmax probabilities" />
      </>
    );
    side = (
      <>
        <Panel>
          <Readout
            rows={[
              ["P(فوتبال | تیم)", fmt(P[POS], 3), P[POS] > 0.5 ? THEME.ok : THEME.text],
              ["dot products for this one P", String(U.length)],
            ]}
          />
          <Goal met={P[POS] > 0.5}>P(فوتبال | تیم) &gt; 0.5</Goal>
        </Panel>
        <Panel>
          <Slider id="l4-v" label="a real vocabulary size V" value={logV} min={3} max={7} stepSize={0.1} onChange={setLogV} display={Math.round(Vbig).toLocaleString("en-US")} />
          <Readout
            rows={[
              ["dot products per training pair", Math.round(Vbig).toLocaleString("en-US")],
              ["for 1 billion training pairs", fmt(Vbig * 1e9, 0)],
            ]}
          />
        </Panel>
        <Note>The denominator Σⱼ exp(uⱼ·v) needs every word. Could you train with only the score of the real pair and a few others (slide 23)?</Note>
      </>
    );
  } else if (screen === 1) {
    const roleOf = (j) => (j === POS ? "pos" : negs.indexOf(j) >= 0 ? "neg" : "other");
    question = (
      <>
        <b>Claim to test:</b> instead of the full softmax, ask k + 1 yes/no questions. Is (<Fa>تیم</Fa>, <Fa>فوتبال</Fa>) a real pair? Then σ(s) should be near 1. Is (<Fa>تیم</Fa>, noise word) a real pair? Then σ(s) should be near 0, so σ(−s) near 1 (slide 24). Drag v or press "one step", and read the arrows: green pulls toward u_<Fa>فوتبال</Fa>, red pushes away from each noise word. Which noise word pushes hardest, and why?
      </>
    );
    stage = (
      <>
        <WordPlane xr={XR} yr={XR} points={uPoints(U.map((_, j) => j), roleOf).concat([vPoint])} arrows={nsArrows(T, v, U[POS].u, negs.map((j) => U[j].u))} onDrag={onDragV} ariaLabel="Negative sampling: pull toward the real context, push away from noise" />
        <Bars
          rows={[{ label: "σ(s) فوتبال", fa: true, value: sigmoid(T.sp), color: THEME.ok, mark: "real" }].concat(negs.map((j, i) => ({ label: "σ(−s) " + U[j].w, fa: true, value: sigmoid(-T.sn[i]), color: THEME.warn, mark: "noise" })))}
          range={[0, 1]}
          labelW={110}
          ariaLabel="sigmoid terms of the objective"
        />
      </>
    );
    side = (
      <>
        <Panel>
          <Slider id="l4-k" label="noise words per pair, k" value={k} min={1} max={5} stepSize={1} onChange={setK} display={"k = " + k} />
          <div style={{ display: "flex", gap: 6, flexWrap: "wrap", marginBottom: 6 }}>
            <Btn onClick={step} strong>
              one step
            </Btn>
            <Btn onClick={redraw}>draw new noise</Btn>
          </div>
          <Toggle label="the draw happens to include بازی" value={forcePlay} onChange={setForcePlay} />
          <Readout
            rows={[
              ["loss = −log σ(s₊) − Σ log σ(−sᵢ)", fmt(T.loss, 3)],
              ["pull weight 1 − σ(s₊)", fmt(T.pull, 3)],
              ["dot products this pair", String(k + 1) + " (not " + U.length + ")"],
            ]}
          />
        </Panel>
        <Note>Noise words are drawn at random, more often for frequent words. Turn on بازی: it really is a context of تیم, but here it is treated as noise. Noise samples are not verified false pairs (slide 24).</Note>
      </>
    );
  } else if (screen === 2) {
    const roleOf = (j) => (j === POS ? "pos" : negs.indexOf(j) >= 0 ? "neg" : "other");
    question = (
      <>
        <b>What you found:</b> each training pair now costs k + 1 dot products instead of V. The real pair's score is pushed up, and each noise pair's score is pushed down in proportion to how high it currently is. So one step moves v toward u_context and away from the noise words v already scores highest. Change k and V to compare the cost.
      </>
    );
    stage = <WordPlane xr={XR} yr={XR} points={uPoints(U.map((_, j) => j), roleOf).concat([vPoint])} arrows={nsArrows(T, v, U[POS].u, negs.map((j) => U[j].u))} onDrag={onDragV} ariaLabel="Negative sampling forces on v" />;
    side = (
      <>
        <Panel>
          <Eq size={12.5}>
            s(w, c) = u_c·v_w, σ(x) = 1 / (1 + e^(−x))
            <br />
            maximize log σ(s(w, c)) + Σᵢ₌₁..ₖ log σ(−s(w, nᵢ)), nᵢ ~ Pₙ
            <br />
            ∂/∂v = (1 − σ(s₊)) u_c − Σᵢ σ(sᵢ) u_nᵢ
          </Eq>
          <Slider id="l4-kn" label="k" value={kName} min={1} max={20} stepSize={1} onChange={setKName} display={String(kName)} />
          <Slider id="l4-vn" label="vocabulary V" value={logV} min={3} max={7} stepSize={0.1} onChange={setLogV} display={Math.round(Math.pow(10, logV)).toLocaleString("en-US")} />
          <Readout
            rows={[
              ["full softmax, per pair", Math.round(Math.pow(10, logV)).toLocaleString("en-US")],
              ["negative sampling, per pair", String(kName + 1)],
              ["work saved", "× " + fmt(Math.pow(10, logV) / (kName + 1), 0)],
            ]}
          />
        </Panel>
        <Note>Slide 24 (Mikolov et al. 2013, arXiv:1310.4546). The result is no longer a probability over the vocabulary: it is k + 1 separate yes/no classifiers, which is enough to learn good vectors. The paper draws noise from unigram counts raised to 3/4. Next lesson: what the trained vectors can do, such as analogies.</Note>
      </>
    );
  } else {
    const R = ROUNDS[rIdx];
    const unegs = R.negs.map((n) => n.u);
    const TT = nsTerms(R.v, R.pos, unegs);
    const g = guess[rIdx];
    const gd = [g[0] - R.v[0], g[1] - R.v[1]];
    const err = angleBetween(gd, TT.g);
    const done = stepped[rIdx];
    const okR = stepped.map((s, i) => {
      const r = ROUNDS[i];
      const t = nsTerms(
        r.v,
        r.pos,
        r.negs.map((n) => n.u)
      );
      const gg = guess[i];
      return s && angleBetween([gg[0] - r.v[0], gg[1] - r.v[1]], t.g) <= TOL;
    });
    const pts = [
      { id: "pos", label: "u فوتبال (real)", fa: true, x: R.pos[0], y: R.pos[1], color: THEME.ok, shape: "circle", r: 7 },
    ]
      .concat(R.negs.map((n, i) => ({ id: "n" + i, label: "u " + n.w + " (noise)", fa: true, x: n.u[0], y: n.u[1], color: THEME.warn, shape: "square", r: 7 })))
      .concat([
        { id: "v", label: "v", x: R.v[0], y: R.v[1], color: THEME.highlight, shape: "triangle", r: 8 },
        { id: "g", label: done ? "your guess" : "drag: your guess", x: g[0], y: g[1], color: THEME.third, shape: "diamond", r: 7, drag: !done },
      ]);
    const arrows = [{ x1: R.v[0], y1: R.v[1], x2: g[0], y2: g[1], color: THEME.third, width: 2, dash: "6 4" }];
    if (done) arrows.push({ x1: R.v[0], y1: R.v[1], x2: R.v[0] + ETA * TT.g[0], y2: R.v[1] + ETA * TT.g[1], color: THEME.highlight, width: 3, label: "actual step" });
    question = (
      <>
        Three new situations. In each, v sits somewhere, the real context is <Fa>فوتبال</Fa>, and three noise words were drawn. Drag the teal diamond to point where you think one step of negative sampling moves v, then take the step. Get within {TOL}° in all three rounds.
      </>
    );
    stage = <WordPlane key={"r" + rIdx} xr={XR} yr={XR} points={pts} arrows={arrows} onDrag={onDragG} ariaLabel="Predict the direction of one negative-sampling step" />;
    side = (
      <>
        <Panel edge={okR.every(Boolean) ? THEME.ok : THEME.grid}>
          <Pills
            options={[
              [0, "round 1"],
              [1, "round 2"],
              [2, "round 3"],
            ]}
            value={rIdx}
            onPick={setRIdx}
          />
          <div style={{ display: "flex", gap: 6, marginBottom: 8 }}>
            <Btn onClick={takeStep} disabled={done} strong>
              take the step
            </Btn>
            <Btn onClick={guessAgain} disabled={!done}>
              guess again
            </Btn>
          </div>
          {done ? (
            <Readout
              rows={[
                ["σ(s) of the real pair", fmt(sigmoid(TT.sp), 2)],
                ["σ(s) of " + R.negs.map((n) => n.w).join(" / "), TT.sn.map((s) => fmt(sigmoid(s), 2)).join(" / ")],
                ["angle between guess and step", fmt(err, 0) + "°", err <= TOL ? THEME.ok : THEME.warn],
              ]}
            />
          ) : null}
          {okR.map((o, i) => (
            <Goal key={i} met={o}>
              round {i + 1} within {TOL}°
            </Goal>
          ))}
          <Hint
            hints={[
              "Each word moves v along its own u: toward u_فوتبال, away from each noise u. How hard depends on how wrong its score is right now.",
              "The pull weight is 1 − σ(s₊), small when the real pair already scores high. The push weight of a noise word is σ(sᵢ), large when v points toward it. Add the arrows.",
            ]}
          />
        </Panel>
        <Note>Round 3 has بازی among the noise words. It is a real context of تیم, so this step pushes v the wrong way. With many draws this averages out, but it is why the slide warns that noise samples are not verified false pairs.</Note>
      </>
    );
  }

  return <LessonFrame screen={screen} onScreen={setScreen} title="Skip the huge softmax" question={question} stage={stage} side={side} />;
}
