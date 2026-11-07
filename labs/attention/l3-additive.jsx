import { useState, useMemo, useCallback } from "react";
import { THEME, MONO } from "./common";

// Lesson 3 - Bahdanau: the decoder decides where to look (Attention, slides 22-30).
// Aha: the score e = vₐᵀ tanh(Wₐ s + Uₐ h) is a small learned network that compares the decoder state
// with each encoder state. The encoder states never change while decoding; only the decoder state does,
// so the decoder state is what moves the attention from word to word. Everything is trained by the
// translation loss alone.

const H = [
  [1, 0],
  [0, 1],
  [1, 1],
];
const SRC = ["I", "drink", "tea"];
const TGT = ["من", "چای", "می‌نوشم"];
// the alignment the translation needs: من ← I, چای ← tea, می‌نوشم ← drink
const NEED = [0, 2, 1];
// a fixed "trained" 2-unit scorer for screens 2-4 (hand-set): Wₐ = I, Uₐ below, vₐ = [3, 3]
const UA = [
  [1.6, -0.6],
  [-0.6, 1.6],
];
const VA = [3, 3];
const SR = [-3, 3];
const NEED_W = 0.6;

// slide 25: one hidden unit, e_j = v · tanh(W·s + U·h_j)
function oneUnit(s, W, U, v) {
  const a = W[0] * s[0] + W[1] * s[1];
  const e = H.map((h) => v * Math.tanh(a + U[0] * h[0] + U[1] * h[1]));
  const al = softmax(e);
  const c = [0, 1].map((k) => al[0] * H[0][k] + al[1] * H[1][k] + al[2] * H[2][k]);
  return { a, e, al, c };
}
// two hidden units with the fixed scorer
function twoUnit(s) {
  const e = H.map((h) => {
    const b = matVec(UA, h);
    return VA[0] * Math.tanh(s[0] + b[0]) + VA[1] * Math.tanh(s[1] + b[1]);
  });
  return { e, al: softmax(e) };
}
function encPoints() {
  return H.map((h, j) => ({ id: "h" + j, label: "h" + (j + 1) + " " + SRC[j], x: h[0], y: h[1], color: THEME.accent, shape: "square", r: 6 }));
}

export default function AdditiveLesson({ initialScreen }) {
  const [screen, setScreen] = useState(initialScreen || 0);
  const [W, setWv] = useState([0.5, 0]);
  const [U, setUv] = useState([-2, 2]);
  const [v, setV] = useState(1);
  const [s, setS] = useState([0.5, -1.5]);
  const [states, setStates] = useState([
    [-2.0, -2.0],
    [2.0, 2.0],
    [-2.2, 2.0],
  ]);

  const setW0 = useCallback((x) => setWv((p) => [x, p[1]]), []);
  const setW1 = useCallback((x) => setWv((p) => [p[0], x]), []);
  const setU0 = useCallback((x) => setUv((p) => [x, p[1]]), []);
  const setU1 = useCallback((x) => setUv((p) => [p[0], x]), []);
  const onDragS = useCallback((id, x, y) => {
    if (id === "s") setS([x, y]);
  }, []);
  const onDragStates = useCallback((id, x, y) => {
    const t = ["s0", "s1", "s2"].indexOf(id);
    if (t >= 0) setStates((S) => S.map((p, i) => (i === t ? [x, y] : p)));
  }, []);

  const alignOf = useCallback((st) => st.map((x) => twoUnit(x).al), []);

  let question, stage, side;
  if (screen === 0) {
    const r = oneUnit([1, 0], W, U, v);
    question = (
      <>
        <b>Where we are:</b> lesson 2 scored keys with a plain dot product. Bahdanau, Cho and Bengio (2015) let the network <i>learn</i> how to compare: a small layer takes the previous decoder state s and an encoder state h, and returns one score (slide 24). <b>The puzzle:</b> these are slide 25's numbers. s = [1, 0], h₁ = [1, 0], h₂ = [0, 1], h₃ = [1, 1], and one hidden unit. The decoder attends mostly to h₂. Which parameter decides that? Can you make it attend mostly to h₁ instead?
      </>
    );
    stage = (
      <>
        <Bars rows={SRC.map((w, j) => ({ label: "h" + (j + 1) + " " + w, value: r.al[j], color: THEME.accent, mark: "e = " + fmt(r.e[j], 3) }))} range={[0, 1]} digits={3} labelW={90} ariaLabel="attention weights" />
        <Eq size={12.5}>
          e(j) = v × tanh(W·s + U·h(j)), with W·s = {fmt(r.a, 2)}
          <br />
          e = [{r.e.map((x) => fmt(x, 3)).join(", ")}]
          <br />
          α = softmax(e) = [{r.al.map((x) => fmt(x, 3)).join(", ")}]
          <br />
          c = Σ α_j h_j = [{fmt(r.c[0], 3)}, {fmt(r.c[1], 3)}]
        </Eq>
      </>
    );
    side = (
      <>
        <Panel>
          <Slider id="a3-w0" label="W first entry" value={W[0]} min={-3} max={3} stepSize={0.05} onChange={setW0} />
          <Slider id="a3-w1" label="W second entry" value={W[1]} min={-3} max={3} stepSize={0.05} onChange={setW1} />
          <Slider id="a3-u0" label="U first entry" value={U[0]} min={-3} max={3} stepSize={0.05} onChange={setU0} />
          <Slider id="a3-u1" label="U second entry" value={U[1]} min={-3} max={3} stepSize={0.05} onChange={setU1} />
          <Slider id="a3-v" label="v" value={v} min={-3} max={3} stepSize={0.05} onChange={setV} />
          <Goal met={r.al[0] > 0.5}>h₁ gets more than half the weight</Goal>
        </Panel>
        <Note>Slide 26 gives α ≈ [0.087, 0.574, 0.340] and c ≈ [0.426, 0.913] for W = [0.5, 0], U = [−2, 2], v = 1. Move one slider at a time.</Note>
      </>
    );
  } else if (screen === 1) {
    const r = twoUnit(s);
    question = (
      <>
        <b>Claim to test:</b> in a trained model, Wₐ, Uₐ and vₐ are fixed, and so are the encoder states. The only thing that changes from one output word to the next is the decoder state s. So s is what decides where to look (slide 22). Here the scorer is fixed, with two hidden units. Drag s around. Can every source word win the attention, and where does s have to be for each?
      </>
    );
    stage = (
      <>
        <WordPlane xr={SR} yr={SR} points={encPoints().concat([{ id: "s", label: "decoder state s", x: s[0], y: s[1], color: THEME.highlight, shape: "triangle", r: 8, drag: true }])} onDrag={onDragS} W={460} H={400} ariaLabel="Drag the decoder state" />
        <Bars rows={SRC.map((w, j) => ({ label: "α " + w, value: r.al[j], color: r.al[j] >= NEED_W ? THEME.ok : THEME.accent, mark: r.al[j] >= NEED_W ? "◆" : "" }))} range={[0, 1]} labelW={80} ariaLabel="attention weights for this decoder state" />
      </>
    );
    side = (
      <>
        <Panel>
          <Readout
            rows={[
              ["e for I, drink, tea", r.e.map((x) => fmt(x, 2)).join(", ")],
              ["winner", SRC[argmax(r.al)] + " (" + fmt(Math.max.apply(null, r.al), 2) + ")"],
            ]}
          />
          <Eq size={12}>
            e_j = vₐᵀ tanh(Wₐ s + Uₐ h_j), Wₐ = I, vₐ = [3, 3]
            <br />
            Uₐ = [[1.6, −0.6], [−0.6, 1.6]]
          </Eq>
        </Panel>
        <Note>The encoder states (squares) never move during decoding. The scorer is hand-set here so that each source word can win somewhere. In the real model all three matrices are learned (slide 24).</Note>
      </>
    );
  } else if (screen === 2) {
    const A = alignOf(states);
    question = (
      <>
        <b>What you found:</b> Bahdanau attention scores each encoder state with a learned function of the previous decoder state, takes a softmax over source positions, and mixes the encoder states into a context for this step. Because the decoder state changes every step, the same encoder states get different weights each time. Nobody gives the model alignments: the translation loss trains the scorer along with everything else (slide 27).
      </>
    );
    stage = (
      <>
        <Matrix rowLabels={TGT} colLabels={SRC} values={A} shade={(x) => x} text={(x) => fmt(x, 2)} faRows cell={58} rowW={90} ariaLabel="alignment: weight of each source word for each target word" />
        <Label>English → French, WMT 2014 test set, BLEU (Bahdanau et al. 2015, Table 1)</Label>
        <Bars
          rows={[
            { label: "RNNencdec-50 (fixed context)", value: 17.82, color: THEME.muted },
            { label: "RNNsearch-50 (attention)", value: 26.75, color: THEME.accent },
          ]}
          range={[0, 30]}
          labelW={220}
          ariaLabel="BLEU with and without attention"
        />
      </>
    );
    side = (
      <>
        <Panel>
          <Eq size={12.5}>
            e_t,j = vₐᵀ tanh(Wₐ s_t−1 + Uₐ h_j)
            <br />
            α_t,j = exp(e_t,j) / Σₖ exp(e_t,k)
            <br />
            c_t = Σⱼ α_t,j h_j
          </Eq>
        </Panel>
        <Note>
          Slides 22–30. Attention adds 8.93 BLEU points in this comparison, and the gap grows with sentence length, because the fixed-context model degrades on long sentences (slide 30). The two models differ in more than attention, and attention weights are not a complete explanation of a prediction (slide 28). Next: Bahdanau still uses an RNN, so each state waits for the previous one. Can attention replace recurrence entirely (slide 31)?
        </Note>
      </>
    );
  } else {
    const A = alignOf(states);
    const ok = NEED.map((j, t) => A[t][j] >= NEED_W);
    const pts = encPoints().concat(
      states.map((p, t) => ({ id: "s" + t, label: "s" + t + " → " + TGT[t], fa: true, x: p[0], y: p[1], color: ok[t] ? THEME.ok : THEME.highlight, shape: "triangle", r: 8, drag: true }))
    );
    question = (
      <>
        You are the decoder RNN. The scorer is fixed. Before emitting each Persian word, place your state: s₀ before <Fa>من</Fa>, s₁ before <Fa>چای</Fa>, s₂ before <Fa>می‌نوشم</Fa>. Goal: the alignment the translation needs, from slide 22. <Fa>من</Fa> attends to I, <Fa>چای</Fa> to tea, and <Fa>می‌نوشم</Fa> to drink, each with weight at least {NEED_W}.
      </>
    );
    stage = (
      <>
        <WordPlane xr={SR} yr={SR} points={pts} onDrag={onDragStates} W={460} H={400} ariaLabel="Place the three decoder states" />
        <Matrix rowLabels={TGT} colLabels={SRC} values={A} shade={(x) => x} text={(x) => fmt(x, 2)} mark={(x, t, j) => (NEED[t] === j ? "★" : null)} faRows cell={54} rowW={90} ariaLabel="alignment matrix" />
      </>
    );
    side = (
      <>
        <Panel edge={ok.every(Boolean) ? THEME.ok : THEME.grid}>
          {TGT.map((t, i) => (
            <Goal key={i} met={ok[i]}>
              {t} attends to {SRC[NEED[i]]} ≥ {NEED_W} (now {fmt(A[i][NEED[i]], 2)})
            </Goal>
          ))}
          <Hint
            hints={[
              "Use what you found in Explore: each source word wins in its own region of the plane. Find the region for each, then put the right state there.",
              "tanh saturates, so a unit only tells the keys apart while Wₐs + Uₐh is near 0. I wins when s's first coordinate is around −1, drink when its second coordinate is around −1, and tea when s is near the origin.",
            ]}
          />
        </Panel>
        <Note>★ marks the cell each row needs. The order differs from English: Persian puts the verb last, so the alignment is not diagonal.</Note>
      </>
    );
  }

  return <LessonFrame screen={screen} onScreen={setScreen} title="Bahdanau: the decoder decides where to look" question={question} stage={stage} side={side} />;
}
