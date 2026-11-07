import { useState, useMemo, useCallback } from "react";
import { THEME, MONO } from "./common";

// Lesson 1 - One-hot vectors treat every word alike (Word Embedding, slides 7-12).
// Aha: two different one-hot vectors always have dot product 0, so what a model learns about one word
// never reaches another. Dense vectors share evidence, because the score w·x is large for every word
// whose vector points the same way.

// 2-D "pretrained" vectors, hand-placed: sports words point right, politics words point up-left.
// The Olympics sits between the two groups on purpose.
const WORDS = [
  { fa: "فوتبال", en: "football", t: 1, e: [2.3, 0.8] },
  { fa: "بسکتبال", en: "basketball", t: 1, e: [1.4, 0.9] },
  { fa: "والیبال", en: "volleyball", t: 1, e: [1.5, 0.4] },
  { fa: "تنیس", en: "tennis", t: 1, e: [1.0, 1.1] },
  { fa: "المپیک", en: "Olympics", t: 1, e: [0.35, 1.35] },
  { fa: "سیاست", en: "politics", t: -1, e: [-0.7, 1.5] },
  { fa: "انتخابات", en: "election", t: -1, e: [-1.3, 1.0] },
  { fa: "مجلس", en: "parliament", t: -1, e: [-0.9, 0.6] },
  { fa: "دولت", en: "government", t: -1, e: [-0.25, 1.6] },
];
const N = WORDS.length;
const XR = [-2, 3];
const YR = [-1, 2.4];
const BASK = 1;

function topicColor(t) {
  return t > 0 ? THEME.accent : THEME.highlight;
}
function topicShape(t) {
  return t > 0 ? "circle" : "square";
}
// score of word i given labelled set: sum over labels of y * (x_i . x_label)
function scores(vecs, labels) {
  return vecs.map((x) => labels.reduce((s, j) => s + WORDS[j].t * dot(x, vecs[j]), 0));
}
function oneHot(i) {
  return WORDS.map((_, j) => (i === j ? 1 : 0));
}
function verdict(s) {
  if (Math.abs(s) < 1e-9) return 0;
  return s > 0 ? 1 : -1;
}

function OneHotTable({ a, b, onPick, labels }) {
  const vals = WORDS.map((_, i) => oneHot(i));
  const shade = useCallback((v, i) => (v ? 0.85 : i === a || i === b ? 0.12 : 0), [a, b]);
  const colorFor = useCallback((v, i) => (i === a ? THEME.highlight : i === b ? THEME.third : THEME.accent), [a, b]);
  const text = useCallback((v) => (v ? "1" : "0"), []);
  const mark = useCallback((v, i, j) => (labels && v && labels.indexOf(i) >= 0 ? "L" : null), [labels]);
  return (
    <div>
      <Matrix
        rowLabels={WORDS.map((w) => w.fa)}
        colLabels={WORDS.map((_, j) => "#" + (j + 1))}
        values={vals}
        shade={shade}
        colorFor={colorFor}
        text={text}
        onCell={onPick}
        mark={mark}
        faRows
        cell={38}
        rowW={78}
        ariaLabel="One-hot vectors: one row per word, a single 1 in its own coordinate"
      />
      <div style={{ fontFamily: MONO, fontSize: 11.5, color: THEME.muted, marginTop: 4 }}>… and 19,991 more columns of 0 for the rest of a 20,000-word vocabulary (slide 8)</div>
    </div>
  );
}

function Plane({ vecs, w, onDrag, dragId, labels, verdicts, showLine }) {
  const pts = WORDS.map((wd, i) => ({
    id: String(i),
    label: wd.fa + (verdicts ? (verdicts[i] === wd.t ? " ✓" : verdicts[i] === 0 ? " ?" : " ✕") : ""),
    fa: true,
    x: vecs[i][0],
    y: vecs[i][1],
    color: topicColor(wd.t),
    shape: topicShape(wd.t),
    drag: dragId === i,
    hollow: labels ? labels.indexOf(i) < 0 : false,
    r: labels && labels.indexOf(i) >= 0 ? 8 : 6,
    arrow: true,
  }));
  const arrows = w && norm(w) > 1e-9 ? [{ x1: 0, y1: 0, x2: w[0] * 0.6, y2: w[1] * 0.6, color: THEME.text, width: 2.4, label: "w (scaled)" }] : [];
  const extra = useCallback(
    (P) => {
      if (!showLine || !w || norm(w) < 1e-9) return null;
      const d = [-w[1] / norm(w), w[0] / norm(w)];
      const a = P(d[0] * -6, d[1] * -6);
      const b = P(d[0] * 6, d[1] * 6);
      return <line x1={a[0]} y1={a[1]} x2={b[0]} y2={b[1]} stroke={THEME.text} strokeDasharray="6 5" strokeWidth={1.2} opacity={0.8} />;
    },
    [w, showLine]
  );
  return <WordPlane xr={XR} yr={YR} points={pts} arrows={arrows} onDrag={onDrag} extra={extra} ariaLabel="Dense 2-D word vectors" axisLabels={["dim 1", "dim 2"]} />;
}

export default function OneHotLesson({ initialScreen }) {
  const [screen, setScreen] = useState(initialScreen || 0);
  const [a, setA] = useState(0);
  const [b, setB] = useState(1);
  const [nextA, setNextA] = useState(false);
  const [rep, setRep] = useState("onehot");
  const [bask, setBask] = useState(WORDS[BASK].e);
  const [V, setV] = useState(20000);
  const [d, setD] = useState(300);
  const [labels, setLabels] = useState([]);
  const [repC, setRepC] = useState("dense");

  const pick = useCallback(
    (i) => {
      if (nextA) setA(i);
      else setB(i);
      setNextA((v) => !v);
    },
    [nextA]
  );
  const onDrag = useCallback((id, x, y) => setBask([x, y]), []);
  const toggleLabel = useCallback((i) => setLabels((L) => (L.indexOf(i) >= 0 ? L.filter((v) => v !== i) : L.concat([i]))), []);

  const vecs = useMemo(() => WORDS.map((w, i) => (i === BASK ? bask : w.e)), [bask]);
  const dense0 = useMemo(() => WORDS.map((w) => w.e), []);
  const hot = useMemo(() => WORDS.map((_, i) => oneHot(i)), []);
  const wFoot = vecs[0];

  // puzzle readouts
  const xa = hot[a];
  const xb = hot[b];

  let question, stage, side;
  if (screen === 0) {
    question = (
      <>
        <b>Where we are:</b> a model needs numbers, but text is symbols (slide 7). The simplest choice is one-hot: a vocabulary of 20,000 words, and each word gets a vector with a single 1 in its own coordinate (slide 8). <b>The puzzle:</b> a classifier was trained on sentences with <Fa>فوتبال</Fa>, and its weight vector w learned "this coordinate means sports". Then a new text mentions <Fa>بسکتبال</Fa> (slide 10). Click two rows to compare any pair of words. Can you find two different words whose dot product is not 0?
      </>
    );
    stage = <OneHotTable a={a} b={b} onPick={pick} />;
    side = (
      <>
        <Panel>
          <Label>selected pair (click rows)</Label>
          <Readout
            rows={[
              ["word A (orange)", WORDS[a].fa + " · " + WORDS[a].en],
              ["word B (teal)", WORDS[b].fa + " · " + WORDS[b].en],
              ["x_A · x_B", fmt(dot(xa, xb), 0), a === b ? THEME.text : THEME.highlight],
              ["‖x_A − x_B‖", fmt(norm(vsub(xa, xb)), 3)],
            ]}
          />
        </Panel>
        <Panel>
          <Label>classifier trained on فوتبال only</Label>
          <Readout
            rows={[
              ["w", "1 at #1, 0 elsewhere"],
              ["score w·x for " + WORDS[a].en, fmt(dot(hot[0], xa), 0)],
              ["score w·x for " + WORDS[b].en, fmt(dot(hot[0], xb), 0)],
            ]}
          />
        </Panel>
        <Note>Try فوتبال against بسکتبال, then فوتبال against سیاست. Compare the two distances.</Note>
      </>
    );
  } else if (screen === 1) {
    const sc = rep === "onehot" ? hot.map((x) => dot(hot[0], x)) : vecs.map((x) => dot(wFoot, x));
    const cosT = cosine(wFoot, bask);
    question = (
      <>
        <b>Claim to test:</b> if word vectors already know which words are related, evidence learned for one word reaches the others (slide 11). The classifier is trained on <Fa>فوتبال</Fa> alone, so w points along <Fa>فوتبال</Fa>'s vector and the score of any word is w·x. Switch between one-hot and dense pretrained vectors and read the scores. Then drag <Fa>بسکتبال</Fa> (dashed ring). What sets its score: its length, its angle to <Fa>فوتبال</Fa>, or both?
      </>
    );
    stage = (
      <>
        {rep === "onehot" ? <OneHotTable a={0} b={BASK} onPick={pick} /> : <Plane vecs={vecs} w={wFoot} onDrag={onDrag} dragId={BASK} />}
        <Bars rows={WORDS.map((w, i) => ({ label: w.fa, fa: true, value: sc[i], color: topicColor(w.t), mark: w.t > 0 ? "●" : "■" }))} range={rep === "onehot" ? [-1, 1] : [-4, 7]} labelW={90} ariaLabel="score w·x for each word" />
      </>
    );
    side = (
      <>
        <Panel>
          <Label>representation</Label>
          <Pills
            options={[
              ["onehot", "one-hot (20,000-d)"],
              ["dense", "dense, pretrained (2-d)"],
            ]}
            value={rep}
            onPick={setRep}
          />
          {rep === "dense" ? (
            <Readout
              rows={[
                ["‖w‖ (= ‖x_فوتبال‖)", fmt(norm(wFoot), 2)],
                ["‖x_بسکتبال‖", fmt(norm(bask), 2)],
                ["cos θ between them", fmt(cosT, 2)],
                ["score = ‖w‖‖x‖cos θ", fmt(dot(wFoot, bask), 2)],
              ]}
            />
          ) : (
            <Readout
              rows={[
                ["score for فوتبال", "1"],
                ["score for every other word", "0"],
              ]}
            />
          )}
        </Panel>
        <Note>Bars: score w·x (● sports, ■ politics). With one-hot, the model needs separate evidence for every coordinate (slide 10). With dense vectors, the evidence spreads to every word that points the same way.</Note>
      </>
    );
  } else if (screen === 2) {
    question = (
      <>
        <b>What you found:</b> distinct one-hot vectors are orthogonal and all the same distance apart, so a model learns nothing about a word it has never seen in training. A dense embedding gives related words vectors that point the same way, so the score w·x = ‖w‖‖x‖cos θ carries evidence over to them. That only helps if the vectors are learned <i>before</i> the task, from a large corpus (slide 11). Change V and d to compare the sizes.
      </>
    );
    stage = (
      <>
        <Plane vecs={vecs} w={wFoot} onDrag={onDrag} dragId={BASK} />
      </>
    );
    side = (
      <>
        <Panel>
          <Slider id="l1-v" label="vocabulary size V" value={V} min={1000} max={100000} stepSize={1000} onChange={setV} display={V.toLocaleString("en-US")} />
          <Slider id="l1-d" label="embedding size d" value={d} min={2} max={500} stepSize={1} onChange={setD} display={String(d)} />
          <Eq size={12.5}>
            one-hot: x ∈ ℝ^V, one 1, V − 1 zeros
            <br />
            i ≠ j ⇒ xᵢ·xⱼ = 0, ‖xᵢ − xⱼ‖ = √2
            <br />
            embedding table E ∈ ℝ^(V×d), e = Eᵀx (row lookup)
            <br />
            score = w·e = ‖w‖‖e‖cos θ
          </Eq>
          <Readout
            rows={[
              ["numbers per word, one-hot", V.toLocaleString("en-US") + " (1 nonzero)"],
              ["numbers per word, dense", String(d)],
              ["table E holds", (V * d).toLocaleString("en-US") + " numbers"],
              ["weights a linear model needs", (d).toLocaleString("en-US") + " instead of " + V.toLocaleString("en-US")],
            ]}
          />
        </Panel>
        <Note>Slides 8–11. Next lesson: where would vectors that "know" relatedness come from? From the contexts the words appear in.</Note>
      </>
    );
  } else {
    const vecC = repC === "dense" ? dense0 : hot;
    const sc = scores(vecC, labels);
    const ver = sc.map(verdict);
    const right = ver.filter((v, i) => v === WORDS[i].t).length;
    const scD = scores(dense0, labels).map(verdict);
    const scH = scores(hot, labels).map(verdict);
    const okD = scD.every((v, i) => v === WORDS[i].t);
    const rightH = scH.filter((v, i) => v === WORDS[i].t).length;
    const w = labels.length ? vsum(labels.map((j) => vscale(dense0[j], WORDS[j].t)), 2) : null;
    question = (
      <>
        Labels are expensive. Click words to label them with their true topic. The classifier gives each word the score Σ y·(x·x_labelled) over your labels (y = +1 sports, −1 politics), so each label passes its evidence on to words that point its way. Get all 9 words right with dense vectors using at most 2 labels. Then switch to one-hot: how many labels does it need?
      </>
    );
    stage = (
      <>
        {repC === "dense" ? <Plane vecs={dense0} w={w} labels={labels} verdicts={ver} showLine /> : <OneHotTable a={-1} b={-1} onPick={toggleLabel} labels={labels} />}
        <Bars rows={WORDS.map((wd, i) => ({ label: wd.fa, fa: true, value: sc[i], color: topicColor(wd.t), mark: (ver[i] === wd.t ? "✓" : ver[i] === 0 ? "?" : "✕") + (labels.indexOf(i) >= 0 ? " L" : "") }))} range={repC === "dense" ? [-8, 8] : [-1.5, 1.5]} labelW={90} ariaLabel="score of each word from the labels" />
      </>
    );
    side = (
      <>
        <Panel edge={okD && labels.length <= 2 ? THEME.ok : THEME.grid}>
          <Label>label a word (click again to remove)</Label>
          <div style={{ display: "flex", gap: 6, flexWrap: "wrap", marginBottom: 8 }}>
            {WORDS.map((wd, i) => (
              <Chip key={i} value={i} on={labels.indexOf(i) >= 0} onPick={toggleLabel} mark={wd.t > 0 ? "●" : "■"}>
                {wd.fa}
              </Chip>
            ))}
          </div>
          <Pills
            options={[
              ["dense", "dense vectors"],
              ["onehot", "one-hot"],
            ]}
            value={repC}
            onPick={setRepC}
          />
          <Readout
            rows={[
              ["labels used", String(labels.length)],
              ["correct in this view", right + " / " + N],
              ["correct with one-hot", rightH + " / " + N],
            ]}
          />
          <Goal met={okD}>dense vectors: all 9 words correct</Goal>
          <Goal met={okD && labels.length <= 2}>… with at most 2 labels</Goal>
          <Hint
            hints={[
              "The dashed line is the boundary: words on w's side score positive. With two labels, w = x_sports − x_politics. Which word sits closest to the line?",
              "المپیک sits between the groups. Pick the pair whose difference vector puts المپیک on the sports side and دولت on the politics side. A long vector like فوتبال pulls w toward itself.",
            ]}
          />
        </Panel>
        <Note>One-hot gives each label to one word only, so it needs all 9 labels. Dense vectors let 2 labels cover 9 words, but only when the vectors point the right way. That is why pretraining matters (slide 11).</Note>
      </>
    );
  }

  return <LessonFrame screen={screen} onScreen={setScreen} title="Every one-hot word is a stranger" question={question} stage={stage} side={side} />;
}
