import { useState, useMemo, useCallback, useRef, useEffect } from "react";
import { THEME, MONO } from "./common";

// Lesson 3 - Predict the neighbours, get the meaning (Word Embedding, slides 14-19, 25).
// Aha: Skip-gram never sees a similarity label. It only learns to predict each word's neighbours,
// and two words that must make the same predictions end up with the same input vector, even if they
// never appear in the same sentence.

// @sim-start
const BASE = ["تیم ما فوتبال بازی کرد", "تیم ما بسکتبال بازی کرد", "مجلس قانون جدید تصویب کرد", "دولت قانون جدید تصویب کرد"];
const VOC = ["تیم", "ما", "فوتبال", "بسکتبال", "بازی", "کرد", "مجلس", "دولت", "قانون", "جدید", "تصویب", "والیبال"];
const NEWW = 11;
const SEED = 3;
const LR = 0.08;
const EPOCHS = 250;
const SNAP = 5;
const POOL = ["تیم ما والیبال بازی کرد", "والیبال قانون جدید تصویب کرد", "والیبال بازی کرد", "تیم والیبال بازی کرد", "مجلس والیبال تصویب کرد", "تیم ما والیبال"];
const NEAR = 0.3;

function pairsOf(sents, win, mode) {
  const idx = {};
  VOC.forEach((w, i) => (idx[w] = i));
  const out = [];
  for (const s of sents) {
    const t = s.split(" ");
    t.forEach((w, i) => {
      const ctx = [];
      for (let j = Math.max(0, i - win); j <= Math.min(t.length - 1, i + win); j++) if (j !== i) ctx.push(idx[t[j]]);
      if (mode === "cbow") out.push({ ctx, c: idx[w] });
      else for (const o of ctx) out.push({ c: idx[w], o });
    });
  }
  return out;
}
// full-softmax Skip-gram on 2-D vectors, plain SGD; returns snapshots of the input vectors v
function trainSG(sents, win) {
  const rng = mulberry32(SEED);
  const v = VOC.map(() => [randn(rng) * 0.6, randn(rng) * 0.6]);
  const u = VOC.map(() => [randn(rng) * 0.6, randn(rng) * 0.6]);
  const pairs = pairsOf(sents, win, "sg");
  const snaps = [{ v: v.map((x) => x.slice()), loss: NaN }];
  for (let e = 1; e <= EPOCHS; e++) {
    let loss = 0;
    for (const p of pairs) {
      const vc = v[p.c];
      const s = u.map((x) => x[0] * vc[0] + x[1] * vc[1]);
      const pr = softmax(s);
      loss -= Math.log(pr[p.o]);
      const g = [0, 0];
      for (let j = 0; j < VOC.length; j++) {
        const d = pr[j] - (j === p.o ? 1 : 0);
        g[0] += d * u[j][0];
        g[1] += d * u[j][1];
        u[j][0] -= LR * d * vc[0];
        u[j][1] -= LR * d * vc[1];
      }
      vc[0] -= LR * g[0];
      vc[1] -= LR * g[1];
    }
    if (e % SNAP === 0) snaps.push({ v: v.map((x) => x.slice()), loss: loss / pairs.length });
  }
  return { snaps, u, v, pairs: pairs.length };
}
function dist(a, b) {
  return Math.hypot(a[0] - b[0], a[1] - b[1]);
}
function nearestTo(v, i, skip) {
  return VOC.map((w, j) => [j, dist(v[i], v[j])])
    .filter((p) => p[0] !== i && skip.indexOf(p[0]) < 0)
    .sort((a, b) => a[1] - b[1]);
}
// @sim-end

const GROUP = [0, 0, 1, 1, 0, 2, 3, 3, 4, 4, 4, 1];
function colorOf(i) {
  const g = GROUP[i];
  return g === 1 ? THEME.accent : g === 3 ? THEME.highlight : THEME.muted;
}
function shapeOf(i) {
  const g = GROUP[i];
  return g === 1 ? "circle" : g === 3 ? "square" : "diamond";
}
function plotPoints(v, opts) {
  const o = opts || {};
  return VOC.map((w, i) => ({
    id: String(i),
    label: w,
    fa: true,
    x: v[i][0],
    y: v[i][1],
    color: i === o.focus ? THEME.text : colorOf(i),
    shape: shapeOf(i),
    r: i === o.focus || i === o.guess ? 8 : 6,
    hollow: i === o.guess,
    dim: o.hideNew && i === NEWW,
  })).filter((p, i) => !(o.hideNew && i === NEWW));
}
function rangeOf(snaps) {
  let m = 1;
  for (const s of snaps) for (const p of s.v) m = Math.max(m, Math.abs(p[0]), Math.abs(p[1]));
  const r = Math.ceil(m * 1.12 * 2) / 2;
  return [-r, r];
}

export default function SkipGramLesson({ initialScreen }) {
  const [screen, setScreen] = useState(initialScreen || 0);
  const [guess, setGuess] = useState(-1);
  const [frame, setFrame] = useState(0);
  const [running, setRunning] = useState(false);
  const [win, setWin] = useState(2);
  const [mode, setMode] = useState("sg");
  const [sent, setSent] = useState(0);
  const [center, setCenter] = useState(2);
  const [ctxw, setCtxw] = useState(4);
  const [pick, setPick] = useState(-1);
  const [winC, setWinC] = useState(1);
  const raf = useRef(0);
  const togglePick = useCallback((j) => setPick((p) => (p === j ? -1 : j)), []);

  const base = useMemo(() => trainSG(BASE, 2), []);
  const baseW = useMemo(() => (win === 2 ? base : trainSG(BASE, win)), [win, base]);
  const chall = useMemo(() => trainSG(pick >= 0 ? BASE.concat([POOL[pick]]) : BASE, winC), [pick, winC]);
  const R = useMemo(() => rangeOf(base.snaps), [base]);
  const last = base.snaps.length - 1;

  const play = useCallback(() => {
    if (REDUCED) {
      setFrame(last);
      return;
    }
    setFrame(0);
    setRunning(true);
  }, [last]);
  useEffect(() => {
    if (!running) return undefined;
    let t0 = null;
    const tick = (t) => {
      if (t0 === null) t0 = t;
      const f = Math.min(last, Math.floor(((t - t0) / 3000) * last));
      setFrame(f);
      if (f < last) raf.current = requestAnimationFrame(tick);
      else setRunning(false);
    };
    raf.current = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf.current);
  }, [running, last]);
  const reset = useCallback(() => {
    setRunning(false);
    setFrame(0);
  }, []);

  let question, stage, side;
  if (screen === 0) {
    const snap = base.snaps[frame];
    const nn = nearestTo(snap.v, 2, [NEWW]);
    const done = frame === last;
    question = (
      <>
        <b>Where we are:</b> lesson 2 built word vectors by counting contexts. Word2Vec (Mikolov et al., 2013) learns them instead, from a prediction task the text provides for free (slides 12–16). Skip-gram uses each word to predict the words around it. <b>The puzzle:</b> the corpus is the four sentences below, and the vectors start random. Before training, click the word you think will end up <i>closest</i> to <Fa>فوتبال</Fa>. Then press Train.
      </>
    );
    stage = (
      <>
        <WordPlane xr={R} yr={R} points={plotPoints(snap.v, { focus: 2, guess, hideNew: true })} ariaLabel="Input vectors v of each word during training" ticks={1} />
        <div dir="rtl" lang="fa" style={{ fontFamily: FA, fontSize: 15, color: THEME.muted, marginTop: 6, lineHeight: 1.7 }}>
          {BASE.map((s, i) => (
            <div key={i}>{s}</div>
          ))}
        </div>
      </>
    );
    side = (
      <>
        <Panel>
          <Label>your prediction: closest to فوتبال</Label>
          <div dir="rtl" style={{ display: "flex", flexWrap: "wrap", gap: 5, marginBottom: 8 }}>
            {VOC.map((w, i) =>
              i === 2 || i === NEWW ? null : (
                <Chip key={i} value={i} on={guess === i} onPick={setGuess}>
                  {w}
                </Chip>
              )
            )}
          </div>
          <div style={{ display: "flex", gap: 6, marginBottom: 8 }}>
            <Btn onClick={play} disabled={guess < 0 || running} strong>
              ▶ Train {EPOCHS} epochs
            </Btn>
            <Btn onClick={reset} disabled={frame === 0}>
              reset
            </Btn>
          </div>
          <Readout
            rows={[
              ["epoch", String(frame * SNAP)],
              ["average loss −log P(o|c)", isFinite(snap.loss) ? fmt(snap.loss, 3) : "—"],
              ["closest to فوتبال now", VOC[nn[0][0]] + " (" + fmt(nn[0][1], 2) + ")"],
              ["your guess", guess >= 0 ? VOC[guess] + " (" + fmt(dist(snap.v[2], snap.v[guess]), 2) + ")" : "—"],
            ]}
          />
          {done && guess >= 0 ? <Goal met={nn[0][0] === guess}>{nn[0][0] === guess ? "your guess ended closest" : "the closest word is " + VOC[nn[0][0]] + ", not " + VOC[guess]}</Goal> : null}
        </Panel>
        <Note>● sports words · ■ politics words · ◆ the rest. Distances are read straight off the plane. Why does the winner end up there, when it never appears in a sentence with فوتبال?</Note>
      </>
    );
  } else if (screen === 1) {
    const prs = pairsOf([BASE[sent]], win, mode);
    const v = baseW.v;
    const sc = baseW.u.map((x) => dot(x, v[center]));
    const P = softmax(sc);
    question = (
      <>
        <b>Claim to test:</b> each word gets an input vector v, and every word also has an output vector u. Skip-gram scores a candidate neighbour o with u_o·v_c and turns all the scores into probabilities with a softmax (slides 17–19). Pick a center word and read which neighbours it predicts. Then compare <Fa>فوتبال</Fa> with <Fa>بسکتبال</Fa>: why must their v vectors end up in the same place? Change the window to see the training pairs change.
      </>
    );
    stage = (
      <>
        <WordPlane xr={R} yr={R} points={plotPoints(v, { focus: center, hideNew: true })} ariaLabel="Trained input vectors" ticks={1} />
        <Label>P(o | c = «{VOC[center]}») over the vocabulary</Label>
        <Bars rows={VOC.slice(0, NEWW).map((w, j) => ({ label: w, fa: true, value: P[j], color: colorOf(j) }))} range={[0, 0.5]} labelW={70} rowH={21} ariaLabel="predicted context probabilities" />
      </>
    );
    side = (
      <>
        <Panel>
          <Label>center word c</Label>
          <div dir="rtl" style={{ display: "flex", flexWrap: "wrap", gap: 5, marginBottom: 8 }}>
            {VOC.slice(0, NEWW).map((w, i) => (
              <Chip key={i} value={i} on={center === i} onPick={setCenter}>
                {w}
              </Chip>
            ))}
          </div>
          <Label>window (retrains the vectors)</Label>
          <Pills
            options={[
              [1, "±1"],
              [2, "±2"],
            ]}
            value={win}
            onPick={setWin}
          />
        </Panel>
        <Panel>
          <Label>training pairs from one sentence (slide 16)</Label>
          <Pills
            options={[
              [0, "s1"],
              [1, "s2"],
              [2, "s3"],
              [3, "s4"],
            ]}
            value={sent}
            onPick={setSent}
          />
          <Pills
            options={[
              ["sg", "Skip-gram: center → context"],
              ["cbow", "CBOW: context → center"],
            ]}
            value={mode}
            onPick={setMode}
          />
          <div lang="fa" dir="rtl" style={{ fontFamily: FA, fontSize: 14, lineHeight: 1.8, maxHeight: 150, overflowY: "auto" }}>
            {prs.map((p, i) => (
              <div key={i}>{mode === "sg" ? VOC[p.c] + " ← " + VOC[p.o] : "{" + p.ctx.map((j) => VOC[j]).join("، ") + "} ← " + VOC[p.c]}</div>
            ))}
          </div>
          <Note>Each line reads: input ← word to predict.</Note>
        </Panel>
      </>
    );
  } else if (screen === 2) {
    const v = base.v;
    const sc = base.u.map((x) => dot(x, v[center]));
    const P = softmax(sc);
    question = (
      <>
        <b>What you found:</b> <Fa>فوتبال</Fa> and <Fa>بسکتبال</Fa> never appear in the same sentence, but they must predict the same neighbours (<Fa>تیم</Fa>, <Fa>ما</Fa>, <Fa>بازی</Fa>, <Fa>کرد</Fa>). The cheapest way to make the same predictions is the same input vector, so training pulls them together. Pick a center c and a context o to see one term of the loss.
      </>
    );
    stage = (
      <>
        <WordPlane xr={R} yr={R} points={plotPoints(v, { focus: center, guess: ctxw, hideNew: true })} ariaLabel="Trained input vectors" ticks={1} />
        <Eq size={12.5}>
          Skip-gram: P(o | c) = exp(u_o·v_c) / Σⱼ exp(uⱼ·v_c), L = −Σ log P(o | c)
          <br />
          CBOW: h = (1/|C|) Σ v_c over the context, P(w | C) = exp(u_w·h) / Σⱼ exp(uⱼ·h)
          <br />
          this pair: u_o·v_c = {fmt(sc[ctxw], 2)}, P(o | c) = {fmt(P[ctxw], 3)}, −log P = {fmt(-Math.log(P[ctxw]), 3)}
        </Eq>
      </>
    );
    side = (
      <>
        <Panel>
          <Label>center c (filled)</Label>
          <div dir="rtl" style={{ display: "flex", flexWrap: "wrap", gap: 5, marginBottom: 8 }}>
            {VOC.slice(0, NEWW).map((w, i) => (
              <Chip key={i} value={i} on={center === i} onPick={setCenter}>
                {w}
              </Chip>
            ))}
          </div>
          <Label>context o (hollow ring)</Label>
          <div dir="rtl" style={{ display: "flex", flexWrap: "wrap", gap: 5 }}>
            {VOC.slice(0, NEWW).map((w, i) => (
              <Chip key={i} value={i} on={ctxw === i} onPick={setCtxw}>
                {w}
              </Chip>
            ))}
          </div>
        </Panel>
        <Note>
          Slides 17–19. v is the input vector, u the output vector, and j runs over every word in the vocabulary. The plane shows v, the vector we keep as the embedding. Real models use 100–600 dimensions and compare words by cosine. With only 2, most angles bunch together, so we read distances here. Mikolov et al. found Skip-gram better on semantic analogies (55% vs 24%) and CBOW slightly better on syntactic ones (64% vs 59%) (slide 22). Next lesson: that softmax sums over every word. What if the vocabulary has a million?
        </Note>
      </>
    );
  } else {
    const v = chall.v;
    const nv = nearestTo(v, NEWW, []);
    const nm = nearestTo(v, 6, [NEWW]);
    const used = pick >= 0;
    const okNear = used && (nv[0][0] === 2 || nv[0][0] === 3) && nv[0][1] <= NEAR;
    const okPol = nm[0][0] === 7;
    const okShort = used && POOL[pick].split(" ").length <= 3;
    const Rc = rangeOf(chall.snaps);
    question = (
      <>
        <Fa>والیبال</Fa> is in the vocabulary, but no sentence uses it, so it gets no training signal and stays where it started (slide 11). Add <i>one</i> sentence from the list and pick the window, then the model retrains. Goals: <Fa>والیبال</Fa> lands within {NEAR} of <Fa>فوتبال</Fa> or <Fa>بسکتبال</Fa>, <Fa>مجلس</Fa>'s closest word is still <Fa>دولت</Fa>, and your sentence has at most 3 words.
      </>
    );
    stage = (
      <WordPlane
        xr={Rc}
        yr={Rc}
        points={plotPoints(v, { focus: NEWW }).map((p) => (p.id === String(NEWW) ? Object.assign({}, p, { color: THEME.third, shape: "triangle", r: 8 }) : p))}
        ariaLabel="Input vectors after training with your sentence"
        ticks={1}
      />
    );
    side = (
      <>
        <Panel edge={okNear && okPol && okShort ? THEME.ok : THEME.grid}>
          <Label>sentence to add (click again to remove)</Label>
          <div style={{ display: "flex", flexDirection: "column", gap: 5, marginBottom: 8 }}>
            {POOL.map((s, i) => (
              <Chip key={i} value={i} on={pick === i} onPick={togglePick} mark={s.split(" ").length + " words"}>
                {s}
              </Chip>
            ))}
          </div>
          <Label>window</Label>
          <Pills
            options={[
              [1, "±1"],
              [2, "±2"],
            ]}
            value={winC}
            onPick={setWinC}
          />
          <Readout
            rows={[
              ["closest to والیبال", VOC[nv[0][0]] + " (" + fmt(nv[0][1], 2) + ")"],
              ["closest to مجلس", VOC[nm[0][0]] + " (" + fmt(nm[0][1], 2) + ")"],
            ]}
          />
          <Goal met={okNear}>والیبال within {NEAR} of a sports word</Goal>
          <Goal met={okPol}>مجلس still closest to دولت</Goal>
          <Goal met={okShort}>the added sentence has at most 3 words</Goal>
          <Hint
            hints={[
              "Training only moves والیبال through the pairs it is in. Which neighbours does فوتبال have to predict? Your sentence must give والیبال some of the same ones.",
              "A 3-word sentence gives والیبال few neighbours. With window ±1 it predicts only the word next to it; with ±2 it predicts two. Try «والیبال بازی کرد» with both windows.",
            ]}
          />
        </Panel>
        <Note>▲ والیبال · ● sports · ■ politics. This is slide 25's "edit the text and change the window", run on a corpus small enough to retrain every time you click.</Note>
      </>
    );
  }

  return <LessonFrame screen={screen} onScreen={setScreen} title="Predict the neighbours, get the meaning" question={question} stage={stage} side={side} />;
}
