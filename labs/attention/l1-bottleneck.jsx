import { useState, useMemo, useCallback } from "react";
import { THEME, MONO } from "./common";

// Lesson 1 - One vector can't hold a long sentence (Attention, slides 2-17).
// Aha: an encoder-decoder squeezes the whole source into one fixed-size vector, so the longer the sentence,
// the less of it the decoder can read back. Attention keeps every encoder state and lets each output step
// take its own weighted mix of them, so nothing has to fit in one vector.

// @sim-start
const SHORT = "the book that i bought yesterday was expensive".split(" ");
const LONG = "the book that i bought yesterday from the small shop near our university was surprisingly expensive".split(" ");
const VOCAB = Array.from(new Set(SHORT.concat(LONG).concat("a cheap pen sold today is new old store city".split(" "))));
const DMAX = 320;
const ENCODERS = 24;
const GOAL = 0.95;
// toy encoder: every word and every position gets a random ±1 code; the fixed vector is the sum of
// word ⊙ position codes. The decoder reads position i back by un-binding and picking the closest word.
const CODES = (function () {
  const out = [];
  for (let s = 0; s < ENCODERS; s++) {
    const r = mulberry32(77 + s);
    const W = {};
    VOCAB.forEach((w) => {
      const v = new Int8Array(DMAX);
      for (let k = 0; k < DMAX; k++) v[k] = r() < 0.5 ? -1 : 1;
      W[w] = v;
    });
    const P = [];
    for (let i = 0; i < 20; i++) {
      const v = new Int8Array(DMAX);
      for (let k = 0; k < DMAX; k++) v[k] = r() < 0.5 ? -1 : 1;
      P.push(v);
    }
    out.push({ W, P });
  }
  return out;
})();
function readBack(sent, d, e) {
  const s = new Float64Array(d);
  sent.forEach((w, i) => {
    const a = e.W[w];
    const p = e.P[i];
    for (let k = 0; k < d; k++) s[k] += p[k] * a[k];
  });
  return sent.map((w, i) => {
    const p = e.P[i];
    let best = null;
    let bs = -Infinity;
    for (const u of VOCAB) {
      const a = e.W[u];
      let dd = 0;
      for (let k = 0; k < d; k++) dd += s[k] * p[k] * a[k];
      if (dd > bs) {
        bs = dd;
        best = u;
      }
    }
    return best === w;
  });
}
// fraction of the 24 random encoders that recover each word, and the overall recall
function recall(sent, d) {
  const per = sent.map(() => 0);
  for (const e of CODES) readBack(sent, d, e).forEach((ok, i) => (per[i] += ok ? 1 : 0));
  const perW = per.map((c) => c / CODES.length);
  return { perW, avg: mean(perW) };
}
// @sim-end

const SRC = ["I", "drink", "tea"];
const TGT = ["من", "چای", "می‌نوشم"];
const ALIGN_SRC = ["I", "read", "my", "books"];
const ALIGN_TGT = ["کتاب‌هایم", "می‌خوانم"];

function WordRecall({ sent, perW }) {
  return (
    <div style={{ display: "flex", flexWrap: "wrap", gap: 5, margin: "4px 0 8px" }}>
      {sent.map((w, i) => {
        const p = perW[i];
        return (
          <span key={i} style={{ fontFamily: MONO, fontSize: 13.5, padding: "2px 8px", borderRadius: 4, border: "1px solid " + (p >= 0.9 ? THEME.ok : p >= 0.5 ? THEME.grid : THEME.warn), background: rgba(p >= 0.5 ? THEME.ok : THEME.warn, 0.08 + 0.25 * (p >= 0.5 ? p : 1 - p)), color: THEME.text }}>
            {w}
            <span style={{ color: THEME.muted, fontSize: 11 }}> {Math.round(p * 100)}%</span>
            {p >= 0.9 ? " ✓" : p < 0.5 ? " ✕" : ""}
          </span>
        );
      })}
    </div>
  );
}
function normRow(r) {
  const s = r.reduce((a, b) => a + b, 0);
  return s > 0 ? r.map((v) => v / s) : r.map(() => 1 / r.length);
}

export default function BottleneckLesson({ initialScreen }) {
  const [screen, setScreen] = useState(initialScreen || 0);
  const [d, setD] = useState(64);
  const [which, setWhich] = useState("short");
  const [step, setStep] = useState(1);
  const [raw, setRaw] = useState([
    [1, 0.2, 0.1],
    [0.1, 0.2, 1],
    [0.1, 1, 0.2],
  ]);
  const [cells, setCells] = useState([
    [false, false, false, true],
    [false, true, false, false],
  ]);
  const [ds, setDs] = useState(48);
  const [dl, setDl] = useState(48);

  const setW = useCallback((j, val) => setRaw((R) => R.map((row, t) => (t === step ? row.map((v, k) => (k === j ? val : v)) : row))), [step]);
  const setW0 = useCallback((v) => setW(0, v), [setW]);
  const setW1 = useCallback((v) => setW(1, v), [setW]);
  const setW2 = useCallback((v) => setW(2, v), [setW]);
  const flipCell = useCallback((i, j) => setCells((C) => C.map((row, a) => (a === i ? row.map((v, b) => (b === j ? !v : v)) : row))), []);

  const sent = which === "short" ? SHORT : LONG;
  const rec = useMemo(() => recall(sent, d), [sent, d]);
  const recS = useMemo(() => recall(SHORT, ds), [ds]);
  const recL = useMemo(() => recall(LONG, dl), [dl]);
  const recSprev = useMemo(() => (ds > 16 ? recall(SHORT, ds - 16) : null), [ds]);
  const recLprev = useMemo(() => (dl > 16 ? recall(LONG, dl - 16) : null), [dl]);

  let question, stage, side;
  if (screen === 0) {
    question = (
      <>
        <b>Where we are:</b> the deck opens with a test. Read a sentence, let it disappear, then translate it from memory (slides 2–6). A long sentence is much harder. An RNN encoder–decoder has the same problem: the encoder reads the source and hands the decoder one final state of fixed size (slides 12–13). <b>The puzzle:</b> here a d-number vector stores the sentence, and the decoder tries to read every word back from it. Switch between the two sentences from the slides. Why does the long one lose more words when d stays the same?
      </>
    );
    stage = (
      <>
        <Panel>
          <Label>source sentence · how often each word is read back correctly</Label>
          <WordRecall sent={sent} perW={rec.perW} />
          <Bars rows={[{ label: "words read back", value: rec.avg, color: rec.avg >= GOAL ? THEME.ok : THEME.accent }]} range={[0, 1]} valueFmt={(v) => Math.round(v * 100) + "%"} labelW={120} />
        </Panel>
        <Note>✓ read back by at least 90% of the 24 random encoders · ✕ by fewer than half.</Note>
      </>
    );
    side = (
      <>
        <Panel>
          <Pills
            options={[
              ["short", "short (8 words)"],
              ["long", "long (16 words)"],
            ]}
            value={which}
            onPick={setWhich}
          />
          <Slider id="a1-d" label="size of the fixed vector, d" value={d} min={16} max={320} stepSize={16} onChange={setD} display={"d = " + d} />
          <Readout
            rows={[
              ["numbers in the fixed vector", String(d)],
              ["numbers per word", fmt(d / sent.length, 1)],
              ["words read back", Math.round(rec.avg * 100) + "%"],
            ]}
          />
        </Panel>
        <Note>A toy encoder, not a trained RNN: each word is stored as a random code tied to its position, and all codes are added into one vector. A trained network packs better, but it faces the same limit: d numbers have to cover the whole sentence.</Note>
      </>
    );
  } else if (screen === 1) {
    const W = raw.map(normRow);
    const mean3 = [1 / 3, 1 / 3, 1 / 3];
    question = (
      <>
        <b>Claim to test:</b> instead of one final state, keep all the encoder states h₁, h₂, h₃ and give each decoder step its own summary. Averaging them gives every step the same blurred mix (slide 14). A <i>weighted</i> sum lets each step lean on the source words it needs (slides 15–17). Pick a decoder step and set the weights. What does the average give <Fa>چای</Fa> that the weighted sum does not?
      </>
    );
    stage = (
      <>
        <Matrix
          rowLabels={TGT}
          colLabels={SRC}
          values={W}
          shade={(v) => v}
          text={(v) => fmt(v, 2)}
          hiRow={step}
          faRows
          cell={58}
          rowW={90}
          ariaLabel="weights each decoder step puts on each encoder state"
        />
        <Label>what step «{TGT[step]}» receives</Label>
        <Bars
          rows={SRC.map((s, j) => ({ label: "from h" + (j + 1) + " " + s, value: W[step][j], color: THEME.accent, mark: "weighted" })).concat(SRC.map((s, j) => ({ label: "from h" + (j + 1) + " " + s, value: mean3[j], color: THEME.muted, mark: "average" })))}
          range={[0, 1]}
          labelW={130}
          rowH={21}
          ariaLabel="share of each encoder state in the context: weighted vs average"
        />
      </>
    );
    side = (
      <>
        <Panel>
          <Label>decoder step</Label>
          <Pills options={TGT.map((t, i) => [i, t])} value={step} onPick={setStep} />
          <Slider id="a1-w0" label="raw weight on h₁ (I)" value={raw[step][0]} min={0} max={1} stepSize={0.01} onChange={setW0} />
          <Slider id="a1-w1" label="raw weight on h₂ (drink)" value={raw[step][1]} min={0} max={1} stepSize={0.01} onChange={setW1} />
          <Slider id="a1-w2" label="raw weight on h₃ (tea)" value={raw[step][2]} min={0} max={1} stepSize={0.01} onChange={setW2} />
          <Note>Weights are rescaled to sum to 1, so the context is an average that can lean.</Note>
        </Panel>
        <Note>Persian puts the verb last: من چای می‌نوشم. So the second step needs tea and the third needs drink. Each step needs a different mix, which one average cannot give (slide 15).</Note>
      </>
    );
  } else if (screen === 2) {
    const W = cells.map((row) => {
      const k = row.filter(Boolean).length;
      return row.map((v) => (v ? 1 / k : 0));
    });
    question = (
      <>
        <b>What you found:</b> a fixed vector has to hold everything, so a longer source leaves less room per word. Averaging all the states gives every output step the same blurred mix. A weighted sum c_t = Σⱼ α_t,j hⱼ, with fresh weights at every step t, lets each output use the source words it needs. Alignment is not one-to-one either (slide 16): click cells to choose which source words each Persian word draws from.
      </>
    );
    stage = (
      <>
        <Matrix rowLabels={ALIGN_TGT} colLabels={ALIGN_SRC} values={W} shade={(v) => v} text={(v) => (v ? fmt(v, 2) : "·")} onCell={flipCell} faRows cell={58} rowW={100} ariaLabel="click to choose which source words feed each target word" />
        <Eq size={12.5}>
          fixed: c = h_n for every step
          <br />
          average: c = (1/n) Σⱼ hⱼ for every step
          <br />
          weighted: c_t = Σⱼ α_t,j hⱼ, α_t,j ≥ 0, Σⱼ α_t,j = 1, new weights at each step t
        </Eq>
      </>
    );
    side = (
      <>
        <Panel>
          <Label>I read my books → کتاب‌هایم را می‌خوانم</Label>
          <Readout
            rows={[
              ["کتاب‌هایم draws from", ALIGN_SRC.filter((s, j) => cells[0][j]).join(" + ") || "nothing"],
              ["می‌خوانم draws from", ALIGN_SRC.filter((s, j) => cells[1][j]).join(" + ") || "nothing"],
            ]}
          />
        </Panel>
        <Note>کتاب‌هایم (my books) combines "my" and "books", and می‌خوانم (I read) combines "I" and "read", in a different order from English. Slides 13–17. Next lesson: how does the model compute those weights instead of us setting them?</Note>
      </>
    );
  } else {
    const okS = recS.avg >= GOAL;
    const okL = recL.avg >= GOAL;
    const minS = okS && (!recSprev || recSprev.avg < GOAL);
    const minL = okL && (!recLprev || recLprev.avg < GOAL);
    question = (
      <>
        How big must the fixed vector be? For each sentence, find the <i>smallest</i> d (in steps of 16) at which at least {Math.round(GOAL * 100)}% of the words are read back. Then compare: how does the size you need grow with sentence length, and what does attention store instead?
      </>
    );
    stage = (
      <>
        <Panel>
          <Label>short sentence (8 words), d = {ds}</Label>
          <WordRecall sent={SHORT} perW={recS.perW} />
          <Label>long sentence (16 words), d = {dl}</Label>
          <WordRecall sent={LONG} perW={recL.perW} />
        </Panel>
        <Bars
          rows={[
            { label: "short, read back", value: recS.avg, color: okS ? THEME.ok : THEME.accent },
            { label: "long, read back", value: recL.avg, color: okL ? THEME.ok : THEME.accent },
          ]}
          range={[0, 1]}
          valueFmt={(v) => Math.round(v * 100) + "%"}
          labelW={130}
        />
      </>
    );
    side = (
      <>
        <Panel edge={minS && minL ? THEME.ok : THEME.grid}>
          <Slider id="a1-ds" label="d for the short sentence" value={ds} min={16} max={320} stepSize={16} onChange={setDs} display={"d = " + ds} />
          <Slider id="a1-dl" label="d for the long sentence" value={dl} min={16} max={320} stepSize={16} onChange={setDl} display={"d = " + dl} />
          <Goal met={minS}>short: smallest d with ≥ {Math.round(GOAL * 100)}%</Goal>
          <Goal met={minL}>long: smallest d with ≥ {Math.round(GOAL * 100)}%</Goal>
          {minS && minL ? (
            <Readout
              rows={[
                ["length ratio", "16 / 8 = 2"],
                ["size ratio you found", fmt(dl / ds, 2)],
                ["attention stores instead", "one d-vector per word, e.g. 16 × d"],
              ]}
            />
          ) : null}
          <Hint
            hints={[
              "Slide each d up until the bar passes 95%, then step back once to check that the size below falls short.",
              "The short sentence needs about 110 numbers and the long one about 210. Check the multiples of 16 around those.",
            ]}
          />
        </Panel>
        <Note>Results are averaged over 24 random encoders, so the curves are smooth. A fixed vector must grow with the longest sentence you expect. Attention keeps every state, so its memory grows with each sentence automatically.</Note>
      </>
    );
  }

  return <LessonFrame screen={screen} onScreen={setScreen} title="One vector can't hold a long sentence" question={question} stage={stage} side={side} />;
}
