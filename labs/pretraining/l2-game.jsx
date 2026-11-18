import { useState, useCallback } from "react";
import { THEME, MONO } from "./common";

// Lesson 2 - The language-model game (Pretraining, slides 20-30).
// Aha: a language model answers one question, "how likely is each token next, given this prefix?"
// Multiplying those answers scores a whole sequence (chain rule), so word order matters, and [EOS] turns
// the scores into a distribution over complete sentences: a longer sentence can beat a shorter one.
// Any task whose answer is text becomes "continue the prefix".

// @sim-start
const VOCAB = ["the", "a", "cat", "dog", "sleeps", "runs", "[EOS]"];
// toy oracle: next-token probabilities after the last token (the rest of the mass goes to "other" words)
const ROWS = {
  "[BOS]": { the: 0.4, a: 0.3, cat: 0.08, dog: 0.08, sleeps: 0.01, runs: 0.03, "[EOS]": 0 },
  the: { the: 0.01, a: 0.01, cat: 0.2, dog: 0.3, sleeps: 0.02, runs: 0.02, "[EOS]": 0.1 },
  a: { the: 0.01, a: 0.01, cat: 0.25, dog: 0.3, sleeps: 0.02, runs: 0.02, "[EOS]": 0.05 },
  cat: { the: 0.01, a: 0.01, cat: 0.01, dog: 0.02, sleeps: 0.5, runs: 0.15, "[EOS]": 0.02 },
  dog: { the: 0.01, a: 0.01, cat: 0.02, dog: 0.01, sleeps: 0.4, runs: 0.3, "[EOS]": 0.03 },
  sleeps: { the: 0.05, a: 0.05, cat: 0.02, dog: 0.02, sleeps: 0.01, runs: 0.01, "[EOS]": 0.1 },
  runs: { the: 0.05, a: 0.05, cat: 0.01, dog: 0.01, sleeps: 0.01, runs: 0.01, "[EOS]": 0.15 },
};
function nextDist(prefix) {
  const last = prefix.length ? prefix[prefix.length - 1] : "[BOS]";
  const row = Object.assign({}, ROWS[last]);
  // the oracle reads the whole prefix: a lone "the" or "a" is almost never a finished sentence
  if (prefix.length === 1 && (last === "the" || last === "a")) row["[EOS]"] = 0.001;
  return row;
}
function seqSteps(seq) {
  const out = [];
  for (let i = 0; i < seq.length; i++) out.push(nextDist(seq.slice(0, i))[seq[i]]);
  return out;
}
function prod(arr) {
  return arr.reduce((a, b) => a * b, 1);
}
// @sim-end

// Explore: three prompts from the slides, each a small tree of next-token distributions
const TREES = {
  tea: {
    prompt: "She poured hot tea into the",
    dist: {
      "": [["cup", 0.55], ["mug", 0.25], ["glass", 0.1], ["other", 0.1]],
      cup: [[".", 0.6], ["and", 0.25], ["other", 0.15]],
      mug: [[".", 0.55], ["and", 0.3], ["other", 0.15]],
      glass: [[".", 0.5], ["and", 0.3], ["other", 0.2]],
      "cup .": [["[EOS]", 0.9], ["other", 0.1]],
      "mug .": [["[EOS]", 0.9], ["other", 0.1]],
      "glass .": [["[EOS]", 0.9], ["other", 0.1]],
    },
    note: "Slides 23 and 25: cup 55%, mug 25%, glass 10%, other 10% (illustrative). Later steps are made up.",
  },
  sent: {
    prompt: "Task: output positive or negative. Review: I loved this film. Sentiment:",
    dist: {
      "": [["positive", 0.72], ["negative", 0.08], ["other", 0.2]],
      positive: [["[EOS]", 0.85], ["other", 0.15]],
      negative: [["[EOS]", 0.85], ["other", 0.15]],
    },
    note: "Slide 28: positive 0.72, negative 0.08, other tokens 0.20. The class label is just the next token.",
  },
  add: {
    prompt: "12 + 7 =",
    dist: {
      "": [["1", 0.9], ["2", 0.08], ["other", 0.02]],
      1: [["9", 0.85], ["8", 0.1], ["other", 0.05]],
      2: [["0", 0.7], ["1", 0.2], ["other", 0.1]],
      "1 9": [["[EOS]", 0.95], ["other", 0.05]],
      "1 8": [["[EOS]", 0.95], ["other", 0.05]],
      "2 0": [["[EOS]", 0.95], ["other", 0.05]],
      "2 1": [["[EOS]", 0.95], ["other", 0.05]],
    },
    note: "Slide 29: the answer 19 comes out one digit-token at a time. The probabilities are made up, and a wrong path (20) still has some.",
  },
};

function StepRow({ seq, steps }) {
  return (
    <div style={{ display: "flex", flexWrap: "wrap", gap: 6, margin: "6px 0" }}>
      {seq.length === 0 ? <span style={{ fontFamily: MONO, fontSize: 13, color: THEME.muted }}>(empty: click tokens below)</span> : null}
      {seq.map((w, i) => (
        <div key={i} style={{ border: "1px solid " + THEME.grid, borderRadius: 4, padding: "3px 8px", background: THEME.bg, textAlign: "center", minWidth: 54 }}>
          <div style={{ fontFamily: MONO, fontSize: 14, color: w === "[EOS]" ? THEME.highlight : THEME.text }}>{w}</div>
          <div style={{ fontFamily: MONO, fontSize: 11.5, color: THEME.muted }}>{fmt(steps[i], 3)}</div>
        </div>
      ))}
    </div>
  );
}
function TokenButtons({ options, onPick, disabled }) {
  return (
    <div style={{ display: "flex", flexWrap: "wrap", gap: 6, margin: "4px 0 8px" }}>
      {options.map((w) => (
        <Pill key={w} value={w} active={false} onPick={onPick}>
          {disabled ? <span style={{ color: THEME.muted }}>{w}</span> : w}
        </Pill>
      ))}
    </div>
  );
}
function DistBars({ dist, chosen }) {
  const rows = Object.keys(dist)
    .filter((k) => dist[k] > 0)
    .map((k) => ({ label: k, value: dist[k], color: k === chosen ? THEME.highlight : THEME.accent }));
  const rest = 1 - rows.reduce((a, r) => a + r.value, 0);
  if (rest > 1e-9) rows.push({ label: "other", value: rest, color: THEME.grid });
  return <Bars rows={rows} range={[0, 1]} digits={3} labelW={80} rowH={21} ariaLabel="next-token distribution" />;
}

export default function GameLesson({ initialScreen }) {
  const [screen, setScreen] = useState(initialScreen || 0);
  const [seq, setSeq] = useState(["sleeps", "cat", "the", "[EOS]"]);
  const [tried, setTried] = useState([]);
  const [tree, setTree] = useState("tea");
  const [path, setPath] = useState([]);
  const [ps, setPs] = useState([0.4, 0.2, 0.5, 0.1]);
  const [eos, setEos] = useState(true);
  const [cseq, setCseq] = useState(["the", "cat", "[EOS]"]);

  const done = seq.length > 0 && seq[seq.length - 1] === "[EOS]";
  const addTok = useCallback(
    (w) => {
      if (seq.length && seq[seq.length - 1] === "[EOS]") return;
      const n = seq.concat([w]);
      setSeq(n);
      if (w === "[EOS]") setTried((T) => (T.some((t) => t.join(" ") === n.join(" ")) ? T : T.concat([n]).slice(-8)));
    },
    [seq]
  );
  const clear = useCallback(() => setSeq([]), []);
  const undo = useCallback(() => setSeq((S) => S.slice(0, -1)), []);
  const presetA = useCallback(() => {
    setSeq(["the", "cat", "sleeps", "[EOS]"]);
    setTried((T) => (T.some((t) => t.join(" ") === "the cat sleeps [EOS]") ? T : T.concat([["the", "cat", "sleeps", "[EOS]"]])));
  }, []);
  const presetB = useCallback(() => {
    setSeq(["sleeps", "cat", "the", "[EOS]"]);
    setTried((T) => (T.some((t) => t.join(" ") === "sleeps cat the [EOS]") ? T : T.concat([["sleeps", "cat", "the", "[EOS]"]])));
  }, []);
  const pickTree = useCallback((t) => {
    setTree(t);
    setPath([]);
  }, []);
  const pathAdd = useCallback((w) => setPath((P) => (w === "other" || (P.length && P[P.length - 1] === "[EOS]") ? P : P.concat([w]))), []);
  const pathReset = useCallback(() => setPath([]), []);
  const setP = useCallback((k, v) => setPs((P) => P.map((x, i) => (i === k ? v : x))), []);
  const setP0 = useCallback((v) => setP(0, v), [setP]);
  const setP1 = useCallback((v) => setP(1, v), [setP]);
  const setP2 = useCallback((v) => setP(2, v), [setP]);
  const setP3 = useCallback((v) => setP(3, v), [setP]);
  const cAdd = useCallback((w) => setCseq((S) => (S.length && S[S.length - 1] === "[EOS]" ? S : S.length >= 8 ? S : S.concat([w]))), []);
  const cClear = useCallback(() => setCseq([]), []);
  const cUndo = useCallback(() => setCseq((S) => S.slice(0, -1)), []);

  let question, stage, side;
  if (screen === 0) {
    const steps = seqSteps(seq);
    const p = prod(steps);
    question = (
      <>
        <b>Where we are:</b> vision pretrains on labeled images (lesson 1). Text has no labels, so where would training targets come from? The deck's answer is a game: guess the next word (slides 19–21). <b>The puzzle:</b> a toy model tells us, for any prefix, how likely each next token is. Build sentences from the same words <b>the, cat, sleeps</b> in different orders, always ending with [EOS]. Which order does the model prefer, and by how much?
      </>
    );
    stage = (
      <>
        <Panel>
          <Label>your sentence · under each token: P(token | everything before it)</Label>
          <StepRow seq={seq} steps={steps} />
          <TokenButtons options={VOCAB} onPick={addTok} disabled={done} />
          <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
            <Btn onClick={undo}>undo</Btn>
            <Btn onClick={clear}>clear</Btn>
            <Btn onClick={presetA}>the cat sleeps [EOS]</Btn>
            <Btn onClick={presetB}>sleeps cat the [EOS]</Btn>
          </div>
        </Panel>
        <Label>next-token distribution after this prefix</Label>
        <DistBars dist={nextDist(done ? seq.slice(0, -1) : seq)} chosen={done ? "[EOS]" : null} />
      </>
    );
    side = (
      <>
        <Panel>
          <Readout
            rows={[
              ["product of the step probabilities", done ? p.toExponential(2).replace("e-", " × 10^−") : "(end with [EOS])"],
              ["number of steps", String(seq.length)],
            ]}
          />
        </Panel>
        <Panel>
          <Label>sentences you finished</Label>
          {tried.length === 0 ? <Note>none yet</Note> : null}
          {tried.map((t, i) => (
            <div key={i} style={{ fontFamily: MONO, fontSize: 12.5, display: "flex", justifyContent: "space-between", gap: 8 }}>
              <span>{t.join(" ")}</span>
              <span style={{ color: THEME.muted }}>{prod(seqSteps(t)).toExponential(2)}</span>
            </div>
          ))}
          <Goal met={tried.some((t) => t.join(" ") === "the cat sleeps [EOS]") && tried.some((t) => t.join(" ") === "sleeps cat the [EOS]")}>compare both orders from slide 27</Goal>
        </Panel>
        <Note>Toy probabilities (slide 27 is reproduced: 0.4 × 0.2 × 0.5 × 0.1 = 0.004 against 0.01 × 0.02 × 0.01 × 0.1 = 0.0000002).</Note>
      </>
    );
  } else if (screen === 1) {
    const T = TREES[tree];
    const key = path.join(" ");
    const dist = T.dist[key];
    let p = 1;
    const steps = path.map((w, i) => {
      const d = T.dist[path.slice(0, i).join(" ")];
      const v = d.find((x) => x[0] === w)[1];
      p *= v;
      return v;
    });
    question = (
      <>
        <b>Claim to test:</b> one interface, "read a prefix, predict a distribution, choose a token, append, repeat" (slide 30), can express very different tasks. Pick a prompt and generate by clicking tokens. Does the classification task need anything other than the next token? What is the probability of the whole continuation you generated?
      </>
    );
    stage = (
      <>
        <Pills
          options={[
            ["tea", "next word"],
            ["sent", "sentiment as text"],
            ["add", "arithmetic as text"],
          ]}
          value={tree}
          onPick={pickTree}
        />
        <Panel>
          <div style={{ fontFamily: MONO, fontSize: 14, lineHeight: 1.6 }}>
            <span style={{ color: THEME.muted }}>{T.prompt} </span>
            {path.map((w, i) => (
              <span key={i} style={{ color: THEME.highlight }}>
                {w}{" "}
              </span>
            ))}
            <span style={{ color: THEME.accent }}>▌</span>
          </div>
        </Panel>
        {dist ? (
          <>
            <Label>click a token to append it</Label>
            <TokenButtons options={dist.map((d) => d[0]).filter((w) => w !== "other")} onPick={pathAdd} />
            <Bars rows={dist.map((d) => ({ label: d[0], value: d[1], color: d[0] === "other" ? THEME.grid : THEME.accent }))} range={[0, 1]} digits={2} labelW={80} rowH={21} ariaLabel="next-token distribution" />
          </>
        ) : (
          <Note>Generation stopped at [EOS].</Note>
        )}
      </>
    );
    side = (
      <>
        <Panel>
          <Readout
            rows={[
              ["tokens generated", String(path.length)],
              ["step probabilities", steps.length ? steps.map((v) => fmt(v, 2)).join(" × ") : "—"],
              ["probability of this continuation", fmt(p, 4)],
            ]}
          />
          <Btn onClick={pathReset}>start again</Btn>
        </Panel>
        <Note>{T.note}</Note>
        <Note>A likely text is not a true text: an oracle of text likelihood is not an oracle of truth (slide 25).</Note>
      </>
    );
  } else if (screen === 2) {
    const used = eos ? ps : ps.slice(0, 3);
    const P = prod(used);
    const lnP = used.reduce((a, v) => a + Math.log(v), 0);
    question = (
      <>
        <b>What you found:</b> the model only ever answers "what comes next?". The probability of a whole sentence is the product of those answers, each conditioned on its entire prefix, so changing the order changes every factor (slides 24–27). Generation and classification use the same answers (slides 28–30). Move the four step probabilities of slide 26. What does [EOS] add?
      </>
    );
    stage = (
      <>
        <Eq size={14}>
          P(x₁, …, x_T) = P(x₁) · P(x₂ | x₁) · … · P(x_T | x₁…x_(T−1))
          <br />
          = ∏ₜ P(xₜ | x_&lt;t)
          <br />
          {used.map((v) => fmt(v, 2)).join(" × ")} = {fmt(P, 4)} · ln P = {fmt(lnP, 3)}
        </Eq>
        <Bars
          rows={["first token", "second token", "third token", "end token [EOS]"].map((l, i) => ({ label: l, value: ps[i], color: i === 3 && !eos ? THEME.grid : THEME.accent, mark: i === 3 && !eos ? "(dropped)" : "" }))}
          range={[0, 1]}
          labelW={120}
          ariaLabel="four step probabilities"
        />
      </>
    );
    side = (
      <>
        <Panel>
          <Slider id="g2-p0" label="P(first token)" value={ps[0]} min={0.01} max={1} stepSize={0.01} onChange={setP0} />
          <Slider id="g2-p1" label="P(second | first)" value={ps[1]} min={0.01} max={1} stepSize={0.01} onChange={setP1} />
          <Slider id="g2-p2" label="P(third | first two)" value={ps[2]} min={0.01} max={1} stepSize={0.01} onChange={setP2} />
          <Slider id="g2-p3" label="P([EOS] | all three)" value={ps[3]} min={0.01} max={1} stepSize={0.01} onChange={setP3} />
          <Toggle label="include [EOS]" value={eos} onChange={setEos} />
        </Panel>
        <Note>Without [EOS] we score a prefix, not a sentence: every extra factor is ≤ 1, so "the cat" would always beat "the cat sleeps". With [EOS], the model must also say that the sentence is finished, and the scores of all complete sentences add up to 1. In practice x is a tokenizer token and V its vocabulary (slide 24).</Note>
      </>
    );
  } else {
    const steps = seqSteps(cseq);
    const fin = cseq.length > 0 && cseq[cseq.length - 1] === "[EOS]";
    const P = fin ? prod(steps) : 0;
    const words = fin ? cseq.length - 1 : cseq.length;
    const ref = prod(seqSteps(["the", "cat", "[EOS]"]));
    const g1 = fin && words >= 3;
    const g2 = g1 && P > ref;
    const g3 = g1 && P > 0.004;
    question = (
      <>
        The toy model rates the two-word sentence "the cat [EOS]" at {fmt(ref, 4)}. Every extra word multiplies in another factor below 1. Can a <b>longer</b> sentence still be more probable? Build one with at least 3 words before [EOS] that beats "the cat [EOS]", and then one that also beats slide 26's 0.004.
      </>
    );
    stage = (
      <>
        <Panel>
          <Label>your sentence</Label>
          <StepRow seq={cseq} steps={steps} />
          <TokenButtons options={VOCAB} onPick={cAdd} disabled={fin} />
          <div style={{ display: "flex", gap: 6 }}>
            <Btn onClick={cUndo}>undo</Btn>
            <Btn onClick={cClear}>clear</Btn>
          </div>
        </Panel>
        <Label>next-token distribution after this prefix</Label>
        <DistBars dist={nextDist(fin ? cseq.slice(0, -1) : cseq)} chosen={fin ? "[EOS]" : null} />
      </>
    );
    side = (
      <>
        <Panel edge={g3 ? THEME.ok : THEME.grid}>
          <Readout
            rows={[
              ["your sentence", fin ? fmt(P, 5) : "(end with [EOS])", g3 ? THEME.ok : THEME.text],
              ["the cat [EOS]", fmt(ref, 5)],
            ]}
          />
          <Goal met={g1}>ends with [EOS], at least 3 words before it</Goal>
          <Goal met={g2}>more probable than "the cat [EOS]"</Goal>
          <Goal met={g3}>more probable than 0.004</Goal>
          <Hint hints={["Build \"the cat\" and look at the bars: how likely is [EOS] right there? What does the model expect instead?", "The factor P([EOS] | the cat) = 0.02 is the cost of stopping early. Continue with the token the model expects most, then compare cat with dog as the second word."]} />
        </Panel>
      </>
    );
  }

  return <LessonFrame screen={screen} onScreen={setScreen} title="The language-model game" question={question} stage={stage} side={side} />;
}
