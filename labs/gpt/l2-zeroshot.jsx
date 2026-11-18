import { useState, useCallback } from "react";
import { THEME, MONO } from "./common";

// Lesson 2 - Zero-shot: let the LM's scores decide (GPT, slides 27-32).
// Aha: with no task head and no weight updates, a task becomes a comparison of LM probabilities:
// score each candidate text and pick the best. The scoring rule is part of the method: a sum of log-probs
// punishes every extra token, so long correct answers lose to short wrong ones; averaging per token fixes it.

// @sim-start
// score = (sum of ln p) / length^alpha : alpha 0 = sum, alpha 1 = average per token
function scoreOf(ps, alpha) {
  let s = 0;
  for (const p of ps) s += Math.log(p);
  return s / Math.pow(ps.length, alpha);
}
const CHALLENGE = [
  {
    doc: "Reza lost his keys at the park. His sister found them and gave them back.",
    q: "Who found the keys?",
    cands: [
      { text: "his sister found them for him", ps: [0.5, 0.55, 0.45, 0.4, 0.45, 0.45], gold: true },
      { text: "the park", ps: [0.3, 0.3] },
    ],
  },
  {
    doc: "Sara baked bread on Friday because her oven was finally fixed.",
    q: "What did Sara bake?",
    cands: [
      { text: "fresh bread", ps: [0.5, 0.6], gold: true },
      { text: "something for the family on the weekend", ps: [0.5, 0.55, 0.6, 0.45, 0.5, 0.4, 0.5, 0.42] },
    ],
  },
  {
    doc: "The train left Tabriz late, so Ali reached Tehran after midnight.",
    q: "When did Ali reach Tehran?",
    cands: [
      { text: "a little after midnight", ps: [0.45, 0.5, 0.5, 0.52], gold: true },
      { text: "late", ps: [0.37] },
    ],
  },
];
function picks(item, alpha) {
  let best = 0;
  for (let i = 1; i < item.cands.length; i++) if (scoreOf(item.cands[i].ps, alpha) > scoreOf(item.cands[best].ps, alpha)) best = i;
  return best;
}
// @sim-end

// slide 30 (candidates lengthened so that their lengths differ)
const RACE = {
  doc: "Mina missed the bus. She walked to school.",
  q: "How did Mina get to school?",
  cands: [
    { text: "She walked to school on foot .", ps: [0.6, 0.3, 0.6, 0.7, 0.4, 0.6, 0.8], gold: true },
    { text: "She took the bus .", ps: [0.6, 0.2, 0.6, 0.5, 0.8] },
    { text: "She drove .", ps: [0.6, 0.08, 0.8] },
  ],
};
// CoLA: average token log-probability (toy) and gold acceptability
const COLA = [
  ["Those children are playing outside.", -2.1, true],
  ["Those children is playing outside.", -2.9, false],
  ["The cats sleep.", -2.3, true],
  ["The cats sleeps.", -3.1, false],
  ["We ate rice for dinner.", -2.5, true],
  ["She has went home.", -2.6, false],
  ["Colorless green ideas sleep furiously.", -5.2, true],
  ["The the dog barked.", -3.8, false],
];
// SST-2: p(positive) and p(negative) as the next word after "<review> very" (toy), with and without "very"
const SST = [
  ["I loved this film.", true, [0.12, 0.01], [0.004, 0.001]],
  ["A wonderful story.", true, [0.1, 0.02], [0.002, 0.003]],
  ["The plot was boring.", false, [0.01, 0.09], [0.001, 0.003]],
  ["Two hours I will never get back.", false, [0.03, 0.05], [0.002, 0.001]],
  ["Great, another sequel nobody asked for.", false, [0.06, 0.04], [0.003, 0.001]],
];
// DPRD: P("was too <adj>." | sentence with "it" replaced)
const DPRD = { large: { trophy: 0.2, suitcase: 0.05 }, small: { trophy: 0.04, suitcase: 0.18 } };

function CandRows({ item, alpha, chosen }) {
  return (
    <div>
      {item.cands.map((c, i) => {
        const s = scoreOf(c.ps, alpha);
        return (
          <div key={i} style={{ border: "1px solid " + (i === chosen ? THEME.highlight : THEME.grid), borderRadius: 5, padding: "6px 8px", margin: "5px 0", background: THEME.bg }}>
            <div style={{ display: "flex", justifyContent: "space-between", gap: 8, fontFamily: MONO, fontSize: 13 }}>
              <span>
                {i === chosen ? "▶ " : "  "}
                {c.text}
              </span>
              <span style={{ color: i === chosen ? THEME.highlight : THEME.muted }}>score {fmt(s, 3)}</span>
            </div>
            <div style={{ display: "flex", gap: 3, marginTop: 4, flexWrap: "wrap" }}>
              {c.ps.map((p, j) => (
                <span key={j} title={c.text.split(" ")[j]} style={{ fontFamily: MONO, fontSize: 10.5, padding: "1px 4px", borderRadius: 2, background: rgba(THEME.accent, clamp(p, 0.08, 1) * 0.8), color: THEME.text }}>
                  {fmt(p, 2)}
                </span>
              ))}
              <span style={{ fontFamily: MONO, fontSize: 10.5, color: THEME.muted, marginLeft: 4 }}>{c.ps.length} tokens{c.gold ? " · correct" : ""}</span>
            </div>
          </div>
        );
      })}
    </div>
  );
}

export default function ZeroShotLesson({ initialScreen }) {
  const [screen, setScreen] = useState(initialScreen || 0);
  const [rule, setRule] = useState("sum");
  const [tried, setTried] = useState({ sum: true });
  const [task, setTask] = useState("cola");
  const [tau, setTau] = useState(-3.5);
  const [very, setVery] = useState(true);
  const [adj, setAdj] = useState("large");
  const [acc, setAcc] = useState(60);
  const [alpha, setAlpha] = useState(0);

  const pickRule = useCallback((r) => {
    setRule(r);
    setTried((T) => Object.assign({}, T, { [r]: true }));
  }, []);

  let question, stage, side;
  if (screen === 0) {
    const a = rule === "sum" ? 0 : 1;
    const ch = picks(RACE, a);
    question = (
      <>
        <b>Where we are:</b> lesson 1 pretrained GPT on next-token prediction. GPT-1 then fine-tuned a head per task, but also asked: can the language model solve a task with no task weights at all (slide 27)? <b>The puzzle (after slide 30):</b> score each candidate answer by how probable the LM finds it after the document and question, and pick the best. Each small box is one token's probability. Does "add up the log-probabilities" choose the right answer? Try the other rule.
      </>
    );
    stage = (
      <>
        <Panel>
          <div style={{ fontFamily: MONO, fontSize: 13, lineHeight: 1.6 }}>
            <div>{RACE.doc}</div>
            <div style={{ color: THEME.muted }}>Q: {RACE.q}</div>
          </div>
        </Panel>
        <CandRows item={RACE} alpha={a} chosen={ch} />
      </>
    );
    side = (
      <>
        <Panel>
          <Label>scoring rule</Label>
          <Pills
            options={[
              ["sum", "sum of ln p"],
              ["avg", "average ln p per token"],
            ]}
            value={rule}
            onPick={pickRule}
          />
          <Readout rows={[["chosen answer", RACE.cands[ch].text, RACE.cands[ch].gold ? THEME.ok : THEME.warn]]} />
          <Goal met={!!tried.sum && !!tried.avg}>compare both rules</Goal>
        </Panel>
        <Note>Toy probabilities. Every token multiplies in a factor below 1, so a sum of logs favors the shortest candidate, whatever it says.</Note>
      </>
    );
  } else if (screen === 1) {
    let body = null;
    let read = [];
    if (task === "cola") {
      const right = COLA.filter((r) => r[1] > tau === r[2]).length;
      read = [["accuracy", right + " / " + COLA.length, right === 7 ? THEME.ok : THEME.text]];
      body = (
        <>
          <Bars rows={COLA.map((r) => ({ label: (r[2] ? "✓ " : "✗ ") + r[0], value: r[1] + 6, color: r[1] > tau ? THEME.accent : THEME.grid, mark: r[1] > tau === r[2] ? "" : "wrong" }))} range={[0, 5]} valueFmt={(v) => fmt(v - 6, 1)} labelW={300} rowH={22} ariaLabel="average log-probability per sentence" />
          <Slider id="z2-tau" label="threshold τ: acceptable if average ln p > τ" value={tau} min={-6} max={-1} stepSize={0.05} onChange={setTau} />
        </>
      );
    } else if (task === "sst") {
      const right = SST.filter((r) => {
        const p = very ? r[2] : r[3];
        return p[0] > p[1] === r[1];
      }).length;
      read = [["accuracy", right + " / " + SST.length]];
      body = (
        <>
          {SST.map((r, i) => {
            const p = very ? r[2] : r[3];
            const ok = p[0] > p[1] === r[1];
            return (
              <div key={i} style={{ fontFamily: MONO, fontSize: 12.5, display: "flex", justifyContent: "space-between", gap: 8, margin: "4px 0" }}>
                <span>
                  {r[0]}
                  {very ? <span style={{ color: THEME.accent }}> very</span> : null} → {p[0] > p[1] ? "positive" : "negative"}
                </span>
                <span style={{ color: ok ? THEME.ok : THEME.warn }}>
                  {fmt(p[0], 3)} vs {fmt(p[1], 3)} {ok ? "◆" : "✗"}
                </span>
              </div>
            );
          })}
          <Toggle label='append " very" to the review' value={very} onChange={setVery} />
        </>
      );
    } else if (task === "race") {
      const ch = picks(RACE, 1);
      read = [["chosen (average rule)", RACE.cands[ch].text]];
      body = <CandRows item={RACE} alpha={1} chosen={ch} />;
    } else {
      const P = DPRD[adj];
      read = [["chosen referent", P.trophy > P.suitcase ? "the trophy" : "the suitcase"]];
      body = (
        <>
          <div style={{ fontFamily: MONO, fontSize: 13, marginBottom: 6 }}>The trophy did not fit in the suitcase because it was too {adj}.</div>
          <Bars
            rows={[
              { label: "the trophy", value: P.trophy, color: THEME.accent },
              { label: "the suitcase", value: P.suitcase, color: THEME.third },
            ]}
            range={[0, 0.25]}
            digits={2}
            labelW={110}
            ariaLabel="probability of the continuation after each substitution"
          />
          <Pills
            options={[
              ["large", "too large"],
              ["small", "too small"],
            ]}
            value={adj}
            onPick={setAdj}
          />
        </>
      );
    }
    question = (
      <>
        <b>Claim to test:</b> GPT-1 turned four tasks into LM scores with a task-specific rule, no fine-tuning (slides 28–31). Try each. CoLA: is there a threshold that gets all 8 sentences right? SST-2: why add "very" before comparing "positive" with "negative"? DPRD: replace "it" by each noun and score the rest of the sentence. What flips the answer?
      </>
    );
    stage = (
      <>
        <Pills
          options={[
            ["cola", "CoLA"],
            ["sst", "SST-2"],
            ["race", "RACE"],
            ["dprd", "DPRD"],
          ]}
          value={task}
          onPick={setTask}
        />
        {body}
      </>
    );
    side = (
      <>
        <Panel>
          <Readout rows={read} />
        </Panel>
        <Note>
          {task === "cola"
            ? "Best threshold: 7 of 8. \"Colorless green ideas sleep furiously\" is grammatical but improbable. Likelihood is a useful signal, not a grammar guarantee (slide 28)."
            : task === "sst"
            ? "After \"… very\", the next word is likely an adjective, so the two label words get real probability to compare. Sarcasm still fools the comparison."
            : task === "race"
            ? "RACE: choose the candidate with the highest average log-probability per token, given document and question (slide 30)."
            : "Only the continuation \"was too large.\" is scored; the substitution changes its context (slide 31). All numbers on this screen are toys."}
        </Note>
      </>
    );
  } else if (screen === 2) {
    const norm = (acc - 50) / (90 - 50);
    question = (
      <>
        <b>What you found:</b> each zero-shot task is "score some text with the LM, then decide by a rule": a threshold, a comparison of two next words, the best average log-probability, or the more likely substitution. No weights change and there is no learned classifier (slide 27). The rule matters as much as the model.
      </>
    );
    stage = (
      <>
        <Eq size={13}>
          CoLA: (1/T) Σₜ ln p(xₜ | x_&lt;t) &gt; τ
          <br />
          SST-2: argmax over y ∈ {"{positive, negative}"} of p(y | review + " very")
          <br />
          RACE: argmax over answers a of (1/|a|) Σₜ ln p(aₜ | D, Q, a_&lt;t)
          <br />
          DPRD: argmax over nouns c of p(continuation | sentence with it → c)
        </Eq>
        <Eq size={13}>
          normalized score = (accuracy − random) / (SOTA − random)
          <br />= ({acc} − 50) / (90 − 50) = {fmt(norm, 2)}
        </Eq>
      </>
    );
    side = (
      <>
        <Panel>
          <Slider id="z2-acc" label="zero-shot accuracy on a 2-class task (%)" value={acc} min={40} max={95} stepSize={0.5} onChange={setAcc} display={fmt(acc, 1) + "%"} />
          <Readout rows={[["normalized score (random 50%, SOTA 90%)", fmt(norm, 2)]]} />
        </Panel>
        <Note>GPT-1's Figure 2 (slide 32) plots this normalized score, 0 = random and 1 = the then-best single model, for each task during pretraining: the scores rise as the LM gets better at predicting text. The 50% and 90% here are made-up example values. Slide 32's own question: how much comes from the LM, and how much from the scoring rule?</Note>
      </>
    );
  } else {
    const res = CHALLENGE.map((it) => picks(it, alpha));
    const okAll = res.every((ch, i) => CHALLENGE[i].cands[ch].gold);
    question = (
      <>
        Three new reading questions. Between the sum (α = 0) and the average (α = 1) there is a family of rules: divide the summed log-probability by length^α. Find an α for which the zero-shot model answers all three correctly. Is the plain average inside the working range?
      </>
    );
    stage = (
      <>
        {CHALLENGE.map((it, i) => (
          <Panel key={i} style={{ marginBottom: 6 }}>
            <div style={{ fontFamily: MONO, fontSize: 12.5, lineHeight: 1.5 }}>
              <div>{it.doc}</div>
              <div style={{ color: THEME.muted }}>Q: {it.q}</div>
            </div>
            <CandRows item={it} alpha={alpha} chosen={res[i]} />
          </Panel>
        ))}
      </>
    );
    side = (
      <>
        <Panel edge={okAll ? THEME.ok : THEME.grid}>
          <Slider id="z2-a" label="length exponent α" value={alpha} min={0} max={1.5} stepSize={0.01} onChange={setAlpha} display={"α = " + fmt(alpha, 2)} />
          <Eq size={12.5}>score(a) = Σₜ ln p(aₜ | …) / |a|^α</Eq>
          {CHALLENGE.map((it, i) => (
            <Goal key={i} met={it.cands[res[i]].gold}>
              {"question " + (i + 1) + ": " + it.cands[res[i]].text}
            </Goal>
          ))}
          <Hint hints={["At α = 0, which questions go wrong, and are the wrong picks shorter or longer than the right answers?", "Raising α forgives length. Too high, and a long, generic answer made of very predictable tokens wins instead. Look between the two extremes."]} />
        </Panel>
      </>
    );
  }

  return <LessonFrame screen={screen} onScreen={setScreen} title="Zero-shot: let the LM's scores decide" question={question} stage={stage} side={side} />;
}
