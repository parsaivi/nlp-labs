import { useState, useCallback, useMemo } from "react";
import { THEME, MONO } from "./common";

// Lesson 6 - GPT-3: examples in the prompt, weights frozen (GPT, slides 56-71).
// Aha: few-shot learning puts solved examples into the input. The weights stay fixed; the prediction for
// the new input changes only because the prefix changed. More demonstrations (and larger models) help, but
// which demonstrations you choose matters, and every one of them costs context tokens.
// Toy: a frozen scorer that compares the query with each demonstration and leans toward the labels of
// similar ones. It is one simple mechanism for illustration, not a claim about how GPT-3 works inside.

// @sim-start
const STOP = { a: 1, the: 1, and: 1, was: 1, from: 1, to: 1, at: 1, all: 1, of: 1, it: 1 };
const DEMOS = [
  ["a wonderful moving story", "positive"],
  ["great acting and a wonderful score", "positive"],
  ["the plot was boring and slow", "negative"],
  ["boring characters and a dull plot", "negative"],
  ["funny and warm from start to end", "positive"],
  ["a slow dull mess", "negative"],
  ["the acting was great", "positive"],
  ["not funny at all", "negative"],
];
const TESTS = [
  ["a warm and funny film", "positive"],
  ["dull and boring", "negative"],
  ["wonderful acting", "positive"],
  ["slow plot", "negative"],
  ["a great story", "positive"],
  ["warm and wonderful", "positive"],
  ["funny acting", "positive"],
  ["a moving film", "positive"],
  ["dull acting", "negative"],
  ["a boring story", "negative"],
  ["slow and not funny", "negative"],
  ["a mess of a plot", "negative"],
];
const BETA = 1;
const GAMMA = 0.2;
const PRIOR_NEG = 0.5;
function words(s) {
  return s.split(" ").filter((w) => !STOP[w]);
}
function shared(a, b) {
  const B = words(b);
  return words(a).filter((w) => B.indexOf(w) >= 0).length;
}
// frozen scorer: label score = prior + BETA * (words shared with demos of that label) + GAMMA * (demos of that label)
function icl(query, demos) {
  const s = { positive: 0, negative: PRIOR_NEG };
  for (const d of demos) s[d[1]] += BETA * shared(query, d[0]) + GAMMA;
  const p = softmax([s.positive, s.negative]);
  return { pPos: p[0], label: p[0] >= 0.5 ? "positive" : "negative" };
}
function subsets(n, k) {
  const out = [];
  const rec = (i, cur) => {
    if (cur.length === k) {
      out.push(cur.slice());
      return;
    }
    for (let j = i; j < n; j++) {
      cur.push(j);
      rec(j + 1, cur);
      cur.pop();
    }
  };
  rec(0, []);
  return out;
}
// exact average test accuracy over every choice of k demonstrations
function avgAccuracy(k) {
  const S = subsets(DEMOS.length, k);
  let tot = 0;
  for (const s of S) {
    const demos = s.map((i) => DEMOS[i]);
    let c = 0;
    for (const t of TESTS) if (icl(t[0], demos).label === t[1]) c++;
    tot += c / TESTS.length;
  }
  return tot / S.length;
}
// @sim-end

const QUERY = "a warm and funny film";
function DemoBtn({ i, on, onFlip }) {
  const flip = useCallback(() => onFlip(i), [onFlip, i]);
  const d = DEMOS[i];
  return (
    <button onClick={flip} aria-pressed={on} style={{ display: "block", width: "100%", textAlign: "left", fontFamily: MONO, fontSize: 12, padding: "4px 7px", margin: "3px 0", borderRadius: 4, cursor: "pointer", border: "1px solid " + (on ? THEME.highlight : THEME.grid), background: on ? rgba(THEME.highlight, 0.12) : THEME.bg, color: THEME.text }}>
      {(on ? "■ " : "□ ") + "d" + (i + 1) + "  " + d[0]}
      <span style={{ color: d[1] === "positive" ? THEME.third : THEME.warn }}>{" → " + d[1]}</span>
    </button>
  );
}
function PromptBox({ demos, query, pred }) {
  return (
    <Panel>
      <Label>the whole input the frozen model reads</Label>
      <div style={{ fontFamily: MONO, fontSize: 12.5, lineHeight: 1.55 }}>
        {demos.map((d, i) => (
          <div key={i} style={{ marginBottom: 3 }}>
            <div>Review: {d[0]}</div>
            <div style={{ color: THEME.muted }}>Sentiment: {d[1]}</div>
          </div>
        ))}
        <div>Review: {query}</div>
        <div>
          Sentiment: <span style={{ color: THEME.highlight }}>{pred ? pred.label + " (" + fmt(pred.label === "positive" ? pred.pPos : 1 - pred.pPos, 2) + ")" : ""}</span>
        </div>
      </div>
    </Panel>
  );
}

export default function InContextLesson({ initialScreen }) {
  const [screen, setScreen] = useState(initialScreen || 0);
  const [pick, setPick] = useState(DEMOS.map(() => false));
  const [flipped, setFlipped] = useState(false);
  const [kk, setKk] = useState(2);
  const [tpd, setTpd] = useState(30);
  const [ctxLen, setCtxLen] = useState(2048);
  const [cpick, setCpick] = useState(DEMOS.map(() => false));

  const flip = useCallback((i) => setPick((P) => P.map((v, j) => (j === i ? !v : v))), []);
  const cflip = useCallback((i) => setCpick((P) => P.map((v, j) => (j === i ? !v : v))), []);
  const curve = useMemo(() => [0, 1, 2, 3, 4, 5, 6, 7, 8].map((k) => [k, 100 * avgAccuracy(k)]), []);

  let question, stage, side;
  if (screen === 0) {
    const demos = DEMOS.filter((d, i) => pick[i]);
    const pred = icl(QUERY, demos);
    const nPicked = demos.length;
    question = (
      <>
        <b>Where we are:</b> GPT-2 specified a task with a hint in the prefix (lesson 3). GPT-3 asks: what if the input also included solved examples (slides 56–58)? <b>The puzzle:</b> the model below is frozen. With no examples it calls <span style={{ fontFamily: MONO }}>{QUERY}</span> negative. Add demonstrations to the prompt until it answers positive. Then ask: which weights changed?
      </>
    );
    stage = <PromptBox demos={demos} query={QUERY} pred={pred} />;
    side = (
      <>
        <Panel>
          <Label>demonstrations in the prompt</Label>
          {DEMOS.map((d, i) => (
            <DemoBtn key={i} i={i} on={pick[i]} onFlip={flip} />
          ))}
          <Readout
            rows={[
              ["P(positive)", fmt(pred.pPos, 3), pred.label === "positive" ? THEME.ok : THEME.text],
              ["demonstrations", String(nPicked)],
              ["weights updated", "0"],
            ]}
          />
          <Goal met={pred.label === "positive"}>the frozen model answers positive</Goal>
        </Panel>
        <Note>Toy frozen scorer: it compares the new review with each demonstration and leans toward the labels of similar ones, plus a slight lean toward "negative" of its own. GPT-3's internals are far richer.</Note>
      </>
    );
  } else if (screen === 1) {
    question = (
      <>
        <b>Claim to test:</b> in-context learning happens during inference: the examples enter the prefix and the weights stay fixed (slides 59–61). If that is real, more demonstrations should help the same frozen model. Read the toy's curve, then GPT-3's reported numbers. How much does one example add over zero? Does it keep growing?
      </>
    );
    stage = (
      <>
        <LinePlot
          series={[{ label: "toy scorer, all demo sets", color: THEME.accent, shape: "circle", pts: curve }]}
          xr={[0, 8]}
          yr={[40, 100]}
          xticks={[0, 1, 2, 3, 4, 5, 6, 7, 8]}
          yticks={[40, 50, 60, 70, 80, 90, 100]}
          xlabel="demonstrations in the prompt (k)"
          ylabel="accuracy %"
          vline={kk}
          H={230}
          legendAt="br"
          ariaLabel="toy accuracy against number of demonstrations"
        />
        <Label>GPT-3 175B, reported (slides 68–69)</Label>
        <Bars
          rows={[
            { label: "TriviaQA 0-shot", value: 64.3, color: THEME.grid },
            { label: "TriviaQA 1-shot", value: 68.0, color: THEME.accent },
            { label: "TriviaQA few-shot", value: 71.2, color: THEME.third },
            { label: "3-digit + 0-shot", value: 34.2, color: THEME.grid },
            { label: "3-digit + 1-shot", value: 65.5, color: THEME.accent },
            { label: "3-digit + few-shot", value: 80.4, color: THEME.third },
            { label: "5-digit + few-shot", value: 9.3, color: THEME.warn },
          ]}
          range={[0, 100]}
          valueFmt={(v) => fmt(v, 1) + "%"}
          labelW={150}
          rowH={21}
          ariaLabel="GPT-3 accuracy by number of demonstrations"
        />
      </>
    );
    side = (
      <>
        <Panel>
          <Slider id="i6-k" label="demonstrations k" value={kk} min={0} max={8} stepSize={1} onChange={setKk} display={"k = " + kk} />
          <Readout
            rows={[
              ["toy accuracy at k", fmt(curve[kk][1], 1) + "%"],
              ["weights updated", "0"],
            ]}
          />
        </Panel>
        <Note>TriviaQA few-shot used 64 examples; a fine-tuned closed-book T5+SSM got 60.5%. Arithmetic jumps from 34.2% to 65.5% with a single example, yet 5-digit addition stays at 9.3%: success on one task does not ensure reliability (slide 71). Slides 65–66: on a symbol-removal task, larger models gain more from each extra example.</Note>
      </>
    );
  } else if (screen === 2) {
    const maxShots = Math.floor((ctxLen - 40) / tpd);
    question = (
      <>
        <b>What you found:</b> demonstrations steer a frozen model; the answer is still generated as the next tokens (slide 58). Fine-tuning moves the weights with gradients; in-context learning moves only the input. "Zero-, one-, few-shot" counts the demonstrations in the current input (slide 63). That input has a length limit.
      </>
    );
    stage = (
      <>
        <Eq size={13.5}>
          fine-tuning: θ ← θ − η ∇θ L(task data), then ŷ = argmax p_θ(y | x)
          <br />
          in-context: θ fixed, ŷ = argmax p_θ(y | demo₁, …, demo_k, x)
        </Eq>
        <Readout
          rows={[
            ["", "GPT-2 XL → GPT-3 175B (slide 64)"],
            ["parameters", "≈1.5B → 175B"],
            ["layers · hidden size", "48 · 1,600 → 96 · 12,288"],
            ["heads per layer", "25 → 96"],
            ["context", "1,024 → 2,048 tokens"],
          ]}
        />
      </>
    );
    side = (
      <>
        <Panel>
          <Slider id="i6-t" label="tokens per demonstration" value={tpd} min={5} max={300} stepSize={1} onChange={setTpd} />
          <Pills
            options={[
              [1024, "GPT-2: 1,024"],
              [2048, "GPT-3: 2,048"],
            ]}
            value={ctxLen}
            onPick={setCtxLen}
          />
          <Readout rows={[["demonstrations that fit (40 tokens left for the query)", String(Math.max(0, maxShots)), THEME.highlight]]} />
        </Panel>
        <Note>With about 30 tokens per question–answer pair, 64 TriviaQA examples need roughly 1,900 tokens: they fit GPT-3's 2,048, not GPT-2's 1,024. A prompt can also describe the task and its output format, not only demonstrate it (slide 62). A definition given in context ("glorp means a brief rain shower") does not update the model permanently (slide 70).</Note>
      </>
    );
  } else {
    const demos = DEMOS.filter((d, i) => cpick[i]);
    const res = TESTS.slice(0, 4).map((t) => icl(t[0], demos));
    const allOk = res.every((r, i) => r.label === TESTS[i][1]);
    const g1 = demos.length <= 2;
    question = (
      <>
        Prompt space is expensive: you may include at most <b>two</b> demonstrations. Pick them from the pool so that the same frozen model labels all four new reviews correctly. Wording and example choice can change the result (slide 71); which two examples carry the most?
      </>
    );
    stage = (
      <>
        {TESTS.slice(0, 4).map((t, i) => (
          <div key={i} style={{ display: "flex", justifyContent: "space-between", gap: 8, fontFamily: MONO, fontSize: 13, border: "1px solid " + THEME.grid, borderRadius: 4, padding: "5px 8px", margin: "4px 0", background: THEME.bg }}>
            <span>
              {t[0]} <span style={{ color: THEME.muted }}>(gold: {t[1]})</span>
            </span>
            <span style={{ color: res[i].label === t[1] ? THEME.ok : THEME.warn }}>
              {res[i].label} {fmt(res[i].pPos, 2)} {res[i].label === t[1] ? "◆" : "✗"}
            </span>
          </div>
        ))}
        <Note>The number is P(positive). The scorer's own lean toward "negative" is the same for every review.</Note>
      </>
    );
    side = (
      <>
        <Panel edge={allOk && g1 ? THEME.ok : THEME.grid}>
          {DEMOS.map((d, i) => (
            <DemoBtn key={i} i={i} on={cpick[i]} onFlip={cflip} />
          ))}
          <Goal met={g1}>at most two demonstrations ({demos.length} chosen)</Goal>
          <Goal met={allOk}>all four reviews labeled correctly</Goal>
          <Hint hints={["The two negative reviews are already right with no demonstrations. Which reviews need help, and which words do they contain?", "One demonstration shares two words with \"a warm and funny film\". The other should share words with \"wonderful acting\". Avoid a demo whose label fights a review it resembles."]} />
        </Panel>
      </>
    );
  }

  return <LessonFrame screen={screen} onScreen={setScreen} title="GPT-3: examples in the prompt, weights frozen" question={question} stage={stage} side={side} />;
}
