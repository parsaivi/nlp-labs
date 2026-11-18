import { useState, useCallback, useMemo } from "react";
import { THEME, MONO } from "./common";

// Lesson 3 - GPT-2: the task lives in the prefix (GPT, slides 33-43 and 46-52).
// Aha: GPT-2 has no task heads. A prefix that looks like a kind of web page ("French:", "TL;DR:", "Q: … A:")
// makes the most likely continuation the answer, because the web already contains demonstrations of those
// tasks. The toy here is an n-gram model that can only continue with text it has seen, which makes two things
// visible: where task demonstrations come from, and why overlap between training text and test data matters.

// @sim-start
const DOCS = [
  { id: "D1", name: "bilingual article", text: "In French , bonjour means hello in English ." },
  { id: "D2", name: "word list page", text: "English : cat French : chat . English : dog French : chien . English : hello French : bonjour ." },
  { id: "D3", name: "forum post", text: "I missed the bus , walked in the rain , and arrived late . TL;DR : a rough commute ." },
  { id: "D4", name: "Q&A page", text: "Q : Why does ice float ? A : Ice is less dense than liquid water ." },
  { id: "D5", name: "quiz page", text: "Q : Who wrote Hamlet ? A : William Shakespeare ." },
  { id: "D6", name: "story", text: "The cat sat on the mat . Then it slept in the sun ." },
  { id: "D7", name: "news item", text: "The train was delayed by snow . It arrived two hours late . TL;DR : snow delayed the train ." },
  { id: "D8", name: "encyclopedia line", text: "Marie Curie won the Nobel Prize in Physics in 1903 ." },
];
function tokenize(s) {
  return s
    .replace(/([.,?!])/g, " $1 ")
    .replace(/:/g, " : ")
    .split(/\s+/)
    .filter(Boolean);
}
// longest-suffix match (up to 6 tokens) against the enabled documents; greedy; stops after "."
function generate(prompt, enabled, maxNew) {
  const docs = DOCS.filter((d, i) => enabled[i]).map((d) => ({ id: d.id, t: tokenize(d.text) }));
  const seq = tokenize(prompt);
  const out = [];
  for (let step = 0; step < (maxNew || 10); step++) {
    let found = null;
    for (let k = Math.min(6, seq.length); k >= 1 && !found; k--) {
      const suf = seq.slice(seq.length - k);
      const counts = {};
      const src = {};
      for (const d of docs) {
        for (let i = 0; i + k < d.t.length; i++) {
          let ok = true;
          for (let j = 0; j < k; j++) if (d.t[i + j] !== suf[j]) ok = false;
          if (ok) {
            const nx = d.t[i + k];
            counts[nx] = (counts[nx] || 0) + 1;
            if (!src[nx]) src[nx] = d.id;
          }
        }
      }
      let best = null;
      for (const w in counts) if (best === null || counts[w] > counts[best]) best = w;
      if (best !== null) found = { tok: best, k, src: src[best] };
    }
    if (!found) break;
    out.push(found);
    seq.push(found.tok);
    if (found.tok === ".") break;
  }
  return out;
}
// @sim-end

const PRESETS = [
  ["English : dog French :", "translate"],
  ["Q : Who wrote Hamlet ? A :", "answer"],
  ["The train was delayed by snow . It arrived two hours late . TL;DR :", "summarize"],
  ["The cat sat on the", "continue a story"],
];
const TESTS = [
  { prompt: "English : dog French :", gold: "chien", overlap: true },
  { prompt: "Q : Who wrote Hamlet ? A :", gold: "William Shakespeare", overlap: true },
  { prompt: "Marie Curie won the Nobel Prize in Physics in", gold: "1903", overlap: true },
  { prompt: "English : bird French :", gold: "oiseau", overlap: false },
  { prompt: "Q : Who painted the Mona Lisa ? A :", gold: "Leonardo da Vinci", overlap: false },
  { prompt: "The bus was delayed by rain . It came late . TL;DR :", gold: "rain delayed the bus", overlap: false },
];
const ALL_ON = DOCS.map(() => true);
function answerText(gen) {
  return gen
    .map((g) => g.tok)
    .filter((t) => t !== ".")
    .join(" ");
}

function GenView({ prompt, gen }) {
  return (
    <Panel>
      <div style={{ fontFamily: MONO, fontSize: 13.5, lineHeight: 1.9 }}>
        <span style={{ color: THEME.muted }}>{prompt} </span>
        {gen.map((g, i) => (
          <span key={i} style={{ whiteSpace: "nowrap" }}>
            <span style={{ color: THEME.highlight, borderBottom: "1px dashed " + THEME.highlight }}>{g.tok}</span>
            <sub style={{ fontSize: 9.5, color: THEME.muted, marginRight: 4 }}>
              {g.src}·{g.k}
            </sub>
          </span>
        ))}
        {gen.length === 0 ? <span style={{ color: THEME.warn }}> (no match: the toy cannot continue)</span> : null}
      </div>
      <Note>Generated tokens in orange. Under each: the document it was copied from · how many prompt tokens matched.</Note>
    </Panel>
  );
}
function DocToggle({ i, on, onFlip }) {
  const flip = useCallback(() => onFlip(i), [onFlip, i]);
  const d = DOCS[i];
  return (
    <button onClick={flip} aria-pressed={on} style={{ display: "block", width: "100%", textAlign: "left", fontFamily: MONO, fontSize: 11.5, padding: "4px 6px", margin: "3px 0", borderRadius: 4, cursor: "pointer", border: "1px solid " + (on ? THEME.accent : THEME.grid), background: on ? rgba(THEME.accent, 0.1) : THEME.bg, color: on ? THEME.text : THEME.muted }}>
      {(on ? "■ " : "□ ") + d.id + " " + d.name + ": "}
      <span style={{ color: THEME.muted }}>{d.text.length > 54 ? d.text.slice(0, 52) + "…" : d.text}</span>
    </button>
  );
}
function TestRow({ i, t, gen, excluded, onFlip }) {
  const flip = useCallback(() => onFlip(i), [onFlip, i]);
  const ans = answerText(gen);
  const ok = ans === t.gold;
  return (
    <div style={{ border: "1px solid " + THEME.grid, borderRadius: 5, padding: "6px 8px", margin: "5px 0", opacity: excluded ? 0.5 : 1, background: THEME.bg }}>
      <div style={{ display: "flex", justifyContent: "space-between", gap: 8, alignItems: "center" }}>
        <span style={{ fontFamily: MONO, fontSize: 12.5 }}>{t.prompt}</span>
        <Toggle label={excluded ? "excluded" : "exclude"} value={excluded} onChange={flip} />
      </div>
      <div style={{ fontFamily: MONO, fontSize: 12 }}>
        <span style={{ color: THEME.muted }}>model: </span>
        <span style={{ color: THEME.highlight }}>{ans || "—"}</span>
        <span style={{ color: THEME.muted }}> · gold: {t.gold} · </span>
        <span style={{ color: ok ? THEME.ok : THEME.warn }}>{ok ? "◆ correct" : "✗ wrong"}</span>
        <span style={{ color: THEME.muted }}> · copied from {gen.length ? gen.map((g) => g.src + "·" + g.k).join(" ") : "—"}</span>
      </div>
    </div>
  );
}

export default function PromptLesson({ initialScreen }) {
  const [screen, setScreen] = useState(initialScreen || 0);
  const [prompt, setPrompt] = useState(PRESETS[3][0]);
  const [usedP, setUsedP] = useState({ 3: true });
  const [enabled, setEnabled] = useState(ALL_ON);
  const [hint, setHint] = useState(true);
  const [excl, setExcl] = useState(TESTS.map(() => false));

  const onType = useCallback((e) => setPrompt(e.target.value), []);
  const pickPreset = useCallback((k) => {
    setPrompt(PRESETS[k][0]);
    setUsedP((U) => Object.assign({}, U, { [k]: true }));
  }, []);
  const flipDoc = useCallback((i) => setEnabled((E) => E.map((v, j) => (j === i ? !v : v))), []);
  const allOn = useCallback(() => setEnabled(ALL_ON), []);
  const flipEx = useCallback((i) => setExcl((E) => E.map((v, j) => (j === i ? !v : v))), []);
  const gen = useMemo(() => generate(prompt, screen === 1 ? enabled : ALL_ON, 10), [prompt, enabled, screen]);
  const testGens = useMemo(() => TESTS.map((t) => generate(t.prompt, ALL_ON, 10)), []);

  let question, stage, side;
  if (screen === 0) {
    question = (
      <>
        <b>Where we are:</b> GPT-1 still needed a fine-tuned head per task, or a hand-made scoring rule (lessons 1–2). GPT-2 asks whether one model can do different tasks with no task-specific training at all (slide 34): just predict, append, repeat (slides 36–39). <b>The puzzle:</b> one fixed model, four prompts. Nothing tells it which task to do. Try the prompts and type your own. What in the prompt picks the task?
      </>
    );
    stage = (
      <>
        <label htmlFor="p3-in" style={{ display: "block" }}>
          <Label>prompt (tokens are split on spaces and punctuation)</Label>
          <input id="p3-in" value={prompt} onChange={onType} style={{ width: "100%", boxSizing: "border-box", fontFamily: MONO, fontSize: 14, padding: "7px 9px", background: THEME.bg, color: THEME.text, border: "1px solid " + THEME.grid, borderRadius: 4 }} />
        </label>
        <div style={{ display: "flex", gap: 6, flexWrap: "wrap", margin: "8px 0" }}>
          {PRESETS.map((p, k) => (
            <Pill key={k} value={k} active={prompt === p[0]} onPick={pickPreset}>
              {p[1]}
            </Pill>
          ))}
        </div>
        <GenView prompt={prompt} gen={gen} />
      </>
    );
    side = (
      <>
        <Panel>
          <Goal met={Object.keys(usedP).length >= 4}>try all four prompts</Goal>
          <Readout rows={[["weights changed between prompts", "0"]]} />
        </Panel>
        <Note>Toy model: a greedy n-gram model over 8 small "web pages". It finds the longest end of your prompt that appears in its training text and copies what came next. GPT-2 is a 1.5B-parameter Transformer, not a lookup, but the same rule holds for both: the prefix decides which continuation is most likely (slide 40).</Note>
      </>
    );
  } else if (screen === 1) {
    question = (
      <>
        <b>Claim to test:</b> the web already contains demonstrations of tasks: bilingual pages, "TL;DR:" posts, Q&amp;A pages (slide 41). If a task works only because such pages were in the training text, removing them should break it. Turn pages off and on. Which page does the translation come from? What does the toy produce when its demonstration is gone?
      </>
    );
    stage = (
      <>
        <label htmlFor="p3-in2" style={{ display: "block" }}>
          <Label>prompt</Label>
          <input id="p3-in2" value={prompt} onChange={onType} style={{ width: "100%", boxSizing: "border-box", fontFamily: MONO, fontSize: 14, padding: "7px 9px", background: THEME.bg, color: THEME.text, border: "1px solid " + THEME.grid, borderRadius: 4 }} />
        </label>
        <div style={{ display: "flex", gap: 6, flexWrap: "wrap", margin: "8px 0" }}>
          {PRESETS.map((p, k) => (
            <Pill key={k} value={k} active={prompt === p[0]} onPick={pickPreset}>
              {p[1]}
            </Pill>
          ))}
        </div>
        <GenView prompt={prompt} gen={gen} />
      </>
    );
    side = (
      <>
        <Panel>
          <Label>training text (click to remove or restore)</Label>
          {DOCS.map((d, i) => (
            <DocToggle key={d.id} i={i} on={enabled[i]} onFlip={flipDoc} />
          ))}
          <Btn onClick={allOn}>restore all</Btn>
        </Panel>
      </>
    );
  } else if (screen === 2) {
    question = (
      <>
        <b>What you found:</b> the task is expressed in the prefix, and the continuation completes it. The model learned those formats from pages that already demonstrate them. GPT-2 kept GPT-1's core, a causal Transformer trained on next-token prediction, and changed the data and the scale (slides 42–43). Toggle the TL;DR hint on the paper's summarization result.
      </>
    );
    stage = (
      <>
        <Label>GPT-1 (2018) → GPT-2 (2019), slide 42</Label>
        <Readout
          rows={[
            ["data", "BooksCorpus → WebText (8M+ docs, 40 GB)"],
            ["largest model", "≈117M, 12 layers → ≈1.5B, 48 layers"],
            ["context", "512 → 1,024 tokens"],
            ["emphasis", "pretrain → fine-tune  →  zero-shot task completion"],
          ]}
        />
        <Label>CNN/Daily Mail summarization, ROUGE-AVG (slide 49)</Label>
        <Bars
          rows={[
            { label: "GPT-2 " + (hint ? "+ TL;DR:" : "no hint"), value: hint ? 21.4 : 15.03, color: THEME.accent },
            { label: "random 3 sentences", value: 20.98, color: THEME.grid },
            { label: "first 3 sentences", value: 31.55, color: THEME.third },
            { label: "Bottom-Up Sum", value: 32.75, color: THEME.third },
          ]}
          range={[0, 35]}
          digits={2}
          labelW={150}
          ariaLabel="summarization scores"
        />
        <Label>CoQA dev F1, no CoQA fine-tuning for GPT-2 (slide 47)</Label>
        <Bars
          rows={[
            { label: "Seq2seq", value: 27.5, color: THEME.grid },
            { label: "PGNet", value: 45.4, color: THEME.grid },
            { label: "DrQA", value: 54.7, color: THEME.grid },
            { label: "GPT-2 (zero-shot)", value: 55.0, color: THEME.accent },
            { label: "DrQA+PGNet", value: 66.2, color: THEME.grid },
          ]}
          range={[0, 70]}
          digits={1}
          labelW={150}
          ariaLabel="CoQA F1"
        />
      </>
    );
    side = (
      <>
        <Panel>
          <Toggle label='add the "TL;DR:" hint' value={hint} onChange={setHint} />
          <Readout rows={[["gain from the hint", "+6.37 ROUGE"]]} />
        </Panel>
        <Note>The hint adds 6.37 points, but the "first 3 sentences" baseline still wins: news articles put the main facts first. WebText kept pages linked from Reddit with ≥ 3 karma, a signal of interest, not of accuracy, and excluded Wikipedia (slide 43). The gray CoQA systems were trained on CoQA. A fluent continuation can still be fiction (slide 50).</Note>
      </>
    );
  } else {
    const kept = TESTS.map((t, i) => i).filter((i) => !excl[i]);
    const right = kept.filter((i) => answerText(testGens[i]) === TESTS[i].gold).length;
    const rawRight = TESTS.filter((t, i) => answerText(testGens[i]) === t.gold).length;
    const g1 = TESTS.every((t, i) => !t.overlap || excl[i]);
    const g2 = TESTS.every((t, i) => t.overlap || !excl[i]);
    question = (
      <>
        You report the toy's zero-shot accuracy on six test prompts: {rawRight} / 6 correct. But zero-shot only means "no task training", not "never saw the test text" (slide 52). Exclude every test item whose answer the model could have copied from its training text, and keep the others. What is the honest score?
      </>
    );
    stage = (
      <>
        {TESTS.map((t, i) => (
          <TestRow key={i} i={i} t={t} gen={testGens[i]} excluded={excl[i]} onFlip={flipEx} />
        ))}
      </>
    );
    side = (
      <>
        <Panel edge={g1 && g2 ? THEME.ok : THEME.grid}>
          <Readout
            rows={[
              ["items kept", kept.length + " / 6"],
              ["accuracy on kept items", kept.length ? right + " / " + kept.length : "—", g1 && g2 ? THEME.highlight : THEME.text],
            ]}
          />
          <Goal met={g1}>every item that overlaps the training text is excluded</Goal>
          <Goal met={g2}>no clean item is excluded</Goal>
          <Hint hints={["Look at \"copied from\" for the first answer token: a match of 5 or 6 tokens means the end of the test prompt appears word for word in a training page.", "Search the pages in the Explore screen: three test prompts appear word for word in D2, D5 and D8."]} />
        </Panel>
        <Note>The toy's honest score is 0: it can only copy. GPT-2 also met overlap: about 15% of CoQA's news passages appeared in WebText, but the estimated benefit was only 0.5–1.0 F1 (slide 52). Most of its 55.0 came from something a lookup table cannot do.</Note>
      </>
    );
  }

  return <LessonFrame screen={screen} onScreen={setScreen} title="GPT-2: the task lives in the prefix" question={question} stage={stage} side={side} />;
}
