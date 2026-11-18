import { useState, useCallback, useMemo, useEffect } from "react";
import { THEME, MONO } from "./common";

// Lesson 3 - Count, divide, multiply: the bigram model (Pretraining, slides 31-39 and 44).
// Aha: an n-gram model's probabilities are counts divided by context totals. A new sentence gets a score
// if each of its pairs was seen somewhere, but a single unseen pair makes the whole product 0, and evidence
// about one context ("dog") never reaches another ("cat").

// @sim-start
const BASE = [
  ["the dog sleeps", 3],
  ["the cat sleeps", 2],
  ["a dog runs", 2],
  ["a cat sleeps", 1],
];
const POOL = ["the cat runs", "a cat runs", "the dog runs", "a dog sleeps"];
function toks(s) {
  return ["[BOS]"].concat(s.split(" "), ["[EOS]"]);
}
// n = 1 (unigram), 2 (bigram) or 3 (trigram): counts[context][next]
function countTable(corpus, n) {
  const C = {};
  for (const item of corpus) {
    const w = toks(item[0]);
    const k = item[1];
    if (!k) continue;
    for (let i = 1; i < w.length; i++) {
      let ctx = "∅";
      if (n === 2) ctx = w[i - 1];
      if (n === 3) ctx = i >= 2 ? w[i - 2] + " " + w[i - 1] : "[BOS]";
      C[ctx] = C[ctx] || {};
      C[ctx][w[i]] = (C[ctx][w[i]] || 0) + k;
    }
  }
  return C;
}
function rowTotal(row) {
  let t = 0;
  for (const k in row) t += row[k];
  return t;
}
function trace(C, s) {
  const w = toks(s);
  const out = [];
  for (let i = 1; i < w.length; i++) {
    const row = C[w[i - 1]] || {};
    const t = rowTotal(row);
    const c = row[w[i]] || 0;
    out.push({ from: w[i - 1], to: w[i], c, t, p: t ? c / t : 0 });
  }
  return out;
}
function score(C, s) {
  return trace(C, s).reduce((a, r) => a * r.p, 1);
}
// @sim-end

const NEXT = ["the", "a", "dog", "cat", "sleeps", "runs", "[EOS]"];
const CTX2 = ["[BOS]", "the", "a", "dog", "cat", "sleeps", "runs"];
const TESTS = ["the dog sleeps", "the dog runs", "the cat runs"];
const SEEN = { "the dog sleeps": 1, "the cat sleeps": 1, "a dog runs": 1, "a cat sleeps": 1 };

function Trace({ C, s }) {
  const tr = trace(C, s);
  const P = tr.reduce((a, r) => a * r.p, 1);
  return (
    <Panel>
      <Label>{s + " [EOS]"}</Label>
      <div style={{ display: "grid", gridTemplateColumns: "auto auto auto", gap: "2px 16px", fontFamily: MONO, fontSize: 13, lineHeight: 1.6 }}>
        <span style={{ color: THEME.muted }}>bigram</span>
        <span style={{ color: THEME.muted, textAlign: "right" }}>count / total</span>
        <span style={{ color: THEME.muted, textAlign: "right" }}>P</span>
        {tr.map((r, i) => (
          <React.Fragment key={i}>
            <span>{r.from + " → " + r.to}</span>
            <span style={{ textAlign: "right" }}>{r.c + "/" + r.t}</span>
            <span style={{ textAlign: "right", color: r.p === 0 ? THEME.warn : THEME.text }}>{fmt(r.p, 3) + (r.p === 0 ? " ✗" : "")}</span>
          </React.Fragment>
        ))}
      </div>
      <div style={{ fontFamily: MONO, fontSize: 14, marginTop: 6 }}>
        {tr.map((r) => r.c + "/" + r.t).join(" × ")} = <b style={{ color: P === 0 ? THEME.warn : THEME.highlight }}>{fmt(P, 3)}</b>
      </div>
    </Panel>
  );
}
function CorpusList({ corpus }) {
  return (
    <div style={{ fontFamily: MONO, fontSize: 13, lineHeight: 1.6 }}>
      {corpus
        .filter((c) => c[1] > 0)
        .map((c, i) => (
          <div key={i} style={{ display: "flex", justifyContent: "space-between", gap: 10 }}>
            <span>{c[0]}</span>
            <span style={{ color: THEME.muted }}>× {c[1]}</span>
          </div>
        ))}
    </div>
  );
}
function CountMatrix({ C, ctxs, onCell, hi }) {
  const vals = ctxs.map((r) => NEXT.map((c) => (C[r] && C[r][c]) || 0));
  return (
    <Matrix
      rowLabels={ctxs}
      colLabels={NEXT}
      values={vals}
      shade={(v, i) => {
        const t = vals[i].reduce((a, b) => a + b, 0);
        return t ? (v / t) * 0.8 : 0;
      }}
      text={(v) => (v ? String(v) : "·")}
      onCell={onCell}
      hiRow={hi ? hi[0] : undefined}
      hiCol={hi ? hi[1] : undefined}
      cell={44}
      rowW={Math.max.apply(null, ctxs.map((x) => x.length)) > 6 ? 104 : 64}
      colH={28}
      ariaLabel="next-token counts: rows are contexts, columns are next tokens"
    />
  );
}
function PlusMinus({ label, value, onChange, max }) {
  const dec = useCallback(() => onChange(label, Math.max(0, value - 1)), [label, value, onChange]);
  const inc = useCallback(() => onChange(label, Math.min(max, value + 1)), [label, value, onChange, max]);
  return (
    <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 8, fontFamily: MONO, fontSize: 13, margin: "3px 0" }}>
      <span>{label}</span>
      <span style={{ display: "flex", gap: 6, alignItems: "center" }}>
        <Btn onClick={dec} disabled={value === 0}>
          −
        </Btn>
        <span style={{ minWidth: 22, textAlign: "center" }}>{value}</span>
        <Btn onClick={inc} disabled={value === max}>
          +
        </Btn>
      </span>
    </div>
  );
}

export default function CountsLesson({ initialScreen }) {
  const [screen, setScreen] = useState(initialScreen || 0);
  const [w1, setW1] = useState("the");
  const [w2, setW2] = useState("dog");
  const [w3, setW3] = useState("sleeps");
  const [found, setFound] = useState({});
  const [ks, setKs] = useState({ "the dog sleeps": 3, "the cat sleeps": 2, "a dog runs": 2, "a cat sleeps": 1 });
  const [order, setOrder] = useState(2);
  const [cell, setCell] = useState([3, 4]);
  const [add, setAdd] = useState({ "the cat runs": 0, "a cat runs": 0, "the dog runs": 0, "a dog sleeps": 0 });

  const C0 = useMemo(() => countTable(BASE, 2), []);
  const pick = useCallback((setter, v) => setter(v), []);
  const pickW1 = useCallback((v) => pick(setW1, v), [pick]);
  const pickW2 = useCallback((v) => pick(setW2, v), [pick]);
  const pickW3 = useCallback((v) => pick(setW3, v), [pick]);
  const setK = useCallback((s, v) => setKs((K) => Object.assign({}, K, { [s]: v })), []);
  const setA = useCallback((s, v) => setAdd((A) => Object.assign({}, A, { [s]: v })), []);
  const onCell = useCallback((i, j) => setCell([i, j]), []);
  const resetK = useCallback(() => setKs({ "the dog sleeps": 3, "the cat sleeps": 2, "a dog runs": 2, "a cat sleeps": 1 }), []);

  const sent = w1 + " " + w2 + " " + w3;
  const sP = score(C0, sent);
  useEffect(() => {
    if (screen === 0 && !SEEN[sent]) setFound((F) => (F[sent] === undefined ? Object.assign({}, F, { [sent]: sP }) : F));
  }, [screen, sent, sP]);

  let question, stage, side;
  if (screen === 0) {
    const vals = Object.keys(found).map((k) => found[k]);
    question = (
      <>
        <b>Where we are:</b> lesson 2 defined a language model as next-token probabilities. Now we have to estimate them from text. The simplest way: assume the last k tokens are enough and count (slides 31–34). <b>The puzzle:</b> here is the deck's 8-sentence corpus. Build sentences it never contains. Can a model that only counts give an unseen sentence a nonzero probability? Find one that gets a score and one that gets exactly 0.
      </>
    );
    stage = (
      <>
        <Panel>
          <Label>build a sentence</Label>
          <Pills options={[["the", "the"], ["a", "a"]]} value={w1} onPick={pickW1} />
          <Pills options={[["dog", "dog"], ["cat", "cat"]]} value={w2} onPick={pickW2} />
          <Pills options={[["sleeps", "sleeps"], ["runs", "runs"]]} value={w3} onPick={pickW3} />
          <div style={{ fontFamily: MONO, fontSize: 12.5, color: SEEN[sent] ? THEME.muted : THEME.highlight }}>{SEEN[sent] ? "this sentence is in the corpus" : "★ not in the corpus"}</div>
        </Panel>
        <Trace C={C0} s={sent} />
      </>
    );
    side = (
      <>
        <Panel>
          <Label>training corpus (slide 35)</Label>
          <CorpusList corpus={BASE} />
          <Note>BOS and EOS are added at every sentence boundary: 24 words + 8 EOS = 32 prediction targets.</Note>
        </Panel>
        <Panel>
          <Goal met={vals.some((v) => v > 0)}>an unseen sentence with probability &gt; 0</Goal>
          <Goal met={vals.some((v) => v === 0)}>an unseen sentence with probability exactly 0</Goal>
        </Panel>
      </>
    );
  } else if (screen === 1) {
    const corpus = BASE.map((b) => [b[0], ks[b[0]]]);
    const C = countTable(corpus, order);
    const ctxs = Object.keys(C);
    let zeros = 0;
    for (const r of ctxs) for (const c of NEXT) if (!C[r][c]) zeros++;
    const C2 = countTable(corpus, 2);
    question = (
      <>
        <b>Claim to test:</b> an n-gram model conditions on the previous n − 1 tokens and nothing else (slides 32–34). Change how often each sentence occurs and watch the count table and the three test sentences. Does adding "the dog sleeps" change P(runs | cat)? Then switch to trigrams: what happens to the number of empty cells?
      </>
    );
    stage = (
      <>
        <Pills
          options={[
            [1, "unigram: 0 previous"],
            [2, "bigram: 1 previous"],
            [3, "trigram: 2 previous"],
          ]}
          value={order}
          onPick={setOrder}
        />
        <CountMatrix C={C} ctxs={ctxs} />
        <Note>Rows: contexts. Columns: next tokens. Shading: the share of the row. · = never seen.</Note>
      </>
    );
    side = (
      <>
        <Panel>
          <Label>occurrences in the corpus</Label>
          {BASE.map((b) => (
            <PlusMinus key={b[0]} label={b[0]} value={ks[b[0]]} onChange={setK} max={6} />
          ))}
          <Btn onClick={resetK}>back to slide 35</Btn>
        </Panel>
        <Panel>
          <Readout
            rows={TESTS.map((t) => [t, fmt(score(C2, t), 3), score(C2, t) === 0 ? THEME.warn : THEME.text]).concat([
              ["P(runs | cat), bigram", C2.cat ? fmt((C2.cat.runs || 0) / rowTotal(C2.cat), 3) : "no cat context"],
              ["contexts × next tokens", ctxs.length + " × " + NEXT.length],
              ["empty cells", String(zeros)],
            ])}
          />
        </Panel>
        <Note>Test sentences are scored with the bigram table. Longer contexts split the same counts across more rows, so more cells stay empty: "longer contexts need many more observations" (slide 46).</Note>
      </>
    );
  } else if (screen === 2) {
    const r = CTX2[cell[0]];
    const c = NEXT[cell[1]];
    const row = C0[r] || {};
    const t = rowTotal(row);
    const k = row[c] || 0;
    question = (
      <>
        <b>What you found:</b> each probability is a count divided by how often its context was followed by anything. A sentence is scored by multiplying its transitions, so it needs every pair to have been seen, not the whole sentence. "the dog runs" reuses familiar pieces and gets 0.150; "the cat runs" contains one unseen pair and gets 0 (slides 37–39, 44). Click any cell.
      </>
    );
    stage = (
      <>
        <CountMatrix C={C0} ctxs={CTX2} onCell={onCell} hi={cell} />
        <Eq size={14}>
          P(wₜ | wₜ₋₁) = count(wₜ₋₁ wₜ) / count(wₜ₋₁ followed by anything)
          <br />
          P({c} | {r}) = {k} / {t} = {t ? fmt(k / t, 3) : "undefined"}
        </Eq>
      </>
    );
    side = (
      <>
        <Panel>
          <Readout
            rows={[
              ["clicked cell", r + " → " + c],
              ["count", String(k)],
              ["row total (prediction opportunities)", String(t)],
              ["unigram P(sleeps) = 6 / 32", fmt(6 / 32, 4)],
            ]}
          />
        </Panel>
        <Note>These are maximum-likelihood estimates: the probabilities that make the training corpus most likely. Denominators count prediction opportunities, which is why "sleeps → [EOS]" is 6 / 6. No smoothing yet: unseen pairs get exactly 0.</Note>
      </>
    );
  } else {
    const corpus = BASE.concat(POOL.map((s) => [s, add[s]]));
    const C = countTable(corpus, 2);
    const pc = C.cat ? (C.cat.runs || 0) / rowTotal(C.cat) : 0;
    const pd = C.dog ? (C.dog.runs || 0) / rowTotal(C.dog) : 0;
    const sc = score(C, "the cat runs");
    const sd = score(C, "the dog runs");
    const nAdd = POOL.reduce((a, s) => a + add[s], 0);
    const g1 = Math.abs(pc - pd) < 1e-9;
    const g2 = sc >= sd - 1e-12 && sc > 0;
    question = (
      <>
        Dogs and cats should behave alike: we want the model to believe cats run as often as dogs do. You may only add sentences to the corpus. Make P(runs | cat) equal P(runs | dog), and make "the cat runs" at least as probable as "the dog runs". What is the fewest sentences that does it?
      </>
    );
    stage = (
      <>
        <Trace C={C} s="the cat runs" />
        <Trace C={C} s="the dog runs" />
      </>
    );
    side = (
      <>
        <Panel edge={g1 && g2 ? THEME.ok : THEME.grid}>
          <Label>add to the corpus</Label>
          {POOL.map((s) => (
            <PlusMinus key={s} label={s} value={add[s]} onChange={setA} max={4} />
          ))}
          <Readout
            rows={[
              ["P(runs | cat)", fmt(pc, 3)],
              ["P(runs | dog)", fmt(pd, 3)],
              ["sentences added", String(nAdd)],
            ]}
          />
          <Goal met={g1}>P(runs | cat) = P(runs | dog)</Goal>
          <Goal met={g2}>P(the cat runs) ≥ P(the dog runs)</Goal>
          <Hint hints={["Dog sentences change only the dog row. Which row must change for P(runs | cat) to move?", "Count the cat row now: sleeps 3, runs 0. How many 'cat runs' pairs make runs 2 out of 5, like dog? Then check which first word keeps 'the → cat' strong enough."]} />
        </Panel>
        <Note>Every context learns only from its own counts. Nothing tells the table that dog and cat are similar words (slide 45).</Note>
      </>
    );
  }

  return <LessonFrame screen={screen} onScreen={setScreen} title="Count, divide, multiply: the bigram model" question={question} stage={stage} side={side} />;
}
