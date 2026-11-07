import { useState, useMemo, useCallback } from "react";
import { THEME, MONO } from "./common";

// Lesson 2 - Similar contexts, related words (Word Embedding, slides 2-5, 12-13, 19).
// Aha: count which words appear around each word; two words with the same counts get the same vector,
// even if one of them is a word you have never seen. Related is not the same as synonymous.

// @sim-start
const MAZ = [
  { s: "سگ گوشت خِرنه", added: false },
  { s: "بامشی گوشت خِرنه", added: false },
  { s: "سگ بازی کِنه", added: false },
  { s: "بامشی بازی کِنه", added: false },
  { s: "بامشی میو کِنه", added: true },
  { s: "سگ واق کِنه", added: true },
];
const CORPUS = [
  "تیم ما فوتبال بازی کرد",
  "تیم آنها بسکتبال بازی کرد",
  "فوتبال تیم ملی برد",
  "بسکتبال تیم ملی باخت",
  "مجلس قانون جدید را تصویب کرد",
  "دولت قانون جدید را اجرا کرد",
  "فیلم خیلی خوب بود",
  "فیلم خیلی بد بود",
  "غذا خوب بود",
  "غذا بد بود",
];
const NEW_WORD = "والیبال";
const CHIPS = ["تیم", "ما", "آنها", "ملی", "بازی", "کرد", "برد", "باخت", "قانون", "جدید", "را", "تصویب", "خوب", "بود", "فیلم", NEW_WORD];
const SPORTS = ["فوتبال", "بسکتبال"];
const MAX_LEN = 5;
const COS_GOAL = 0.85;

// counts[w][c] = how often c appears within `win` words of w (win 0 = anywhere in the sentence)
function countContexts(sents, win) {
  const C = {};
  for (const s of sents) {
    const t = s.split(" ").filter(Boolean);
    t.forEach((w, i) => {
      if (!C[w]) C[w] = {};
      for (let j = 0; j < t.length; j++) {
        if (j === i) continue;
        if (win > 0 && Math.abs(j - i) > win) continue;
        C[w][t[j]] = (C[w][t[j]] || 0) + 1;
      }
    });
  }
  return C;
}
function cosRow(a, b) {
  let d = 0;
  let na = 0;
  let nb = 0;
  for (const k in a) {
    na += a[k] * a[k];
    if (b[k]) d += a[k] * b[k];
  }
  for (const k in b) nb += b[k] * b[k];
  return na && nb ? d / Math.sqrt(na * nb) : 0;
}
function neighbours(C, w) {
  if (!C[w]) return [];
  return Object.keys(C)
    .filter((x) => x !== w)
    .map((x) => [x, cosRow(C[w], C[x])])
    .sort((p, q) => q[1] - p[1]);
}
// @sim-end

function SentCard({ i, on, text, added, onPick }) {
  const click = useCallback(() => onPick(i), [onPick, i]);
  return (
    <button
      onClick={click}
      aria-pressed={on}
      style={{
        display: "flex",
        justifyContent: "space-between",
        alignItems: "center",
        gap: 8,
        width: "100%",
        textAlign: "right",
        cursor: "pointer",
        padding: "5px 10px",
        borderRadius: 4,
        border: "1px " + (on ? "solid " + THEME.accent : "dashed " + THEME.grid),
        background: on ? rgba(THEME.accent, 0.1) : THEME.bg,
        color: on ? THEME.text : THEME.muted,
        marginBottom: 5,
      }}
    >
      <span style={{ fontFamily: MONO, fontSize: 11 }}>
        {on ? "■ in corpus" : "□ left out"}
        {added ? " · added" : ""}
      </span>
      <span lang="fa" dir="rtl" style={{ fontFamily: FA, fontSize: 16, textDecoration: on ? "none" : "line-through" }}>
        {text}
      </span>
    </button>
  );
}
function ContextChips({ row }) {
  const keys = Object.keys(row || {});
  if (!keys.length) return <Note>no contexts yet</Note>;
  return (
    <div dir="rtl" style={{ display: "flex", flexWrap: "wrap", gap: 6 }}>
      {keys.map((k) => (
        <span key={k} style={{ border: "1px solid " + THEME.grid, borderRadius: 4, padding: "1px 8px", fontFamily: FA, fontSize: 14.5, background: THEME.bg }}>
          <span lang="fa">{k}</span>
          <span style={{ fontFamily: MONO, fontSize: 11, color: THEME.highlight, marginRight: 4 }}>×{row[k]}</span>
        </span>
      ))}
    </div>
  );
}
function PairTable({ C, a, b }) {
  const cols = Array.from(new Set(Object.keys(C[a] || {}).concat(Object.keys(C[b] || {}))));
  const ra = cols.map((c) => (C[a] && C[a][c]) || 0);
  const rb = cols.map((c) => (C[b] && C[b][c]) || 0);
  const pr = cols.map((c, i) => ra[i] * rb[i]);
  const mx = Math.max(1, ...ra, ...rb);
  const shade = useCallback((v, i) => (i === 2 ? (v ? 0.6 : 0) : v / mx), [mx]);
  const colorFor = useCallback((v, i) => (i === 0 ? THEME.highlight : i === 1 ? THEME.third : THEME.accent), []);
  const text = useCallback((v) => String(v), []);
  return (
    <Matrix
      rowLabels={[a, b, "product"]}
      colLabels={cols}
      values={[ra, rb, pr]}
      shade={shade}
      colorFor={colorFor}
      text={text}
      faCols
      faRows
      cell={cols.length > 9 ? 36 : 44}
      rowW={78}
      colH={34}
      ariaLabel="context counts of the two words and their product"
    />
  );
}

export default function ContextLesson({ initialScreen }) {
  const [screen, setScreen] = useState(initialScreen || 0);
  const [onM, setOnM] = useState([true, true, true, true, false, false]);
  const [win, setWin] = useState(2);
  const [target, setTarget] = useState("خوب");
  const [pa, setPa] = useState("فوتبال");
  const [pb, setPb] = useState("بسکتبال");
  const [pickA, setPickA] = useState(true);
  const [built, setBuilt] = useState([]);

  const flipM = useCallback((i) => setOnM((v) => v.map((x, j) => (j === i ? !x : x))), []);
  const pickWord = useCallback(
    (w) => {
      if (pickA) setPa(w);
      else setPb(w);
      setPickA((v) => !v);
    },
    [pickA]
  );
  const addChip = useCallback((w) => setBuilt((b) => (b.indexOf(w) >= 0 || b.length >= MAX_LEN ? b : b.concat([w]))), []);
  const undo = useCallback(() => setBuilt((b) => b.slice(0, -1)), []);
  const clear = useCallback(() => setBuilt([]), []);

  const Cm = useMemo(() => countContexts(MAZ.filter((m, i) => onM[i]).map((m) => m.s), 0), [onM]);
  const Ce = useMemo(() => countContexts(CORPUS, win), [win]);
  const C2 = useMemo(() => countContexts(CORPUS, 2), []);
  const Cc = useMemo(() => countContexts(built.length ? CORPUS.concat([built.join(" ")]) : CORPUS, 2), [built]);

  let question, stage, side;
  if (screen === 0) {
    const others = ["سگ", "گوشت", "بازی", "خِرنه", "کِنه"];
    const rows = others.map((w) => ({ label: w, fa: true, value: Cm["بامشی"] && Cm[w] ? cosRow(Cm["بامشی"], Cm[w]) : 0, color: w === "سگ" ? THEME.highlight : THEME.accent, mark: w === "سگ" ? "◆" : "" }));
    question = (
      <>
        <b>Where we are:</b> lesson 1 showed that one-hot vectors know nothing about which words are related. Before building better vectors, the deck asks what "related" could even mean for a computer (slides 2–5). <b>The puzzle:</b> these sentences are Mazandarani. What might <Fa>بامشی</Fa> mean, and why? Turn sentences on and off and watch which word ends up with the same contexts as <Fa>بامشی</Fa>.
      </>
    );
    stage = (
      <>
        <Panel>
          {MAZ.map((m, i) => (
            <SentCard key={i} i={i} on={onM[i]} text={m.s} added={m.added} onPick={flipM} />
          ))}
        </Panel>
        <Label>cosine between the context counts of بامشی and each word</Label>
        <Bars rows={rows} range={[0, 1]} labelW={80} ariaLabel="similarity of بامشی to other words" />
      </>
    );
    side = (
      <>
        <Panel>
          <Label>contexts of بامشی (words in the same sentence)</Label>
          <ContextChips row={Cm["بامشی"]} />
          <div style={{ height: 8 }} />
          <Label>contexts of سگ (dog)</Label>
          <ContextChips row={Cm["سگ"]} />
        </Panel>
        <Note>The last two sentences are made up for this lab. Add them. Does the evidence still say بامشی is a dog? Similar contexts are evidence of relatedness, not of identity (slide 4).</Note>
      </>
    );
  } else if (screen === 1) {
    const nb = neighbours(Ce, target).slice(0, 6);
    const words = Object.keys(Ce);
    question = (
      <>
        <b>Claim to test:</b> words used in similar contexts tend to have related meanings, which is the distributional hypothesis (slide 5). Here the corpus is ten Persian sentences. Each word's vector is its row of context counts, and similarity is the cosine of two rows. Pick a word and read its nearest neighbours. Start with <Fa>خوب</Fa>: is its nearest word a synonym? Then change the window. Which neighbours survive?
      </>
    );
    stage = (
      <>
        <Panel>
          <div dir="rtl" style={{ display: "flex", flexDirection: "column", gap: 2 }}>
            {CORPUS.map((s, i) => (
              <div key={i} lang="fa" style={{ fontFamily: FA, fontSize: 15.5, color: s.split(" ").indexOf(target) >= 0 ? THEME.text : THEME.muted }}>
                {s.split(" ").map((t, j) => (
                  <span key={j} style={{ color: t === target ? THEME.highlight : undefined, fontWeight: t === target ? 600 : 400 }}>
                    {t}{" "}
                  </span>
                ))}
              </div>
            ))}
          </div>
        </Panel>
        <Label>nearest neighbours of «{target}» by cosine</Label>
        <Bars rows={nb.map((p) => ({ label: p[0], fa: true, value: p[1], color: THEME.accent }))} range={[0, 1]} labelW={80} ariaLabel="nearest neighbours" />
      </>
    );
    side = (
      <>
        <Panel>
          <Label>window: which neighbours count as context</Label>
          <Pills
            options={[
              [1, "±1 word"],
              [2, "±2 words"],
              [0, "whole sentence"],
            ]}
            value={win}
            onPick={setWin}
          />
          <Label>target word</Label>
          <div dir="rtl" style={{ display: "flex", flexWrap: "wrap", gap: 5 }}>
            {words.map((w) => (
              <Chip key={w} value={w} on={w === target} onPick={setTarget}>
                {w}
              </Chip>
            ))}
          </div>
        </Panel>
        <Panel>
          <Label>context row of «{target}»</Label>
          <ContextChips row={Ce[target]} />
        </Panel>
        <Note>خوب and بد share every context, so their rows are identical. Related words are not always synonyms: good and bad share contexts too (slide 5).</Note>
      </>
    );
  } else if (screen === 2) {
    const c = cosRow(C2[pa] || {}, C2[pb] || {});
    let dd = 0;
    let na = 0;
    let nbb = 0;
    for (const k in C2[pa] || {}) {
      na += C2[pa][k] * C2[pa][k];
      if (C2[pb] && C2[pb][k]) dd += C2[pa][k] * C2[pb][k];
    }
    for (const k in C2[pb] || {}) nbb += C2[pb][k] * C2[pb][k];
    question = (
      <>
        <b>What you found:</b> counting contexts already gives every word a vector, and words that share contexts get similar vectors. That is how <Fa>بامشی</Fa> lands next to <Fa>سگ</Fa> and <Fa>فوتبال</Fa> next to <Fa>بسکتبال</Fa>. Formally, each word is a row of the co-occurrence matrix X, and similarity is the cosine of two rows. Pick two words (they alternate as A and B) to see the computation.
      </>
    );
    stage = (
      <>
        <PairTable C={C2} a={pa} b={pb} />
        <Eq size={12.5}>
          X[w, c] = number of times c appears within ±2 words of w
          <br />
          cos(u, v) = u·v / (‖u‖ ‖v‖) = {fmt(dd, 0)} / ({fmt(Math.sqrt(na), 2)} × {fmt(Math.sqrt(nbb), 2)}) = {fmt(c, 3)}
        </Eq>
      </>
    );
    side = (
      <>
        <Panel>
          <Label>pick word {pickA ? "A" : "B"}</Label>
          <div dir="rtl" style={{ display: "flex", flexWrap: "wrap", gap: 5 }}>
            {Object.keys(C2).map((w) => (
              <Chip key={w} value={w} on={w === pa || w === pb} onPick={pickWord}>
                {w}
              </Chip>
            ))}
          </div>
        </Panel>
        <Note>
          Slides 5, 13, 19. Count-based methods build this matrix; LSA then compresses it, often with SVD, so the rows become short dense vectors (slide 13). Cosine ignores vector length, so a frequent word and a rare word with the same context pattern still score 1. Next lesson: learn the vectors by <i>predicting</i> contexts instead of counting them (Word2Vec).
        </Note>
      </>
    );
  } else {
    const nb = built.indexOf(NEW_WORD) >= 0 ? neighbours(Cc, NEW_WORD) : [];
    const top = nb[0];
    const okIn = built.indexOf(NEW_WORD) >= 0;
    const okNear = okIn && top && SPORTS.indexOf(top[0]) >= 0;
    const okCos = okNear && top[1] >= COS_GOAL - 1e-9;
    question = (
      <>
        A new word, <Fa>والیبال</Fa> (volleyball), shows up in one new sentence. Build that sentence from the chips, at most {MAX_LEN} words, and it joins the ten-sentence corpus (window ±2). Goal: <Fa>والیبال</Fa>'s nearest neighbour is <Fa>فوتبال</Fa> or <Fa>بسکتبال</Fa>, with cosine at least {COS_GOAL}.
      </>
    );
    stage = (
      <>
        <Panel>
          <Label>your sentence (click chips in reading order)</Label>
          <div lang="fa" dir="rtl" style={{ fontFamily: FA, fontSize: 20, minHeight: 36, padding: "4px 8px", border: "1px dashed " + THEME.grid, borderRadius: 4, background: THEME.bg }}>
            {built.length ? built.join(" ") : <span style={{ color: THEME.muted, fontSize: 14 }}>…</span>}
          </div>
          <div style={{ display: "flex", gap: 6, marginTop: 8 }}>
            <Btn onClick={undo} disabled={!built.length}>
              remove last
            </Btn>
            <Btn onClick={clear} disabled={!built.length}>
              clear
            </Btn>
          </div>
        </Panel>
        <Label>nearest neighbours of «والیبال»</Label>
        {okIn ? <Bars rows={nb.slice(0, 6).map((p) => ({ label: p[0], fa: true, value: p[1], color: SPORTS.indexOf(p[0]) >= 0 ? THEME.accent : THEME.highlight, mark: SPORTS.indexOf(p[0]) >= 0 ? "●" : "" }))} range={[0, 1]} labelW={80} ariaLabel="nearest neighbours of the new word" /> : <Note>Put والیبال in the sentence first.</Note>}
      </>
    );
    side = (
      <>
        <Panel edge={okCos ? THEME.ok : THEME.grid}>
          <div dir="rtl" style={{ display: "flex", flexWrap: "wrap", gap: 5, marginBottom: 8 }}>
            {CHIPS.map((w) => (
              <Chip key={w} value={w} on={built.indexOf(w) >= 0} onPick={addChip} disabled={built.indexOf(w) < 0 && built.length >= MAX_LEN}>
                {w}
              </Chip>
            ))}
          </div>
          <Readout
            rows={[
              ["words used", built.length + " / " + MAX_LEN],
              ["nearest neighbour", top ? top[0] + " (" + fmt(top[1], 2) + ")" : "—"],
            ]}
          />
          <Goal met={okNear}>nearest neighbour is فوتبال or بسکتبال</Goal>
          <Goal met={okCos}>… with cosine ≥ {COS_GOAL}</Goal>
          <Hint
            hints={[
              "A word's vector is its context counts. Which words sit within two places of فوتبال in the corpus?",
              "Copy the neighbourhood of فوتبال in «تیم ما فوتبال بازی کرد»: two words before والیبال and two after. Watch out: a sentence like «والیبال تیم ملی برد» makes it identical to برد instead.",
            ]}
          />
        </Panel>
        <Note>One sentence of evidence is enough here, because the counts are all the model has. With real corpora, a word needs many sentences before its row is reliable.</Note>
      </>
    );
  }

  return <LessonFrame screen={screen} onScreen={setScreen} title="Similar contexts, related words" question={question} stage={stage} side={side} />;
}
