import { useState, useCallback } from "react";
import { THEME, MONO } from "./common";

// Lesson 1 - Text becomes IDs (Tokenization, slides 6-13).
// Aha: a model only takes numbers, so text goes through a fixed lookup. Split it into tokens, give each token
// its ID from ONE shared vocabulary (the ID depends only on the word, never on where it sits), and turn the
// ID into a one-hot vector of length V. Multiplying that one-hot vector by the embedding table E (V × d)
// just picks the word's row, so the vocabulary size V sets both how long each input vector is (a sentence
// is N × V entries) and how many parameters the table costs (V × d).
//
// Numbers from the slides: the 3 × 3 pixel values [139, 136, 142, 119, 174, 155, 50, 118, 98] and
// score = w·x + b (slides 6-7); "We read books. We love books." → [we, read, books, we, love, books] (slide 8);
// V = 10,000 with IDs 0…9,999 and the vocabulary rows a 0, about 1, above 2, accept 3, across 4, action 5,
// after 6, again 7, book 125, books 126, learn 4,710, love 4,821, read 7,014, reading 7,015, we 9,820,
// year 9,993, yellow 9,994, yes 9,995, yesterday 9,996, young 9,997, zebra 9,998, zoo 9,999 (slides 9-11);
// 30,000 × 768 = 23,040,000 embedding parameters (slide 12); N = 6 tokens → N × V entries (slide 13).
// Teaching toys: the other ~55 words and their IDs (made up, kept in alphabetical order between the
// slide's rows), the values inside the embedding table E (seeded random stand-ins for learned values),
// the "position" and "first appearance" numbering schemes of the puzzle, and the challenge's numbers
// (9,216,000 parameters, 120,000 entries, 6 rows).

// @sim-start
const IDS_V = 10000;
// slide 9's rows, exactly
const IDS_SLIDE = {
  a: 0, about: 1, above: 2, accept: 3, across: 4, action: 5, after: 6, again: 7,
  book: 125, books: 126, learn: 4710, love: 4821, read: 7014, reading: 7015,
  we: 9820, year: 9993, yellow: 9994, yes: 9995, yesterday: 9996, young: 9997, zebra: 9998, zoo: 9999,
};
// teaching toys: more words, IDs made up but kept in the slide's alphabetical order
const IDS_TOY = {
  all: 12, am: 21, an: 24, and: 30, are: 41, at: 55, be: 70, big: 96, bird: 104,
  but: 410, buy: 433, cat: 520, cats: 521, class: 610, dog: 1180, drink: 1320, every: 1630, good: 2050,
  he: 2410, her: 2470, his: 2560, i: 2810, in: 2870, is: 2950, it: 2990, language: 4505,
  like: 4760, long: 4801,
  many: 4990, model: 5120, models: 5121, my: 5400, new: 5560, not: 5610, of: 5720, on: 5790, one: 5795,
  our: 5880, people: 6205,
  she: 7690, short: 7712, students: 8203, tea: 8581, text: 8640, the: 8702, they: 8744, this: 8790,
  to: 8850, today: 8862, tokens: 8870, very: 9301, want: 9620, was: 9650,
  what: 9844, with: 9907, word: 9930, words: 9931, write: 9961,
};
const IDS_VOCAB = Object.assign({}, IDS_SLIDE, IDS_TOY);
const IDS_SENT = "We read books. We love books.";

// slide 8: lowercase, drop punctuation, split on spaces
function idsTokenize(s) {
  return String(s)
    .toLowerCase()
    .replace(/[^\p{L}\p{N}'\s]+/gu, " ")
    .split(/\s+/)
    .filter(Boolean);
}
function idsLookup(tokens) {
  return tokens.map((t) => (Object.prototype.hasOwnProperty.call(IDS_VOCAB, t) ? IDS_VOCAB[t] : -1));
}
function idsStats(text, V) {
  const tokens = idsTokenize(text);
  const ids = idsLookup(tokens);
  const known = ids.filter((v) => v >= 0);
  const distinct = new Set(known).size;
  const N = tokens.length;
  return { tokens, ids, N, entries: N * V, ones: known.length, distinct, unknown: N - known.length };
}

// puzzle: three ways to number the words of several sentences
function idsNumber(scheme, sentences) {
  return sentences.map((toks) => {
    if (scheme === "pos") return toks.map((t, i) => i + 1);
    if (scheme === "first") {
      const seen = {};
      let k = 0;
      return toks.map((t) => {
        if (!Object.prototype.hasOwnProperty.call(seen, t)) seen[t] = ++k;
        return seen[t];
      });
    }
    return idsLookup(toks).map((v) => (v >= 0 ? v : null));
  });
}
// rule 1: a word always gets the same number; rule 2: two different words never share one
function idsCheck(sentences, nums) {
  const byWord = {};
  const byNum = {};
  sentences.forEach((toks, s) =>
    toks.forEach((t, i) => {
      const n = nums[s][i];
      if (n === null) return;
      (byWord[t] = byWord[t] || new Set()).add(n);
      (byNum[n] = byNum[n] || new Set()).add(t);
    })
  );
  const split = Object.keys(byWord).filter((w) => byWord[w].size > 1);
  const clash = Object.keys(byNum).filter((n) => byNum[n].size > 1);
  return { byWord, byNum, split, clash, same: split.length === 0, distinct: clash.length === 0 };
}

// challenge: d is slide 12's 768; reproduce three numbers with V and one sentence
const IDS_D = 768;
const IDS_GOAL = { params: 9216000, entries: 120000, rows: 6 };
function idsChallenge(text, V) {
  const st = idsStats(text, V);
  const params = V * IDS_D;
  const okParams = params === IDS_GOAL.params;
  const okEntries = st.entries === IDS_GOAL.entries;
  const okRows = st.distinct === IDS_GOAL.rows;
  const okKnown = st.N > 0 && st.unknown === 0;
  return { st, params, okParams, okEntries, okRows, okKnown, pass: okParams && okEntries && okRows && okKnown };
}
// @sim-end

function idsComma(n) {
  return String(n).replace(/\B(?=(\d{3})+(?!\d))/g, ",");
}
const IDS_PIX = [
  [139, 136, 142],
  [119, 174, 155],
  [50, 118, 98],
];
const IDS_MAXROWS = 10;
const IDS_MAXCHIPS = 24;

// columns of a one-hot picture: 0, …, each ID in the text, …, V − 1 ("…" hides runs of zeros, as on slide 11)
function idsColumns(ids, V) {
  const set = new Set([0, V - 1]);
  ids.forEach((v) => {
    if (v >= 0) set.add(v);
  });
  const sorted = Array.from(set).sort((a, b) => a - b);
  const cols = [];
  sorted.forEach((v, i) => {
    if (i > 0 && v - sorted[i - 1] > 1) cols.push(null);
    cols.push(v);
  });
  return cols;
}

// ---------- puzzle: one sentence as numbered chips ----------
function IdsNumRow({ label, toks, nums, chk }) {
  return (
    <div style={{ margin: "4px 0 10px" }}>
      <Label>{label}</Label>
      <div style={{ display: "flex", flexWrap: "wrap", gap: 6 }}>
        {toks.slice(0, IDS_MAXCHIPS).map((t, i) => {
          const n = nums[i];
          const bad = n !== null && (chk.byWord[t].size > 1 || chk.byNum[n].size > 1);
          return (
            <div
              key={i}
              style={{
                fontFamily: MONO,
                textAlign: "center",
                padding: "4px 9px",
                borderRadius: 4,
                minWidth: 46,
                border: "1px " + (bad ? "dashed " + THEME.warn : "solid " + THEME.grid),
                background: THEME.bg,
              }}
            >
              <div style={{ fontSize: 14, color: THEME.text }}>{t}</div>
              <div style={{ fontSize: 13, color: n === null ? THEME.muted : bad ? THEME.warn : THEME.highlight }}>
                {n === null ? "?" : idsComma(n)}
                {bad ? " ✕" : ""}
              </div>
            </div>
          );
        })}
        {toks.length > IDS_MAXCHIPS ? <div style={{ fontFamily: MONO, color: THEME.muted, alignSelf: "center" }}>+{toks.length - IDS_MAXCHIPS} more</div> : null}
      </div>
    </div>
  );
}
function IdsTextBox({ id, label, value, onChange }) {
  const change = useCallback((e) => onChange(e.target.value.slice(0, 200)), [onChange]);
  return (
    <label htmlFor={id} style={{ display: "block", margin: "2px 0 8px" }}>
      <Label>{label}</Label>
      <input
        id={id}
        type="text"
        value={value}
        onChange={change}
        spellCheck={false}
        style={{
          width: "100%",
          boxSizing: "border-box",
          fontFamily: MONO,
          fontSize: 14,
          padding: "6px 8px",
          borderRadius: 4,
          border: "1px solid " + THEME.accent,
          background: THEME.bg,
          color: THEME.text,
        }}
      />
    </label>
  );
}

// ---------- explore / challenge: clickable tokens with their IDs ----------
function IdsTokBtn({ tok, id, on, onPick }) {
  const click = useCallback(() => onPick(tok), [onPick, tok]);
  return (
    <button
      onClick={click}
      aria-pressed={!!on}
      style={{
        fontFamily: MONO,
        textAlign: "center",
        padding: "3px 9px",
        borderRadius: 4,
        minWidth: 46,
        cursor: "pointer",
        border: "1px " + (on ? "solid " + THEME.highlight : id < 0 ? "dashed " + THEME.warn : "solid " + THEME.grid),
        background: on ? rgba(THEME.highlight, 0.16) : THEME.bg,
        color: THEME.text,
      }}
    >
      <div style={{ fontSize: 14 }}>{tok}</div>
      <div style={{ fontSize: 12.5, color: id < 0 ? THEME.warn : THEME.accent }}>{id < 0 ? "? not listed" : idsComma(id)}</div>
    </button>
  );
}
function IdsTokens({ tokens, ids, hi, onPick }) {
  if (!tokens.length) return <Note>Type a sentence: there are no tokens yet.</Note>;
  return (
    <div style={{ display: "flex", flexWrap: "wrap", gap: 6, margin: "2px 0 8px" }}>
      {tokens.slice(0, IDS_MAXCHIPS).map((t, i) => (
        <IdsTokBtn key={i} tok={t} id={ids[i]} on={hi === t} onPick={onPick} />
      ))}
      {tokens.length > IDS_MAXCHIPS ? <div style={{ fontFamily: MONO, color: THEME.muted, alignSelf: "center" }}>+{tokens.length - IDS_MAXCHIPS} more</div> : null}
    </div>
  );
}

// ---------- the N × V one-hot matrix, slide 11/13 style ----------
function IdsOneHot({ tokens, ids, V, hi }) {
  const rows = tokens.slice(0, IDS_MAXROWS);
  const rids = ids.slice(0, IDS_MAXROWS);
  const cols = idsColumns(rids, V);
  const cw = 46;
  const gw = 18;
  const lw = 104;
  const rh = 24;
  const top = 44;
  const xs = [];
  let x = lw;
  cols.forEach((c) => {
    xs.push(x);
    x += c === null ? gw : cw;
  });
  const W = x + 6;
  const H = top + rows.length * rh + 6;
  const hiId = hi && Object.prototype.hasOwnProperty.call(IDS_VOCAB, hi) ? IDS_VOCAB[hi] : null;
  return (
    <svg viewBox={"0 0 " + W + " " + H} style={{ width: "100%", maxWidth: W * 1.2, display: "block", background: THEME.bg, border: "1px solid " + THEME.grid, borderRadius: 6 }} role="img" aria-label="one-hot matrix: one row per token, one column per vocabulary entry">
      <SvgLabel x={(lw + W) / 2} y={15} anchor="middle" color={THEME.muted} size={11}>
        {"one row = V = " + idsComma(V) + " entries (… hides zeros)"}
      </SvgLabel>
      <line x1={lw} x2={W - 6} y1={21} y2={21} stroke={THEME.muted} strokeWidth={0.8} />
      {cols.map((c, j) =>
        c === null ? (
          <text key={"h" + j} x={xs[j] + gw / 2} y={top - 8} textAnchor="middle" fontFamily={MONO} fontSize={11} fill={THEME.muted}>
            …
          </text>
        ) : (
          <text key={"h" + j} x={xs[j] + cw / 2} y={top - 8} textAnchor="middle" fontFamily={MONO} fontSize={11} fill={c === hiId ? THEME.highlight : THEME.text}>
            {c}
          </text>
        )
      )}
      {rows.map((t, i) => {
        const y = top + i * rh;
        const on = hi === t;
        return (
          <g key={"r" + i}>
            {on ? <rect x={2} y={y} width={W - 4} height={rh - 2} rx={3} fill={rgba(THEME.highlight, 0.12)} stroke={THEME.highlight} strokeWidth={1} /> : null}
            <text x={lw - 8} y={y + rh / 2 + 4} textAnchor="end" fontFamily={MONO} fontSize={12} fill={rids[i] < 0 ? THEME.warn : THEME.text}>
              {(t.length > 11 ? t.slice(0, 10) + "…" : t) + (rids[i] < 0 ? " ?" : "")}
            </text>
            {cols.map((c, j) => {
              if (c === null)
                return (
                  <text key={j} x={xs[j] + gw / 2} y={y + rh / 2 + 4} textAnchor="middle" fontFamily={MONO} fontSize={11} fill={THEME.muted}>
                    …
                  </text>
                );
              const one = rids[i] === c;
              return (
                <g key={j}>
                  {one ? <rect x={xs[j] + 6} y={y + 2} width={cw - 12} height={rh - 6} rx={3} fill={rgba(THEME.accent, 0.55)} /> : null}
                  <text x={xs[j] + cw / 2} y={y + rh / 2 + 4} textAnchor="middle" fontFamily={MONO} fontSize={12} fontWeight={one ? 700 : 400} fill={one ? THEME.text : THEME.muted}>
                    {one ? "1" : "0"}
                  </text>
                </g>
              );
            })}
          </g>
        );
      })}
    </svg>
  );
}

// ---------- name: h = xE picks one row of E ----------
const IDS_DSHOW = 4;
function idsEVal(id, c) {
  const r = mulberry32(hashStr("E" + id + ":" + c));
  return Math.round(randn(r) * 50) / 100;
}
function IdsLookup({ ids, pick }) {
  const known = ids.filter((v) => v >= 0);
  const cols = idsColumns(known, IDS_V);
  const rh = 22;
  const top = 26;
  const xId = 58;
  const xX = 96;
  const x0 = 130;
  const cw = 62;
  const W = x0 + IDS_DSHOW * cw + 40;
  const H = top + cols.length * rh + 8;
  return (
    <svg viewBox={"0 0 " + W + " " + H} style={{ width: "100%", maxWidth: W * 1.25, display: "block", background: THEME.bg, border: "1px solid " + THEME.grid, borderRadius: 6 }} role="img" aria-label="one-hot x next to the rows of the embedding table E">
      <text x={xId} y={16} textAnchor="end" fontFamily={MONO} fontSize={11} fill={THEME.muted}>
        ID
      </text>
      <text x={xX} y={16} textAnchor="middle" fontFamily={MONO} fontSize={11} fill={THEME.muted}>
        x
      </text>
      <text x={x0 + (IDS_DSHOW * cw) / 2} y={16} textAnchor="middle" fontFamily={MONO} fontSize={11} fill={THEME.muted}>
        row of E (first 4 of d values)
      </text>
      {cols.map((c, i) => {
        const y = top + i * rh;
        if (c === null)
          return (
            <text key={i} x={xX} y={y + 15} textAnchor="middle" fontFamily={MONO} fontSize={11} fill={THEME.muted}>
              ⋮
            </text>
          );
        const on = c === pick;
        const vals = [];
        for (let k = 0; k < IDS_DSHOW; k++) vals.push(idsEVal(c, k));
        return (
          <g key={i}>
            {on ? <rect x={2} y={y} width={W - 4} height={rh - 2} rx={3} fill={rgba(THEME.highlight, 0.14)} stroke={THEME.highlight} strokeWidth={1} /> : null}
            <text x={xId} y={y + 15} textAnchor="end" fontFamily={MONO} fontSize={11.5} fill={THEME.text}>
              {idsComma(c)}
            </text>
            <text x={xX} y={y + 15} textAnchor="middle" fontFamily={MONO} fontSize={12} fontWeight={on ? 700 : 400} fill={on ? THEME.highlight : THEME.muted}>
              {on ? "1" : "0"}
            </text>
            {vals.map((v, k) => (
              <text key={k} x={x0 + k * cw + cw / 2} y={y + 15} textAnchor="middle" fontFamily={MONO} fontSize={11.5} fill={on ? THEME.text : THEME.muted}>
                {fmt(v, 2)}
              </text>
            ))}
            <text x={x0 + IDS_DSHOW * cw + 8} y={y + 15} fontFamily={MONO} fontSize={11.5} fill={THEME.muted}>
              …
            </text>
          </g>
        );
      })}
    </svg>
  );
}

export default function IdsLesson({ initialScreen }) {
  const [screen, setScreen] = useState(initialScreen || 0);
  const [scheme, setScheme] = useState("pos");
  const [show2, setShow2] = useState(false);
  const [s2, setS2] = useState("I love reading books.");
  const [exText, setExText] = useState(IDS_SENT);
  const [exV, setExV] = useState(IDS_V);
  const [hi, setHi] = useState("books");
  const [nmPick, setNmPick] = useState("books");
  const [nmV, setNmV] = useState(30000);
  const [nmD, setNmD] = useState(768);
  const [chText, setChText] = useState(IDS_SENT);
  const [chV, setChV] = useState(IDS_V);
  const [chHi, setChHi] = useState(null);

  const pickHi = useCallback((t) => setHi((h) => (h === t ? null : t)), []);
  const pickNm = useCallback((t) => setNmPick(t), []);
  const pickCh = useCallback((t) => setChHi((h) => (h === t ? null : t)), []);

  let question, stage, side;
  if (screen === 0) {
    const sents = [idsTokenize(IDS_SENT)];
    if (show2) sents.push(idsTokenize(s2));
    const nums = idsNumber(scheme, sents);
    const chk = idsCheck(sents, nums);
    const probs = chk.split
      .slice(0, 3)
      .map((w) => w + " gets " + Array.from(chk.byWord[w]).join(" and "))
      .concat(chk.clash.slice(0, 3).map((n) => n + " is used for " + Array.from(chk.byNum[n]).join(" and ")));
    const unknown = show2 && nums[1].some((n) => n === null);
    question = (
      <>
        <b>Where we are:</b> this is the start of the deck, part 1, from text to tokens (slide 5). A model computes with numbers: slide 6 turns an image into 9 pixel values, and slide 7 feeds that vector to score = w·x + b. A sentence has no pixel values. <b>The puzzle:</b> give every word a number. Pick a numbering scheme and check the two rules on the right for sentence 1. Then add sentence 2. Which scheme still keeps both rules?
      </>
    );
    stage = (
      <>
        <Pills
          options={[
            ["pos", "number = position in the sentence"],
            ["first", "number = order of first appearance"],
            ["vocab", "number = row in one fixed word list"],
          ]}
          value={scheme}
          onPick={setScheme}
        />
        <Panel>
          <IdsNumRow label={"sentence 1 (slide 8): " + IDS_SENT} toks={sents[0]} nums={nums[0]} chk={chk} />
          {show2 ? <IdsNumRow label="sentence 2" toks={sents[1]} nums={nums[1]} chk={chk} /> : null}
          <Toggle label="add sentence 2" value={show2} onChange={setShow2} />
          {show2 ? <IdsTextBox id="t1-s2" label="edit sentence 2 (lowercased, punctuation dropped)" value={s2} onChange={setS2} /> : null}
        </Panel>
        <Note>✕ marks a word whose number breaks a rule. "?" means the word is not on this page's short list (lesson 2 asks what happens when a word is truly missing).</Note>
      </>
    );
    side = (
      <>
        <Panel edge={chk.same && chk.distinct && show2 ? THEME.ok : THEME.grid}>
          <Goal met={chk.same}>rule 1: the same word always gets the same number</Goal>
          <Goal met={chk.distinct}>rule 2: two different words never share a number</Goal>
          <Goal met={show2}>tested on more than one sentence</Goal>
          {probs.length ? (
            <div style={{ fontFamily: MONO, fontSize: 12.5, color: THEME.warn, marginTop: 6, lineHeight: 1.5 }}>
              {probs.map((p, i) => (
                <div key={i}>✕ {p}</div>
              ))}
            </div>
          ) : null}
          {unknown ? <Note>Some words in sentence 2 have no row on this page's list, so they are skipped.</Note> : null}
        </Panel>
        <Panel>
          <Label>slide 6: an image is already numbers</Label>
          <div style={{ maxWidth: 150 }}>
            <Matrix rowLabels={["", "", ""]} colLabels={["", "", ""]} values={IDS_PIX} shade={(v) => (v / 255) * 0.7} colorFor={() => THEME.text} text={(v) => String(v)} cell={44} rowW={4} colH={4} ariaLabel="3 by 3 grayscale pixel values from slide 6" />
          </div>
          <div style={{ fontFamily: MONO, fontSize: 12, color: THEME.muted, marginTop: 6, lineHeight: 1.5 }}>
            x = [139, 136, 142, 119, 174, 155, 50, 118, 98]
            <br />
            score = w·x + b (slide 7)
          </div>
        </Panel>
      </>
    );
  } else if (screen === 1) {
    const st = idsStats(exText, exV);
    question = (
      <>
        <b>Claim to test:</b> after the lookup, each ID becomes a one-hot vector: V entries, all 0 except a 1 at the word's ID (slide 11). A sentence of N tokens becomes N such vectors (slide 13). Edit the sentence and change the vocabulary size V. Which rows come out identical? How many entries does the six-word sentence need, and how many of them are 1?
      </>
    );
    stage = (
      <>
        <IdsTextBox id="t1-ex" label="sentence" value={exText} onChange={setExText} />
        <Label>tokens and their IDs (click one to follow its row)</Label>
        <IdsTokens tokens={st.tokens} ids={st.ids} hi={hi} onPick={pickHi} />
        {st.N ? <IdsOneHot tokens={st.tokens} ids={st.ids} V={exV} hi={hi} /> : null}
        {st.N > IDS_MAXROWS ? <Note>Showing the first {IDS_MAXROWS} of {st.N} rows.</Note> : null}
      </>
    );
    side = (
      <>
        <Panel>
          <Slider id="t1-v" label="vocabulary size V" value={exV} min={10000} max={50000} stepSize={1000} onChange={setExV} display={"V = " + idsComma(exV)} />
          <Readout
            rows={[
              ["tokens N", String(st.N)],
              ["entries per row", idsComma(exV)],
              ["entries N × V", idsComma(st.entries), THEME.highlight],
              ["entries that are 1", String(st.ones)],
              ["different words (different rows)", String(st.distinct)],
            ]}
          />
        </Panel>
        <Note>Result: the rows for “we” and for “books” are identical, because the ID depends only on the word, not its position (slide 10). Every row has V entries whatever the sentence length, so a sentence needs N × V entries and only N of them are 1. For slide 13's sentence that is 6 × 10,000 = 60,000.</Note>
        <Note>The vocabulary keeps its IDs when V grows: new words get IDs after 9,999. Words this page doesn't list show “?”.</Note>
      </>
    );
  } else if (screen === 2) {
    const toks = idsTokenize(IDS_SENT);
    const ids = idsLookup(toks);
    const pid = IDS_VOCAB[nmPick];
    const h = [];
    for (let k = 0; k < IDS_DSHOW; k++) h.push(fmt(idsEVal(pid, k), 2));
    question = (
      <>
        <b>What you found:</b> text becomes numbers by a fixed lookup. Split it into tokens, give each token its ID from one shared vocabulary, and write the ID as a one-hot vector x of length V. The model then multiplies x by an embedding table E with V rows and d columns. Because x has a single 1, h = xE is just the word's row of E (slide 12). Click a word and read its row.
      </>
    );
    stage = (
      <>
        <IdsTokens tokens={toks} ids={ids} hi={nmPick} onPick={pickNm} />
        <IdsLookup ids={ids} pick={pid} />
        <Eq size={12.5}>
          h = xE = row {idsComma(pid)} of E = [{h.join(", ")}, …] ({nmPick})
        </Eq>
        <Eq size={12.5}>
          1 split: text → [we, read, books, we, love, books] (slide 8)
          <br />
          2 look up: id(w) ∈ {"{"}0, …, V − 1{"}"}, same word → same ID (slides 9–10)
          <br />
          3 one-hot: x has V entries, x[id(w)] = 1, the rest 0 (slide 11)
          <br />
          4 embed: h = xE, E is V × d, so V × d parameters (slide 12)
          <br />
          5 sentence: N tokens → X is N × V, H = XE is N × d (slide 13)
        </Eq>
      </>
    );
    side = (
      <>
        <Panel>
          <Label>slide 12: what the table costs</Label>
          <Slider id="t1-nv" label="vocabulary size V" value={nmV} min={1000} max={100000} stepSize={1000} onChange={setNmV} display={"V = " + idsComma(nmV)} />
          <Slider id="t1-nd" label="embedding size d" value={nmD} min={32} max={1024} stepSize={32} onChange={setNmD} display={"d = " + nmD} />
          <Readout
            rows={[
              ["E has", idsComma(nmV) + " × " + nmD],
              ["parameters V × d", idsComma(nmV * nmD), THEME.highlight],
              ["slide 12 (30,000 × 768)", "23,040,000"],
            ]}
          />
        </Panel>
        <Note>Larger V means more embedding parameters, and other layers add more (slide 12). The values in E here are random stand-ins for learned ones; only 4 of the d columns are shown.</Note>
        <Note>Next lesson: a word vocabulary is fixed. What happens to “unstoppable”, or the typo “unstopabel”, if it is not in the list (slide 14)? Lesson 2 compares words and characters as units.</Note>
      </>
    );
  } else {
    const r = idsChallenge(chText, chV);
    const words = Object.keys(IDS_VOCAB).sort();
    question = (
      <>
        A model uses d = {IDS_D}, as on slide 12. Three numbers are known: its embedding table E has {idsComma(IDS_GOAL.params)} parameters; for one input sentence the one-hot matrix X has {idsComma(IDS_GOAL.entries)} entries; and looking that sentence up reads {IDS_GOAL.rows} different rows of E. Set the vocabulary size V and write a sentence that reproduces all three.
      </>
    );
    stage = (
      <>
        <IdsTextBox id="t1-ch" label="your sentence" value={chText} onChange={setChText} />
        <IdsTokens tokens={r.st.tokens} ids={r.st.ids} hi={chHi} onPick={pickCh} />
        {r.st.N ? <IdsOneHot tokens={r.st.tokens} ids={r.st.ids} V={chV} hi={chHi} /> : null}
        {r.st.N > IDS_MAXROWS ? <Note>Showing the first {IDS_MAXROWS} of {r.st.N} rows.</Note> : null}
        <Label>words this page knows</Label>
        <div style={{ fontFamily: MONO, fontSize: 12, color: THEME.muted, lineHeight: 1.6 }}>{words.join(" · ")}</div>
      </>
    );
    side = (
      <>
        <Panel edge={r.pass ? THEME.ok : THEME.grid}>
          <Slider id="t1-cv" label="vocabulary size V" value={chV} min={10000} max={50000} stepSize={1000} onChange={setChV} display={"V = " + idsComma(chV)} />
          <Readout
            rows={[
              ["E parameters V × d", idsComma(r.params), r.okParams ? THEME.ok : THEME.text],
              ["X entries N × V", idsComma(r.st.entries), r.okEntries ? THEME.ok : THEME.text],
              ["different rows of E read", String(r.st.distinct), r.okRows ? THEME.ok : THEME.text],
            ]}
          />
          <Goal met={r.okParams}>E has {idsComma(IDS_GOAL.params)} parameters</Goal>
          <Goal met={r.okEntries}>X has {idsComma(IDS_GOAL.entries)} entries</Goal>
          <Goal met={r.okRows}>the lookup reads {IDS_GOAL.rows} different rows of E</Goal>
          <Goal met={r.okKnown}>every token has an ID</Goal>
          <Hint
            hints={[
              "Only V changes the parameter count, since d is fixed at 768. Which V gives 9,216,000? Then X = N × V tells you how many tokens N the sentence needs.",
              "V = 12,000 gives N = 10 tokens. A repeated word is a new token, so it adds a row to X, but it reads the same row of E. You need 10 tokens made of only 6 different words.",
            ]}
          />
        </Panel>
        <Note>Notice which number the sentence cannot change: E's size depends on V and d only, never on the text.</Note>
      </>
    );
  }

  return <LessonFrame screen={screen} onScreen={setScreen} title="Text becomes IDs" question={question} stage={stage} side={side} />;
}
