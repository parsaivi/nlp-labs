import { useState, useCallback } from "react";
import { THEME, MONO } from "./common";

// Lesson 2 - Words or characters? (Tokenization, slides 14-21).
// Aha: the choice of unit is a trade between vocabulary and length. A fixed word vocabulary gives short
// sequences, but every form it lacks becomes [UNK], so different unknown words reach the model as the same
// ID. Character tokens cover every word built from known characters with a tiny V, but the same text is
// k times longer, and full attention costs O(n²d), so it does k² times the work. Subword pieces sit
// between the two: shared fragments keep V small without making n long.
//
// Numbers from the slides: V = 10,000 with IDs 0…9,999 and the rows a 0, about 1, above 2, accept 3,
// across 4, action 5, after 6, again 7, book 125, books 126, learn 4,710, love 4,821, read 7,014,
// reading 7,015, we 9,820, year 9,993, yellow 9,994, yes 9,995, yesterday 9,996, young 9,997, zebra 9,998,
// zoo 9,999 (slide 9); d = 768 (slide 12); unstoppable, unstoppability and the typo unstopabel (slide 14);
// [UNK] = 10,000, V = 10,001 and "we are unstoppable" → [we, are, [UNK]] (slide 15); unstoppable → 11
// character tokens (slide 16); O(n²d), n → kn gives k² × cost, 4× tokens → 16× attention (slide 17);
// the word-vs-character table (slide 18); "we read unstoppable" → [we] [read] [un] [stop] [p] [able]
// (slide 20); the two questions, training and inference (slide 21).
// Teaching toys: the other words and their IDs (lesson 1's toy list, plus stop = 8,150, kept in the
// slide's alphabetical order), the character vocabulary of 27 entries (a–z and [UNK]), not counting spaces
// as tokens, the challenge's extra words (unreadable, rereading, relearn, unlike, rewrite) and its budgets
// (n ≤ 20 tokens, at most 5 new entries).

// @sim-start
const UNITS_SLIDE = {
  a: 0, about: 1, above: 2, accept: 3, across: 4, action: 5, after: 6, again: 7,
  book: 125, books: 126, learn: 4710, love: 4821, read: 7014, reading: 7015,
  we: 9820, year: 9993, yellow: 9994, yes: 9995, yesterday: 9996, young: 9997, zebra: 9998, zoo: 9999,
};
// teaching toys: lesson 1's extra words (same IDs) plus "stop"; IDs made up, in the slide's alphabetical order
const UNITS_TOY = {
  all: 12, am: 21, an: 24, and: 30, are: 41, at: 55, be: 70, big: 96, bird: 104,
  but: 410, buy: 433, cat: 520, cats: 521, class: 610, dog: 1180, drink: 1320, every: 1630, good: 2050,
  he: 2410, her: 2470, his: 2560, i: 2810, in: 2870, is: 2950, it: 2990, language: 4505,
  like: 4760, long: 4801,
  many: 4990, model: 5120, models: 5121, my: 5400, new: 5560, not: 5610, of: 5720, on: 5790, one: 5795,
  our: 5880, people: 6205,
  she: 7690, short: 7712, stop: 8150, students: 8203, tea: 8581, text: 8640, the: 8702, they: 8744, this: 8790,
  to: 8850, today: 8862, tokens: 8870, very: 9301, want: 9620, was: 9650,
  what: 9844, with: 9907, word: 9930, words: 9931, write: 9961,
};
const UNITS_VOCAB = Object.assign({}, UNITS_SLIDE, UNITS_TOY);
const UNITS_UNK = 10000;
const UNITS_V = 10001;
const UNITS_ALPHA = "abcdefghijklmnopqrstuvwxyz";
const UNITS_VC = 27;
const UNITS_D = 768;

// lowercase, drop punctuation, split on spaces (as in lesson 1); spaces only separate words here
function unitsWords(text) {
  return String(text)
    .toLowerCase()
    .replace(/[^\p{L}\p{M}\p{N}'‌\s]+/gu, " ")
    .split(/\s+/)
    .filter(Boolean);
}
function unitsHas(w) {
  return Object.prototype.hasOwnProperty.call(UNITS_VOCAB, w);
}
function unitsWordTok(text) {
  return unitsWords(text).map((w) => (unitsHas(w) ? { s: w, id: UNITS_VOCAB[w], unk: false } : { s: "[UNK]", id: UNITS_UNK, unk: true, orig: w }));
}
function unitsCharTok(text) {
  const out = [];
  for (const w of unitsWords(text))
    for (const c of Array.from(w)) {
      const k = UNITS_ALPHA.indexOf(c);
      out.push(k >= 0 ? { s: c, id: k, unk: false } : { s: "[UNK]", id: 26, unk: true, orig: c });
    }
  return out;
}

// challenge: cut each word into pieces; a piece that is a vocabulary word is free, any other distinct piece
// is one new vocabulary entry
const UNITS_CH = ["unstoppable", "unstopabel", "unreadable", "rereading", "relearn", "unlike", "rewrite"];
const UNITS_NMAX = 20;
const UNITS_MMAX = 5;
function unitsSplit(word, cuts) {
  const out = [];
  let s = 0;
  for (let i = 1; i < word.length; i++)
    if (cuts[i - 1]) {
      out.push(word.slice(s, i));
      s = i;
    }
  out.push(word.slice(s));
  return out;
}
function unitsScore(cutsAll) {
  let n = 0;
  const fresh = [];
  const per = UNITS_CH.map((w, i) => {
    const p = unitsSplit(w, cutsAll[i]);
    n += p.length;
    for (const x of p) if (!unitsHas(x) && fresh.indexOf(x) < 0) fresh.push(x);
    return p;
  });
  return { per, n, fresh, okN: n <= UNITS_NMAX, okV: fresh.length <= UNITS_MMAX };
}
function unitsCutsFill(v) {
  return UNITS_CH.map((w) => new Array(w.length - 1).fill(v));
}
// cuts that produce the given pieces (used by the sim check)
function unitsCutsOf(pieceLists) {
  return pieceLists.map((p) => {
    const c = [];
    for (let i = 0; i < p.length; i++) {
      for (let j = 1; j < p[i].length; j++) c.push(false);
      if (i < p.length - 1) c.push(true);
    }
    return c;
  });
}
// @sim-end

const UNITS_FA_RE = /[؀-ۿ]/;
const UNITS_PRESETS_B = [
  ["we read books", "we read books"],
  ["we love books", "we love books"],
  ["we are unstoppability", "we are unstoppability"],
  ["we are unstopabel", "we are unstopabel"],
];
const UNITS_PRESETS_X = [
  ["unstoppable", "unstoppable (slide 16)"],
  ["we read unstoppable", "we read unstoppable (slide 20)"],
  ["we read books we love books", "we read books we love books"],
  ["the students write new words with short tokens every day", "a longer sentence"],
  ["من کتاب می‌خوانم", <Fa key="fa">من کتاب می‌خوانم</Fa>],
];

function unitsNum(v) {
  return Number(v).toLocaleString("en-US");
}
function unitsSameIds(x, y) {
  if (x.length !== y.length) return false;
  for (let i = 0; i < x.length; i++) if (x[i].id !== y[i].id) return false;
  return true;
}

function UnitsText({ id, value, onChange, label }) {
  const change = useCallback((e) => onChange(e.target.value.slice(0, 140)), [onChange]);
  return (
    <label htmlFor={id} style={{ display: "block", margin: "2px 0 6px" }}>
      {label ? <Label>{label}</Label> : null}
      <input
        id={id}
        value={value}
        onChange={change}
        dir="auto"
        spellCheck={false}
        style={{ width: "100%", boxSizing: "border-box", fontFamily: UNITS_FA_RE.test(value) ? FA : MONO, fontSize: 16, padding: "6px 10px", background: THEME.bg, color: THEME.text, border: "1px solid " + THEME.accent, borderRadius: 4 }}
      />
    </label>
  );
}
function UnitsTok({ t, color, showId }) {
  const c = t.unk ? THEME.warn : color;
  return (
    <span
      title={t.unk ? "not in the vocabulary: " + t.orig : undefined}
      style={{ display: "inline-flex", flexDirection: "column", alignItems: "center", border: "1px " + (t.unk ? "dashed " : "solid ") + c, background: rgba(c, 0.13), borderRadius: 4, padding: "1px 6px", minWidth: 16 }}
    >
      <span style={{ fontFamily: MONO, fontSize: 13.5, color: THEME.text, lineHeight: 1.35 }}>{t.s}</span>
      {showId ? <span style={{ fontFamily: MONO, fontSize: 10.5, color: t.unk ? THEME.warn : THEME.muted, lineHeight: 1.2 }}>{t.id}</span> : null}
    </span>
  );
}
function UnitsStrip({ tokens, color, showIds, max }) {
  const cap = max || 90;
  if (!tokens.length) return <div style={{ fontFamily: MONO, fontSize: 12.5, color: THEME.muted, padding: "4px 0" }}>(no tokens: type some text)</div>;
  return (
    <div style={{ display: "flex", flexWrap: "wrap", gap: 4, margin: "2px 0 8px" }}>
      {tokens.slice(0, cap).map((t, i) => (
        <UnitsTok key={i} t={t} color={color} showId={showIds} />
      ))}
      {tokens.length > cap ? <span style={{ fontFamily: MONO, fontSize: 12, color: THEME.muted, alignSelf: "center" }}>+{tokens.length - cap} more</span> : null}
    </div>
  );
}
// attention compares every token with every token: an n × n square of pairs, area n²
function UnitsCost({ nw, nc }) {
  const S = 210;
  const x0 = 16;
  const y0 = 14;
  const big = Math.max(nc, 1);
  const sw = (S * nw) / big;
  const lines = [];
  if (nc > 0 && nc <= 40)
    for (let i = 1; i < nc; i++) {
      const v = (S * i) / nc;
      lines.push(<line key={"v" + i} x1={x0 + v} x2={x0 + v} y1={y0} y2={y0 + S} stroke={THEME.highlight} strokeWidth={0.5} opacity={0.35} />);
      lines.push(<line key={"h" + i} y1={y0 + v} y2={y0 + v} x1={x0} x2={x0 + S} stroke={THEME.highlight} strokeWidth={0.5} opacity={0.35} />);
    }
  const ratio = nw > 0 ? (nc * nc) / (nw * nw) : NaN;
  return (
    <svg viewBox="0 0 600 240" role="img" aria-label="attention pairs: word tokens versus character tokens" style={{ width: "100%", display: "block", background: THEME.bg, border: "1px solid " + THEME.grid, borderRadius: 6 }}>
      <rect x={x0} y={y0} width={S} height={S} fill={rgba(THEME.highlight, 0.12)} stroke={THEME.highlight} strokeWidth={1.5} strokeDasharray="5 3" />
      {lines}
      {nw > 0 ? <rect x={x0} y={y0} width={sw} height={sw} fill={rgba(THEME.third, 0.45)} stroke={THEME.third} strokeWidth={2} /> : null}
      <SvgLabel x={250} y={44} color={THEME.highlight} size={13}>
        ▭ dashed: character tokens
      </SvgLabel>
      <SvgLabel x={250} y={64} color={THEME.text} size={13}>
        {"n = " + nc + " → n² = " + unitsNum(nc * nc) + " query–key pairs"}
      </SvgLabel>
      <SvgLabel x={250} y={102} color={THEME.third} size={13}>
        ■ solid: word tokens
      </SvgLabel>
      <SvgLabel x={250} y={122} color={THEME.text} size={13}>
        {"n = " + nw + " → n² = " + unitsNum(nw * nw) + " query–key pairs"}
      </SvgLabel>
      <SvgLabel x={250} y={168} color={THEME.text} size={14} weight={600}>
        {isFinite(ratio) ? "area ratio = k² = " + fmt(ratio, 1) + "×" : "area ratio: no word tokens"}
      </SvgLabel>
      <SvgLabel x={250} y={190} color={THEME.muted} size={12}>
        {nw > 0 ? "k = " + nc + " / " + nw + " = " + fmt(nc / nw, 2) + " times more tokens" : ""}
      </SvgLabel>
      <SvgLabel x={250} y={222} color={THEME.muted} size={11}>
        each side is n tokens; each cell is one pair
      </SvgLabel>
    </svg>
  );
}
// slide 18's table, with the learner's text filled in
function UnitsTable({ wt, ct }) {
  const nw = wt.length;
  const nc = ct.length;
  const rows = [
    ["vocabulary size V", unitsNum(UNITS_V) + " (slide 15)", UNITS_VC + " (a–z, [UNK])"],
    ["embedding parameters V × d, d = 768", unitsNum(UNITS_V * UNITS_D), unitsNum(UNITS_VC * UNITS_D)],
    ["sequence length n (your text)", String(nw), String(nc)],
    ["[UNK] tokens (your text)", String(wt.filter((t) => t.unk).length), String(ct.filter((t) => t.unk).length)],
    ["attention pairs n² (your text)", unitsNum(nw * nw), unitsNum(nc * nc)],
  ];
  const cell = { padding: "5px 8px", borderTop: "1px solid " + THEME.grid, fontFamily: MONO, fontSize: 12.5, fontVariantNumeric: "tabular-nums" };
  return (
    <div style={{ display: "grid", gridTemplateColumns: "minmax(0, 1.5fr) minmax(0, 1fr) minmax(0, 1fr)", background: THEME.bg, border: "1px solid " + THEME.grid, borderRadius: 6, overflow: "hidden" }}>
      <div style={Object.assign({}, cell, { color: THEME.muted, borderTop: "none" })}>property</div>
      <div style={Object.assign({}, cell, { color: THEME.third, borderTop: "none" })}>■ word tokens</div>
      <div style={Object.assign({}, cell, { color: THEME.highlight, borderTop: "none" })}>▭ character tokens</div>
      {rows.map((r) => (
        <React.Fragment key={r[0]}>
          <div style={Object.assign({}, cell, { color: THEME.muted, fontFamily: "inherit", fontSize: 13 })}>{r[0]}</div>
          <div style={Object.assign({}, cell, { color: THEME.text })}>{r[1]}</div>
          <div style={Object.assign({}, cell, { color: THEME.text })}>{r[2]}</div>
        </React.Fragment>
      ))}
    </div>
  );
}
function UnitsGap({ wi, gi, cut, onToggle }) {
  const click = useCallback(() => onToggle(wi, gi), [onToggle, wi, gi]);
  return (
    <button
      onClick={click}
      aria-label={cut ? "join the pieces at this cut" : "cut here"}
      aria-pressed={cut}
      style={{
        width: cut ? 16 : 10,
        height: 30,
        padding: 0,
        margin: 0,
        border: "none",
        background: "transparent",
        cursor: "pointer",
        fontFamily: MONO,
        fontSize: cut ? 17 : 12,
        fontWeight: cut ? 700 : 400,
        color: cut ? THEME.highlight : rgba(THEME.muted, 0.5),
        lineHeight: 1,
      }}
    >
      {cut ? "|" : "·"}
    </button>
  );
}
function UnitsCutRow({ wi, word, cuts, onToggle }) {
  const pieces = unitsSplit(word, cuts);
  const els = [];
  let pos = 0;
  pieces.forEach((p, pi) => {
    const known = unitsHas(p);
    const inner = [];
    for (let j = 0; j < p.length; j++) {
      inner.push(
        <span key={"l" + j} style={{ fontFamily: MONO, fontSize: 16, color: THEME.text }}>
          {p[j]}
        </span>
      );
      if (j < p.length - 1) inner.push(<UnitsGap key={"g" + j} wi={wi} gi={pos + j} cut={false} onToggle={onToggle} />);
    }
    els.push(
      <span
        key={"p" + pi}
        style={{
          display: "inline-flex",
          alignItems: "center",
          padding: "0 4px",
          borderRadius: 4,
          border: "1px " + (known ? "solid " + THEME.accent : "dashed " + THEME.highlight),
          background: known ? rgba(THEME.accent, 0.18) : rgba(THEME.highlight, 0.08),
        }}
      >
        {inner}
        <span style={{ fontFamily: MONO, fontSize: 10, color: known ? THEME.accent : THEME.highlight, marginLeft: 3, alignSelf: "flex-start" }}>{known ? "✓" : "+"}</span>
      </span>
    );
    if (pi < pieces.length - 1) els.push(<UnitsGap key={"c" + pi} wi={wi} gi={pos + p.length - 1} cut onToggle={onToggle} />);
    pos += p.length;
  });
  return (
    <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 8, padding: "3px 0", borderBottom: "1px dashed " + THEME.grid }}>
      <div style={{ display: "flex", alignItems: "center", flexWrap: "nowrap", minWidth: 0, overflowX: "auto" }}>{els}</div>
      <span style={{ fontFamily: MONO, fontSize: 12, color: THEME.muted, whiteSpace: "nowrap" }}>{pieces.length} tok</span>
    </div>
  );
}

export default function UnitsLesson({ initialScreen }) {
  const [screen, setScreen] = useState(initialScreen || 0);
  const [sa, setSa] = useState("we are unstoppable");
  const [sb, setSb] = useState("we read books");
  const [tx, setTx] = useState("we read unstoppable");
  const [k, setK] = useState(4);
  const [cuts, setCuts] = useState(unitsCutsFill(false));

  const toggle = useCallback((wi, gi) => setCuts((C) => C.map((r, a) => (a === wi ? r.map((v, b) => (b === gi ? !v : v)) : r))), []);
  const whole = useCallback(() => setCuts(unitsCutsFill(false)), []);
  const letters = useCallback(() => setCuts(unitsCutsFill(true)), []);

  let question, stage, side, below;
  if (screen === 0) {
    const ta = unitsWordTok(sa);
    const tb = unitsWordTok(sb);
    const differ = unitsWords(sa).join(" ") !== unitsWords(sb).join(" ");
    const same = ta.length > 0 && unitsSameIds(ta, tb);
    const met = differ && same;
    question = (
      <>
        <b>Where we are:</b> lesson 1 turned text into IDs with one vocabulary built once from the training corpus, V = 10,000 words (slide 9). <b>What is still broken:</b> new text keeps bringing forms the list lacks, such as unstoppable, unstoppability and the typo unstopabel (slide 14). <b>The puzzle:</b> here a missing word is looked up as [UNK]. Edit sentence B, or pick one, until the model receives exactly the same list of IDs for A and B, while the words differ.
      </>
    );
    stage = (
      <>
        <UnitsText id="u1-a" label="sentence A" value={sa} onChange={setSa} />
        <UnitsStrip tokens={ta} color={THEME.third} showIds />
        <UnitsText id="u1-b" label="sentence B" value={sb} onChange={setSb} />
        <Pills options={UNITS_PRESETS_B} value={sb} onPick={setSb} />
        <UnitsStrip tokens={tb} color={THEME.third} showIds />
        <Note>Each box is one word token with its ID underneath (slide 9's IDs; dashed red boxes are words missing from the vocabulary).</Note>
      </>
    );
    side = (
      <>
        <Panel edge={met ? THEME.ok : THEME.grid}>
          <Readout
            rows={[
              ["A → IDs", "[" + ta.map((t) => t.id).join(", ") + "]"],
              ["B → IDs", "[" + tb.map((t) => t.id).join(", ") + "]"],
              ["same words?", differ ? "no" : "yes"],
              ["same IDs?", same ? "yes" : "no", same ? THEME.ok : THEME.text],
            ]}
          />
          <Goal met={met}>different sentences, identical ID lists</Goal>
        </Panel>
        {met ? (
          <Note>
            The model cannot tell A from B: both are [{ta.map((t) => t.id).join(", ")}]. Slide 15 maps every out-of-vocabulary word to [UNK], ID 10,000, which makes V = 10,001. Different unknown words become identical, and the pieces they share (un, stop, able) are lost too.
          </Note>
        ) : (
          <Note>The model sees only the ID list. Try a word the 10,000-word list would not contain.</Note>
        )}
      </>
    );
  } else if (screen === 1) {
    const wt = unitsWordTok(tx);
    const ct = unitsCharTok(tx);
    const nw = wt.length;
    const nc = ct.length;
    question = (
      <>
        <b>Claim to test:</b> character tokens remove [UNK]. Any word can be written as long as each of its characters is in the vocabulary (slide 16). The price is length. Full attention compares every token with every other token, so its work grows as n², with n the number of tokens (slide 17). Type or pick a text. How many times more tokens does the character version need (k), and how many times more query–key pairs does attention then compute?
      </>
    );
    stage = (
      <>
        <UnitsText id="u2-t" label="text" value={tx} onChange={setTx} />
        <Pills options={UNITS_PRESETS_X} value={tx} onPick={setTx} />
        <Label>■ word tokens · V = 10,001</Label>
        <UnitsStrip tokens={wt} color={THEME.third} showIds />
        <Label>▭ character tokens · V = 27 (a–z and [UNK])</Label>
        <UnitsStrip tokens={ct} color={THEME.highlight} showIds />
        <UnitsCost nw={nw} nc={nc} />
      </>
    );
    side = (
      <>
        <Panel>
          <Readout
            rows={[
              ["word tokens n", String(nw)],
              ["character tokens n", String(nc)],
              ["k = longer by", nw ? fmt(nc / nw, 2) + "×" : "—"],
              ["attention pairs: words", unitsNum(nw * nw)],
              ["attention pairs: characters", unitsNum(nc * nc)],
              ["k² = more attention work", nw ? fmt((nc * nc) / (nw * nw), 1) + "×" : "—", THEME.highlight],
              ["[UNK]: words / characters", wt.filter((t) => t.unk).length + " / " + ct.filter((t) => t.unk).length, ct.some((t) => t.unk) ? THEME.warn : THEME.text],
            ]}
          />
        </Panel>
        <Note>
          Result: k is about the average word length, so English text in characters is roughly 4–6 times longer and costs roughly 16–36 times more attention, since n → kn turns n² into k²n². The Persian sentence gets [UNK] even as characters: character tokens only help when the characters are in the vocabulary (slide 18).
        </Note>
        <Note>Spaces only separate words here. A real character tokenizer can also give spaces and punctuation their own tokens (slide 16), which makes n longer still.</Note>
      </>
    );
  } else if (screen === 2) {
    const wt = unitsWordTok(tx);
    const ct = unitsCharTok(tx);
    question = (
      <>
        <b>What you found:</b> word tokens keep sequences short, but any form the fixed vocabulary lacks becomes [UNK], and [UNK] forgets which word it was. Character tokens build every word from known characters with a tiny V, but the text is k times longer, and full attention costs O(n²d), so k times more tokens means k² times more attention work (slides 15–18). Move k to check slide 17's numbers.
      </>
    );
    stage = (
      <>
        <UnitsTable wt={wt} ct={ct} />
        <Eq size={13}>
          full attention: cost = O(n² d) n = tokens, d = hidden width
          <br />
          n → k·n ⇒ cost → (k·n)² d = k² × cost
          <br />
          k = {fmt(k, 1)} ⇒ {fmt(k, 1)}× more tokens, {fmt(k * k, 1)}× more attention work
        </Eq>
        <Bars
          rows={[
            { label: "tokens × k", value: k, color: THEME.third, mark: "" },
            { label: "attention × k²", value: k * k, color: THEME.highlight, mark: "" },
          ]}
          range={[0, 64]}
          digits={1}
          labelW={120}
          ariaLabel="token growth k and attention growth k squared"
        />
      </>
    );
    side = (
      <>
        <Panel>
          <Slider id="u3-k" label="k = tokens per word token" value={k} min={1} max={8} stepSize={0.1} onChange={setK} display={"k = " + fmt(k, 1)} />
          <Readout
            rows={[
              ["tokens grow by", fmt(k, 1) + "×"],
              ["attention work grows by", fmt(k * k, 1) + "×", THEME.highlight],
            ]}
          />
          <Goal met={Math.abs(k - 4) < 0.05}>slide 17: 4× more tokens → 16× attention</Goal>
        </Panel>
        <Note>The table uses your text from the Explore screen. The other parts of a layer (projections, feed-forward) grow only like n, so attention is the part that blows up (slide 17). Longer sequences also mean more generation steps.</Note>
      </>
    );
    below = (
      <Panel style={{ marginTop: 12 }}>
        <div style={{ fontSize: 15, lineHeight: 1.55 }}>
          <b>Slide 19 asks:</b> can we keep the vocabulary manageable without making sequences so long? <b>Slide 20 answers:</b> one vocabulary that holds whole words, word fragments and single characters.
        </div>
        <div style={{ display: "flex", flexWrap: "wrap", alignItems: "center", gap: 6, margin: "8px 0" }}>
          <span style={{ fontFamily: MONO, fontSize: 14, color: THEME.muted, marginRight: 4 }}>we read unstoppable →</span>
          {["we", "read", "un", "stop", "p", "able"].map((s) => (
            <UnitsTok key={s} t={{ s: "[" + s + "]", unk: false }} color={THEME.accent} />
          ))}
          <span style={{ fontFamily: MONO, fontSize: 12.5, color: THEME.muted }}>6 tokens, no [UNK] (words: 3 with one [UNK]; characters: 17)</span>
        </div>
        <div style={{ fontSize: 14.5, lineHeight: 1.55, color: THEME.text }}>
          Such a tokenizer must answer two questions (slide 21): <b>1.</b> which units go into the vocabulary (tokenizer training), and <b>2.</b> how a new string is split into those units (tokenization, or inference). The units need not be meaningful pieces of the language. Next lesson, <a href="#merge" style={{ color: THEME.accent }}>BPE training</a>, answers question 1 by repeatedly merging the most frequent pair; lesson 4 answers question 2. First, the challenge: build such a vocabulary by hand.
        </div>
      </Panel>
    );
  } else {
    const r = unitsScore(cuts);
    const ok = r.okN && r.okV;
    const nWhole = UNITS_CH.length;
    const nChar = UNITS_CH.reduce((s, w) => s + w.length, 0);
    question = (
      <>
        Seven words that the 10,000-word vocabulary lacks. Click between two letters to cut a word there, and click a cut to join it again. A piece that is already a vocabulary word (✓, such as stop, read, like) costs nothing. Every other distinct piece (+) must become a new vocabulary entry, counted once however often it is used. Find cuts with at most {UNITS_MMAX} new entries and at most {UNITS_NMAX} tokens in total.
      </>
    );
    stage = (
      <>
        {UNITS_CH.map((w, i) => (
          <UnitsCutRow key={w} wi={i} word={w} cuts={cuts[i]} onToggle={toggle} />
        ))}
        <Note>✓ solid violet: already a vocabulary word · + dashed: needs a new entry · bold | marks a cut · faint dots mark places you can cut</Note>
      </>
    );
    side = (
      <>
        <Panel edge={ok ? THEME.ok : THEME.grid}>
          <Readout
            rows={[
              ["tokens n", String(r.n), r.okN ? THEME.ok : THEME.warn],
              ["new entries", r.fresh.length + (r.fresh.length ? ": " + r.fresh.join(", ") : ""), r.okV ? THEME.ok : THEME.warn],
              ["vocabulary size V", unitsNum(UNITS_V + r.fresh.length)],
              ["attention pairs n²", unitsNum(r.n * r.n)],
              ["(whole words / letters)", unitsNum(nWhole * nWhole) + " / " + unitsNum(nChar * nChar)],
            ]}
          />
          <Goal met={r.okN}>at most {UNITS_NMAX} tokens</Goal>
          <Goal met={r.okV}>at most {UNITS_MMAX} new vocabulary entries</Goal>
          <div style={{ display: "flex", gap: 6, flexWrap: "wrap", marginTop: 6 }}>
            <Btn onClick={whole}>whole words</Btn>
            <Btn onClick={letters}>every letter</Btn>
          </div>
          <Hint
            hints={[
              "Whole words are short, but each one is a new entry, and the next unseen word would need yet another. Look for a piece that several of these words share.",
              "Four words start with un and three with re. stop, read, reading, learn, like and write are already in the vocabulary, so as pieces they are free.",
            ]}
          />
        </Panel>
        {ok ? (
          <Note>
            You built a subword vocabulary. Shared fragments such as un and re are stored once and reused, known words stay whole inside longer ones, and only small leftovers are new, as in slide 20's [un] [stop] [p] [able]. Choosing the entries was question 1 of slide 21 (training) and placing the cuts was question 2 (inference). Lesson 3 lets BPE choose the entries from pair counts.
          </Note>
        ) : (
          <Note>
            Whole words: {nWhole} tokens but {nWhole} new entries. Every letter: {nChar} tokens. Somewhere between them both budgets hold.
          </Note>
        )}
      </>
    );
  }

  return <LessonFrame screen={screen} onScreen={setScreen} title="Words or characters?" question={question} stage={stage} side={side} below={below} />;
}
