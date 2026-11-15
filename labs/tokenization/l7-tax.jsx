import { useState, useCallback, useMemo } from "react";
import { THEME, MONO } from "./common";

// Lesson 7 - The Persian token tax (Tokenization, slides 42-48).
// Aha: the extra tokens Persian pays are not a property of Persian. The same BPE algorithm with the same
// merge budget hands its merges to whichever language has the more frequent pairs in the training corpus,
// so a mostly-English corpus leaves Persian cut into letters (slide 42). To measure the tax, compare token
// counts for the same meaning: Δ = FA − EN cancels the fixed message overhead that a ratio cannot (slide 45).
//
// Which numbers are real and which are toys:
// - SLIDE NUMBERS (slides 45-47, OpenRouter API, 8-9 Nov 2026): the sentence pair and every Persian / English
//   input-token count in TAX_ROWS, with Δ = FA − EN. One sentence pair, not a general tokenizer ranking.
// - Character counts 96 / 95 are counted here from the slide's sentences (spaces, punctuation and ZWNJ included).
// - TEACHING TOY: the English + Persian corpus (TAX_EN, TAX_FA: words and frequencies are made up), the corpus
//   size 1000, the merge budgets M, and every token count the toy BPE produces. The toy is character-based,
//   word-bounded BPE with </w> and lexicographic ties, exactly as on slides 24-32 (real tokenizers start from
//   bytes, lesson 6). The "fixed overhead c" on the Name screen is a what-if knob; the true overhead is unknown.

// ---------- slides 46-47: API-reported input tokens for one user message ----------
const TAX_ROWS = [
  ["Gemini 2.5 Pro", 29, 21],
  ["Gemini 3.8 Flash", 28, 20],
  ["Ministral 8B 2512", 34, 23],
  ["Mistral Large 4", 46, 35],
  ["Llama 3.3 70B", 67, 55],
  ["Llama 4 Maverick", 42, 30],
  ["Cohere Command A", 33, 20],
  ["GPT-4o", 40, 27],
  ["GPT-6 Luna", 39, 26],
  ["GPT-6.1 Sol", 39, 26],
  ["Qwen 3.8 27B", 48, 32],
  ["GLM-5.3", 50, 32],
  ["Kimi K3", 60, 38],
  ["Amazon Nova Micro", 43, 20],
  ["DeepSeek V4 Pro", 47, 24],
  ["DeepSeek V4.1 Flash", 47, 24],
  ["Claude Haiku 5.5", 76, 28],
  ["Claude Opus 5.5", 78, 30],
  ["Claude Sonnet 5.5", 78, 30],
];

// @sim-start
// ---- teaching toy: a tiny English + Persian corpus (word -> relative frequency) ----
const TAX_EN = [
  ["the", 60], ["and", 36], ["of", 32], ["to", 34], ["a", 28], ["is", 22], ["in", 22], ["it", 14], ["that", 14],
  ["they", 14], ["do", 12], ["not", 14], ["but", 12], ["what", 10], ["when", 6], ["where", 5],
  ["student", 10], ["students", 8], ["want", 8], ["wants", 5], ["read", 8], ["reads", 4], ["reading", 6],
  ["book", 10], ["books", 8], ["library", 6], ["libraries", 3], ["know", 7], ["knows", 4], ["time", 8], ["times", 3],
  ["open", 5], ["opens", 3], ["study", 5], ["class", 4], ["hours", 3], ["day", 5], [",", 30], [".", 30],
];
const TAX_FA = [
  ["و", 40], ["در", 34], ["به", 30], ["از", 30], ["که", 24], ["این", 20], ["است", 24], ["را", 28], ["با", 14],
  ["ولی", 10], ["چه", 10], ["کی", 5], ["کجا", 4],
  ["دانشجو", 9], ["دانشجوها", 5], ["می‌خواهم", 5], ["می‌خواهند", 4], ["می‌خوانم", 4], ["بخوانند", 3], ["خواندن", 5],
  ["کتاب", 10], ["کتاب‌ها", 6], ["کتاب‌های", 4], ["کتابخانه", 6], ["کتابخانه‌ها", 3], ["می‌دانم", 5], ["نمی‌دانند", 3],
  ["ساعت", 6], ["ساعتی", 3], ["باز", 5], ["می‌شود", 5], ["می‌شوند", 3], ["درس", 5], ["کلاس", 4], ["روز", 5], ["،", 30], [".", 30],
];
// held-out text: slide 45's sentence pair (never in the toy corpus as a sentence)
const TAX_FA_SENT = "دانشجوها می‌خواهند کتاب‌های کتابخانه را بخوانند، ولی نمی‌دانند کتابخانه‌ها چه ساعتی باز می‌شوند.";
const TAX_EN_SENT = "The students want to read the library books, but they do not know what time the libraries open.";
const TAX_N = 1000; // words in the toy corpus
const TAX_EOW = "</w>";
const TAX_CM = 120; // challenge: fixed merge budget
const TAX_TOL = 1.2; // challenge: neither sentence may cost more than 1.2x the other
const TAX_START = 0.1; // challenge: starting Persian share

// split on spaces, punctuation is its own word, English lower-cased (same preprocessing for training and test)
function taxWords(s) {
  const out = [];
  for (const w of s.toLowerCase().split(" ")) {
    const m = w.match(/^(.*?)([.,،]?)$/);
    if (m[1]) out.push(m[1]);
    if (m[2]) out.push(m[2]);
  }
  return out;
}
// word counts of a corpus of TAX_N words with Persian share p (0..1)
function taxCorpus(p) {
  const out = [];
  const add = (list, share) => {
    let tot = 0;
    for (const e of list) tot += e[1];
    for (const e of list) {
      const c = Math.round((TAX_N * share * e[1]) / tot);
      if (c > 0) out.push([e[0], c]);
    }
  };
  add(TAX_EN, 1 - p);
  add(TAX_FA, p);
  return out;
}
function taxMergeOnce(s, a, b) {
  const out = [];
  for (let i = 0; i < s.length; i++) {
    if (i + 1 < s.length && s[i] === a && s[i + 1] === b) {
      out.push(a + b);
      i++;
    } else out.push(s[i]);
  }
  return out;
}
// character-based, word-bounded BPE (slides 24-29): merge the most frequent pair, ties by lexicographic pair order
function taxTrain(p, M) {
  const words = taxCorpus(p).map((e) => ({ s: Array.from(e[0]).concat([TAX_EOW]), c: e[1] }));
  const rules = [];
  for (let m = 0; m < M; m++) {
    const cnt = new Map();
    for (const w of words)
      for (let i = 0; i + 1 < w.s.length; i++) {
        const k = w.s[i] + "\u0000" + w.s[i + 1];
        cnt.set(k, (cnt.get(k) || 0) + w.c);
      }
    let best = null;
    let bc = 0;
    for (const [k, c] of cnt)
      if (c > bc || (c === bc && k < best)) {
        best = k;
        bc = c;
      }
    if (!best || bc < 2) break;
    const ab = best.split("\u0000");
    rules.push({ a: ab[0], b: ab[1], c: bc });
    for (const w of words) w.s = taxMergeOnce(w.s, ab[0], ab[1]);
  }
  return rules;
}
function taxRank(rules) {
  const m = new Map();
  rules.forEach((r, i) => m.set(r.a + "\u0000" + r.b, i));
  return m;
}
// inference (slides 30-32): apply the eligible rule with the earliest rank until none applies
function taxEncodeWord(w, rank) {
  let s = Array.from(w).concat([TAX_EOW]);
  for (;;) {
    let br = Infinity;
    let bi = -1;
    for (let i = 0; i + 1 < s.length; i++) {
      const r = rank.get(s[i] + "\u0000" + s[i + 1]);
      if (r !== undefined && r < br) {
        br = r;
        bi = i;
      }
    }
    if (bi < 0) return s;
    s = taxMergeOnce(s, s[bi], s[bi + 1]);
  }
}
function taxEncode(sent, rank) {
  return taxWords(sent).map((w) => taxEncodeWord(w, rank));
}
function taxCount(enc) {
  let n = 0;
  for (const w of enc) n += w.length;
  return n;
}
function taxIsFa(piece) {
  return /[؀-ۿ]/.test(piece);
}
// base symbols: every character of both corpora and of the held-out pair, plus </w> (so nothing is [UNK])
function taxBase() {
  const s = new Set([TAX_EOW]);
  for (const e of TAX_EN.concat(TAX_FA)) for (const ch of e[0]) s.add(ch);
  for (const w of taxWords(TAX_FA_SENT).concat(taxWords(TAX_EN_SENT))) for (const ch of w) s.add(ch);
  return s.size;
}
const TAX_CACHE = new Map();
// one tokenizer trained with Persian share p and budget M, measured on the held-out pair (cached)
function taxRun(p, M) {
  const key = Math.round(p * 100) + "|" + M;
  const hit = TAX_CACHE.get(key);
  if (hit) return hit;
  const rules = taxTrain(p, M);
  const rank = taxRank(rules);
  const fa = taxEncode(TAX_FA_SENT, rank);
  const en = taxEncode(TAX_EN_SENT, rank);
  const nf = taxCount(fa);
  const ne = taxCount(en);
  let faRules = 0;
  for (const r of rules) if (taxIsFa(r.a + r.b)) faRules++;
  const out = { p, M, rules, fa, en, nf, ne, ratio: nf / ne, faRules };
  TAX_CACHE.set(key, out);
  return out;
}
function taxCurve(M, stepPct) {
  const pts = [];
  for (let i = 0; i <= 100; i += stepPct) pts.push(taxRun(i / 100, M));
  return pts;
}
function taxPass(r) {
  return r.nf <= TAX_TOL * r.ne && r.ne <= TAX_TOL * r.nf;
}
// slide 45: a ratio of API counts mixes in a fixed overhead c; the content ratio is (FA − c) / (EN − c)
function taxContentRatio(fa, en, c) {
  return (fa - c) / (en - c);
}
// @sim-end

const TAX_B = taxBase();
const TAX_CH_FA = Array.from(TAX_FA_SENT).length;
const TAX_CH_EN = Array.from(TAX_EN_SENT).length;

function taxOrder(sortKey, c) {
  const idx = TAX_ROWS.map((r, i) => i);
  const key = (i) => {
    const r = TAX_ROWS[i];
    if (sortKey === "fa") return r[1];
    if (sortKey === "ratio") return taxContentRatio(r[1], r[2], c);
    return r[1] - r[2];
  };
  idx.sort((a, b) => key(a) - key(b) || a - b);
  return idx;
}

// ---------- the real table (slides 46-47) as paired bars ----------
function TaxTable({ sortKey, sel, onSel, c }) {
  const order = taxOrder(sortKey, c || 0);
  const W = 600;
  const rh = 23;
  const top = 26;
  const x0 = 150;
  const x1 = 430;
  const X = (v) => x0 + (v / 80) * (x1 - x0);
  const H = top + TAX_ROWS.length * rh + 6;
  return (
    <svg viewBox={"0 0 " + W + " " + H} style={{ width: "100%", display: "block", background: THEME.bg, border: "1px solid " + THEME.grid, borderRadius: 6 }} role="img" aria-label="Persian and English input tokens per model, slides 46 and 47; click a row to select a model">
      <text x={x0 - 6} y={17} textAnchor="end" fontFamily={MONO} fontSize={10.5} fill={THEME.muted}>
        MODEL
      </text>
      <rect x={x0} y={8} width={14} height={8} fill={THEME.highlight} />
      <text x={x0 + 18} y={16} fontFamily={MONO} fontSize={10.5} fill={THEME.muted}>
        Persian (solid)
      </text>
      <rect x={x0 + 118} y={8} width={14} height={8} fill={rgba(THEME.accent, 0.25)} stroke={THEME.accent} strokeWidth={1} />
      <text x={x0 + 136} y={16} fontFamily={MONO} fontSize={10.5} fill={THEME.muted}>
        English (outlined)
      </text>
      <text x={480} y={17} textAnchor="middle" fontFamily={MONO} fontSize={10.5} fill={sortKey === "delta" ? THEME.highlight : THEME.muted}>
        Δ FA−EN
      </text>
      <text x={556} y={17} textAnchor="middle" fontFamily={MONO} fontSize={10.5} fill={sortKey === "ratio" ? THEME.highlight : THEME.muted}>
        {c ? "ratio, c=" + c : "FA / EN"}
      </text>
      {order.map((ri, k) => {
        const r = TAX_ROWS[ri];
        const y = top + k * rh;
        const on = ri === sel;
        const ratio = taxContentRatio(r[1], r[2], c || 0);
        return (
          <g key={r[0]} onClick={() => onSel(ri)} style={{ cursor: "pointer" }}>
            <rect x={2} y={y} width={W - 4} height={rh - 1} rx={3} fill={on ? rgba(THEME.highlight, 0.12) : "transparent"} stroke={on ? THEME.highlight : "none"} strokeWidth={1} />
            <text x={x0 - 6} y={y + 15} textAnchor="end" fontFamily={MONO} fontSize={11.5} fill={on ? THEME.highlight : THEME.text}>
              {(on ? "▸ " : "") + r[0]}
            </text>
            <rect x={x0} y={y + 3} width={X(r[1]) - x0} height={8} fill={THEME.highlight} />
            <text x={X(r[1]) + 4} y={y + 10.5} fontFamily={MONO} fontSize={9.5} fill={THEME.text}>
              {r[1]}
            </text>
            <rect x={x0} y={y + 12.5} width={X(r[2]) - x0} height={7} fill={rgba(THEME.accent, 0.25)} stroke={THEME.accent} strokeWidth={1} />
            <text x={X(r[2]) + 4} y={y + 19.5} fontFamily={MONO} fontSize={9.5} fill={THEME.muted}>
              {r[2]}
            </text>
            <text x={480} y={y + 15} textAnchor="middle" fontFamily={MONO} fontSize={12} fill={THEME.text}>
              {"+" + (r[1] - r[2])}
            </text>
            <text x={556} y={y + 15} textAnchor="middle" fontFamily={MONO} fontSize={12} fill={THEME.text}>
              {"×" + fmt(ratio, 2)}
            </text>
          </g>
        );
      })}
    </svg>
  );
}

// ---------- the toy tokenizer's output on the held-out pair ----------
function TaxPiece({ piece, fa }) {
  const txt = piece === TAX_EOW ? "</w>" : piece.replace(TAX_EOW, "");
  const shown = txt === "‌" ? "ZWNJ" : txt;
  const single = Array.from(piece).length === 1 || piece === TAX_EOW;
  const small = shown === "ZWNJ" || shown === "</w>";
  return (
    <span
      style={{
        fontFamily: fa && !small ? FA : MONO,
        fontSize: small ? 10 : fa ? 15 : 13,
        padding: fa ? "0 4px 2px" : "1px 4px",
        borderRadius: 3,
        border: "1px " + (single ? "dashed " + THEME.warn : "solid " + (fa ? THEME.highlight : THEME.accent)),
        background: single ? "transparent" : rgba(fa ? THEME.highlight : THEME.accent, 0.13),
        color: small ? THEME.muted : THEME.text,
        lineHeight: 1.6,
        whiteSpace: "nowrap",
      }}
    >
      {shown}
    </span>
  );
}
function TaxPieces({ enc, fa }) {
  return (
    <div dir={fa ? "rtl" : "ltr"} lang={fa ? "fa" : "en"} style={{ display: "flex", flexWrap: "wrap", gap: "7px 16px", margin: "2px 0 6px" }}>
      {enc.map((w, i) => (
        <span key={i} style={{ display: "inline-flex", gap: 2 }}>
          {w.map((pc, j) => (
            <TaxPiece key={j} piece={pc} fa={fa} />
          ))}
        </span>
      ))}
    </div>
  );
}
function TaxSegments({ run }) {
  return (
    <Panel>
      <Label>
        held-out Persian sentence (slide 45) · {run.nf} tokens
      </Label>
      <TaxPieces enc={run.fa} fa />
      <Label>
        held-out English sentence · {run.ne} tokens
      </Label>
      <TaxPieces enc={run.en} />
      <Note>Solid box: a merged piece. Dashed red box: a single character or a lone &lt;/w&gt; that no merge reached. A gap separates words; a piece that ends at the gap carries &lt;/w&gt;.</Note>
    </Panel>
  );
}

// ---------- token counts against corpus mix ----------
function TaxCurve({ pts, p, dots, label }) {
  const W = 600;
  const H = 270;
  const L = 44;
  const R = 584;
  const T = 16;
  const B = 238;
  const X = (v) => L + v * (R - L);
  const Y = (v) => B - (Math.min(v, 100) / 100) * (B - T);
  const sorted = pts.slice().sort((a, b) => a.p - b.p);
  const line = (key) => sorted.map((r) => X(r.p) + "," + Y(r[key])).join(" ");
  const cur = pts.find((r) => Math.abs(r.p - p) < 1e-9);
  return (
    <svg viewBox={"0 0 " + W + " " + H} style={{ width: "100%", display: "block", background: THEME.bg, border: "1px solid " + THEME.grid, borderRadius: 6 }} role="img" aria-label={label}>
      {[0, 25, 50, 75, 100].map((v) => (
        <g key={"y" + v}>
          <line x1={L} x2={R} y1={Y(v)} y2={Y(v)} stroke={THEME.grid} strokeWidth={0.6} />
          <text x={L - 6} y={Y(v) + 4} textAnchor="end" fontFamily={MONO} fontSize={10.5} fill={THEME.muted}>
            {v}
          </text>
        </g>
      ))}
      {[0, 0.25, 0.5, 0.75, 1].map((v) => (
        <text key={"x" + v} x={X(v)} y={B + 16} textAnchor="middle" fontFamily={MONO} fontSize={10.5} fill={THEME.muted}>
          {Math.round(v * 100) + "%"}
        </text>
      ))}
      <text x={R} y={H - 4} textAnchor="end" fontFamily={MONO} fontSize={10.5} fill={THEME.muted}>
        Persian share p of the training corpus →
      </text>
      <text x={L + 4} y={T + 2} fontFamily={MONO} fontSize={10.5} fill={THEME.muted}>
        tokens for the held-out sentence
      </text>
      {sorted.length > 1 ? <polyline points={line("nf")} fill="none" stroke={THEME.highlight} strokeWidth={2} /> : null}
      {sorted.length > 1 ? <polyline points={line("ne")} fill="none" stroke={THEME.accent} strokeWidth={2} strokeDasharray="6 4" /> : null}
      {dots
        ? sorted.map((r) => (
            <g key={"d" + r.p}>
              <Marker x={X(r.p)} y={Y(r.nf)} shape="square" color={THEME.highlight} r={3} />
              <Marker x={X(r.p)} y={Y(r.ne)} shape="circle" color={THEME.accent} r={3} />
            </g>
          ))
        : null}
      <line x1={X(p)} x2={X(p)} y1={T} y2={B} stroke={THEME.text} strokeWidth={1} strokeDasharray="3 3" opacity={0.6} />
      {cur ? (
        <g>
          <Marker x={X(p)} y={Y(cur.nf)} shape="square" color={THEME.highlight} r={6} />
          <Marker x={X(p)} y={Y(cur.ne)} shape="circle" color={THEME.accent} r={6} />
          <SvgLabel x={X(p) + (p > 0.7 ? -10 : 10)} y={Y(cur.nf) - 8} anchor={p > 0.7 ? "end" : "start"} color={THEME.highlight}>
            {"■ Persian " + cur.nf}
          </SvgLabel>
          <SvgLabel x={X(p) + (p > 0.7 ? -10 : 10)} y={Y(cur.ne) + 18} anchor={p > 0.7 ? "end" : "start"} color={THEME.accent}>
            {"● English " + cur.ne}
          </SvgLabel>
        </g>
      ) : null}
    </svg>
  );
}

const TAX_CHECKS = [
  ["Sequence length", "tokens for the same held-out texts; inspect the long tail"],
  ["Coverage", "unknown characters, rare names, emoji, code, typos"],
  ["Reconstruction", "does decode(encode(x)) return the intended text?"],
  ["Cost", "embedding size, throughput, memory, token budget"],
  ["Model quality", "downstream results under a controlled training budget"],
];
const TAX_EFFECTS = [
  ["Potentially lower quality", "fragmentation and truncation can make learning harder"],
  ["Higher monetary cost", "more billed tokens for the same content"],
  ["Higher latency", "longer sequences can require more computation"],
  ["Less usable context", "a fixed token window holds less actual text"],
];
function TaxList({ title, rows }) {
  return (
    <Panel style={{ flex: "1 1 320px", minWidth: 0 }}>
      <Label>{title}</Label>
      {rows.map((r) => (
        <div key={r[0]} style={{ fontSize: 13.5, lineHeight: 1.45, margin: "3px 0" }}>
          <span style={{ color: THEME.highlight, fontFamily: MONO, fontSize: 12.5 }}>{r[0]}</span>
          <span style={{ color: THEME.muted }}> · {r[1]}</span>
        </div>
      ))}
    </Panel>
  );
}

export default function TaxLesson({ initialScreen }) {
  const [screen, setScreen] = useState(initialScreen || 0);
  const [sortKey, setSortKey] = useState("delta");
  const [sel, setSel] = useState(17);
  const [c, setC] = useState(0);
  const [p, setP] = useState(0.1);
  const [M, setM] = useState(100);
  const [pc, setPc] = useState(TAX_START);
  const [seen, setSeen] = useState([Math.round(TAX_START * 100)]);

  const pickSort = useCallback((v) => setSortKey(v), []);
  const onP = useCallback((v) => setP(v), []);
  const onM = useCallback((v) => setM(v), []);
  const onC = useCallback((v) => setC(v), []);
  const onPc = useCallback((v) => {
    setPc(v);
    const k = Math.round(v * 100);
    setSeen((S) => (S.indexOf(k) >= 0 ? S : S.concat([k])));
  }, []);

  const curve = useMemo(() => (screen === 1 ? taxCurve(M, 2) : []), [screen, M]);

  const row = TAX_ROWS[sel];
  const sortPills = (
    <Pills
      options={[
        ["delta", "sort by Δ (slide order)"],
        ["fa", "by Persian tokens"],
        ["ratio", "by ratio FA / EN"],
      ]}
      value={sortKey}
      onPick={pickSort}
    />
  );

  let question, stage, side, below;
  if (screen === 0) {
    question = (
      <>
        <b>Where we are:</b> lesson 6 showed that byte-level BPE can encode any Persian text, so coverage is solved. Coverage is not efficiency. Slide 45 sent one Persian sentence and its English translation to 19 models through the OpenRouter API and read back the input tokens each one billed. <b>The puzzle:</b> the two sentences mean the same thing and have about the same number of characters, yet every model bills more tokens for the Persian one, from +8 up to +48. Click models and change the sort. If the extra cost came from the Persian language itself, would the gap differ this much from model to model?
      </>
    );
    stage = <TaxTable sortKey={sortKey} sel={sel} onSel={setSel} c={0} />;
    side = (
      <>
        <Panel>
          <Label>the sentence pair (slide 45)</Label>
          <div dir="rtl" style={{ fontFamily: FA, fontSize: 16, lineHeight: 1.7, margin: "0 0 6px" }} lang="fa">
            {TAX_FA_SENT}
          </div>
          <div style={{ fontSize: 14, lineHeight: 1.5, color: THEME.text }}>{TAX_EN_SENT}</div>
          <Note>
            Characters: Persian {TAX_CH_FA}, English {TAX_CH_EN} (spaces, punctuation and the zero-width non-joiners in <Fa>می‌خواهند</Fa> and the like included).
          </Note>
        </Panel>
        <Panel edge={THEME.highlight}>
          <Label>{row[0]}</Label>
          <Readout
            rows={[
              ["Persian input tokens", String(row[1]), THEME.highlight],
              ["English input tokens", String(row[2]), THEME.accent],
              ["Δ = FA − EN", "+" + (row[1] - row[2])],
              ["ratio FA / EN", "×" + fmt(row[1] / row[2], 2)],
              ["characters per token, Persian", fmt(TAX_CH_FA / row[1], 2)],
              ["characters per token, English", fmt(TAX_CH_EN / row[2], 2)],
            ]}
          />
        </Panel>
        <Panel>
          {sortPills}
          <Note>Sort by ratio and look where Llama 3.3 70B lands, then sort by Δ. The two rankings disagree. Keep that question for screen 3.</Note>
        </Panel>
      </>
    );
  } else if (screen === 1) {
    const run = taxRun(p, M);
    const pts = Math.round(p * 100) % 2 === 0 ? curve : curve.concat([run]);
    question = (
      <>
        <b>Claim to test (slide 42):</b> with the same BPE algorithm and the same budget of M merges, the training corpus decides which recurring chunks get vocabulary space. Here a tiny BPE (the one from lessons 3–4) trains on a toy corpus of {TAX_N} words, a share p Persian and the rest English, then encodes slide 45's sentence pair as held-out text. Move p and M. Where do the two curves cross, and what happens to the English sentence as p approaches 100%?
      </>
    );
    stage = (
      <>
        <TaxCurve pts={pts} p={run.p} label={"toy BPE with M = " + M + ": tokens for the held-out Persian and English sentence against the Persian share of the corpus"} />
        <div style={{ height: 8 }} />
        <TaxSegments run={run} />
      </>
    );
    side = (
      <>
        <Panel>
          <Slider id="t7-p" label="Persian share p of the training corpus" value={p} min={0} max={1} stepSize={0.01} onChange={onP} display={Math.round(p * 100) + "%"} />
          <Slider id="t7-m" label="merge budget M" value={M} min={40} max={160} stepSize={10} onChange={onM} display={"M = " + M} />
          <Readout
            rows={[
              ["Persian sentence (15 words)", run.nf + " tokens", THEME.highlight],
              ["English sentence (20 words)", run.ne + " tokens", THEME.accent],
              ["ratio FA / EN", "×" + fmt(run.ratio, 2), run.ratio > 1.2 ? THEME.warn : THEME.text],
              ["merges that built Persian pieces", run.faRules + " of " + run.rules.length],
              ["vocabulary V = base + M", TAX_B + " + " + run.rules.length + " = " + (TAX_B + run.rules.length)],
            ]}
          />
        </Panel>
        <Note>
          Each merge goes to the most frequent pair in the whole corpus. At p = 0% no Persian pair ever wins, so the Persian sentence stays in single letters (98 tokens for 15 words) while every English word is one token. The budget is shared: merges that Persian gains, English loses. The curve marks every 2%; small bumps are real, because one merge changes which pairs win later.
        </Note>
      </>
    );
  } else if (screen === 2) {
    const cr = taxContentRatio(row[1], row[2], c);
    question = (
      <>
        <b>What you found:</b> the extra tokens are not in the Persian text, they are in the tokenizer. Same algorithm and same budget, a different corpus mix, so different pair counts, different merges and different pieces (slide 42). To measure that tax, slide 45 compares two counts for the same meaning from the same model. Each API count is a fixed message overhead c plus the content, so the difference Δ cancels c, while the ratio does not. Move the what-if overhead c and sort by ratio.
      </>
    );
    stage = (
      <>
        <TaxTable sortKey={sortKey} sel={sel} onSel={setSel} c={c} />
        <Eq size={12.5}>
          T_FA = c + content_FA,&nbsp;&nbsp; T_EN = c + content_EN
          <br />
          Δ = T_FA − T_EN = content_FA − content_EN&nbsp;&nbsp; (c cancels)
          <br />
          T_FA / T_EN ≤ (T_FA − c) / (T_EN − c) = content ratio
        </Eq>
      </>
    );
    side = (
      <>
        <Panel>
          {sortPills}
          <Slider id="t7-c" label="what-if fixed overhead c (unknown on the slide)" value={c} min={0} max={18} stepSize={1} onChange={onC} display={"c = " + c} />
          <Label>{row[0]}</Label>
          <Readout
            rows={[
              ["Δ (any c)", "+" + (row[1] - row[2])],
              ["ratio of API counts", "×" + fmt(row[1] / row[2], 2)],
              ["content ratio at this c", "×" + fmt(cr, 2), THEME.highlight],
              ["100 such sentences: extra Persian tokens", "+" + 100 * (row[1] - row[2])],
              ["fixed window: Persian text vs English", "≤ " + Math.round(100 / cr) + "%"],
            ]}
          />
        </Panel>
        <Note>
          Llama 3.3 70B bills 55 tokens for the English sentence where most models bill 20–35, which suggests a larger fixed overhead. That overhead pulls its ratio towards 1, so it looks the most Persian-friendly by ratio. Δ is not fooled. Slide 46–47 counts, one sentence pair: a measurement, not a general ranking.
        </Note>
      </>
    );
    below = (
      <div style={{ display: "flex", flexWrap: "wrap", gap: 10, marginTop: 12 }}>
        <TaxList title="evaluating a tokenizer (slide 43, Ali et al. 2024)" rows={TAX_CHECKS} />
        <TaxList title="why the token count matters (slide 48)" rows={TAX_EFFECTS} />
        <Note>
          Fewer tokens alone do not prove a better language model: compare tokenizers with the same V on held-out text first (slide 44's two V = 32,000 tokenizers for a Persian assistant), then train under a controlled budget. This closes the Tokenization Lab. Text became IDs (lesson 1), BPE learned which pieces get an ID (lessons 3–5), bytes made every Persian character encodable (lesson 6), and here the training corpus decided how many of those IDs a Persian sentence costs.
        </Note>
      </div>
    );
  } else {
    const run = taxRun(pc, TAX_CM);
    const pts = seen.map((k) => taxRun(k / 100, TAX_CM));
    const g1 = run.nf <= TAX_TOL * run.ne;
    const g2 = run.ne <= TAX_TOL * run.nf;
    question = (
      <>
        Slide 44 asks which of two tokenizers with the same vocabulary size to pick for a Persian assistant. Its users also mix English into their messages. Fix the budget at M = {TAX_CM} merges and choose the corpus mix p so that the slide 45 pair is fair: <b>the Persian sentence costs at most {TAX_TOL}× the English one, and the English one at most {TAX_TOL}× the Persian one</b>. Each value of p you try leaves a dot on the chart.
      </>
    );
    stage = (
      <>
        <TaxCurve pts={pts} p={run.p} dots label={"toy BPE with M = " + TAX_CM + ": the corpus mixes you have tried"} />
        <div style={{ height: 8 }} />
        <TaxSegments run={run} />
      </>
    );
    side = (
      <>
        <Panel edge={g1 && g2 ? THEME.ok : THEME.grid}>
          <Slider id="t7-pc" label="Persian share p of the training corpus" value={pc} min={0} max={1} stepSize={0.01} onChange={onPc} display={Math.round(pc * 100) + "%"} />
          <Readout
            rows={[
              ["budget", "M = " + TAX_CM + " (V = " + (TAX_B + TAX_CM) + ")"],
              ["Persian sentence", run.nf + " tokens", THEME.highlight],
              ["English sentence", run.ne + " tokens", THEME.accent],
              ["FA / EN", "×" + fmt(run.ratio, 2)],
              ["merges that built Persian pieces", run.faRules + " of " + run.rules.length],
            ]}
          />
          <Goal met={g1}>Persian ≤ {TAX_TOL} × English (no Persian tax beyond 20%)</Goal>
          <Goal met={g2}>English ≤ {TAX_TOL} × Persian (the tax did not move onto English)</Goal>
          <Hint
            hints={[
              "Look at the two ends of the slider: which sentence is longer at p = 0%, and which at p = 100%? The fair mixes lie between, where the two dots meet.",
              "Persian words here are longer than English ones, so Persian needs a bit more than half the corpus before its words become whole pieces. Try shares between 50% and 70%, and check both goals at each one.",
            ]}
          />
        </Panel>
        <Note>Toy corpus and toy budget: the fair mix for a real tokenizer must be measured on real held-out text (slide 43), including informal Persian, names, code-switching, numbers and emoji (slide 44).</Note>
      </>
    );
  }

  return <LessonFrame screen={screen} onScreen={setScreen} title="The Persian token tax" question={question} stage={stage} side={side} below={below} />;
}
