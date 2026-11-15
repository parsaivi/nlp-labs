import { useState, useCallback } from "react";
import { THEME, MONO } from "./common";

// Lesson 6 - Bytes, Persian text, and meaning (Tokenization, slides 38-41, with the opening puzzles on slides 2-4).
// Aha: a byte-level tokenizer reads UTF-8 bytes, not the letters we see. 256 byte values cover every
// possible input, so no character is ever unknown, but that coverage says nothing about meaning: BPE merges
// frequent byte pairs, not morphemes, and two Persian strings that look identical (ZWNJ vs space, Persian vs
// Arabic yeh/kaf) are different bytes, so the model sees different tokens unless we normalize first.
//
// Numbers from the slides: the toy alphabet d e i l n o r s t w and base IDs 0-10 with </w> = 0 (slides 25, 29),
// the unseen word "lowest" (slide 31), d = 768 for embedding sizes (slide 37), ژ and 🦉 (slide 38),
// A = 41, ش = D8 B4, 🦉 = F0 9F A6 89 and the 256 byte values (slide 39), Unputdownable = un + put + down + able
// (slides 2-3, 40), اجتناب‌ناپذیرمندی (slide 4), and the three spellings of می‌روم with U+200C / U+0020 / none and
// the code points U+06CC, U+064A, U+06A9, U+0643 (slide 41).
// Exact but not on a slide: every other byte sequence (computed live with the browser's TextEncoder), the
// 1,114,112 Unicode code points, and NFC/NFKC results (the browser's String.normalize).
// Teaching toys: "tomorrow's inputs" on screen 1, the 1,000-row budget, the two BPE training corpora and their
// counts on screen 3, and the eight-string corpus on screen 4 (including the PDF copy and به‌روز / بهروز).

// @sim-start
const BY_ENC = typeof TextEncoder !== "undefined" ? new TextEncoder() : null;
function bytesOf(s) {
  return Array.from(BY_ENC.encode(s));
}
function bytesHex(b) {
  return (b < 16 ? "0" : "") + b.toString(16).toUpperCase();
}
function bytesU(cp) {
  let h = cp.toString(16).toUpperCase();
  while (h.length < 4) h = "0" + h;
  return "U+" + h;
}
// one entry per code point, with its real UTF-8 bytes
function bytesCps(s) {
  return Array.from(s).map((ch) => ({ ch: ch, cp: ch.codePointAt(0), bytes: bytesOf(ch) }));
}

// ---- screen 1: which base alphabet covers which input ----
const BY_TOY = "deilnorstw"; // slide 25/29: base IDs 1-10, </w> = 0
const BY_UNICODE_ROWS = 1114112; // code points U+0000 .. U+10FFFF
const BY_D = 768; // slide 37
const BY_BUDGET = 1000; // teaching toy: a base table we are willing to keep
const BY_PAGE_INPUTS = ["lowest", "\u0698", "\uD83E\uDD89"]; // lowest (slide 31), ژ and 🦉 (slide 38)
const BY_TOMORROW = ["caf\u00E9", "\u5B57", "\u20AC5", "\uD83E\uDD8A"]; // café, 字, €5, 🦊 (toys)
function bytesSeenAlphabet(inputs) {
  const extra = [];
  for (const s of inputs)
    for (const ch of Array.from(s)) {
      if (/\s/.test(ch)) continue;
      if (BY_TOY.indexOf(ch) < 0 && extra.indexOf(ch) < 0) extra.push(ch);
    }
  return extra;
}
function bytesBaseRows(mode, extra) {
  if (mode === "toy") return 11;
  if (mode === "seen") return 11 + extra.length;
  if (mode === "unicode") return BY_UNICODE_ROWS;
  return 256;
}
function bytesCovers(mode, extra, s) {
  if (mode === "unicode" || mode === "bytes") return true;
  for (const ch of Array.from(s)) {
    if (/\s/.test(ch)) continue;
    if (BY_TOY.indexOf(ch) >= 0) continue;
    if (mode === "seen" && extra.indexOf(ch) >= 0) continue;
    return false;
  }
  return true;
}
function bytesPuzzle(mode, userText) {
  const inputs = BY_PAGE_INPUTS.concat(userText ? [userText] : []);
  const extra = bytesSeenAlphabet(inputs);
  const rows = bytesBaseRows(mode, extra);
  const pageOk = inputs.every((s) => bytesCovers(mode, extra, s));
  const tomorrowOk = BY_TOMORROW.every((s) => bytesCovers(mode, extra, s));
  return { inputs: inputs, extra: extra, rows: rows, params: rows * BY_D, pageOk: pageOk, tomorrowOk: tomorrowOk, small: rows <= BY_BUDGET, pass: pageOk && tomorrowOk && rows <= BY_BUDGET };
}

// ---- screen 3: a word-bounded BPE (slide 24 rules, lexicographic ties as on slide 25) on toy corpora ----
// English letters are one byte each in UTF-8, so characters and bytes give the same merges here.
const BY_CORPORA = {
  A: [["input", 9], ["inputs", 4], ["output", 5], ["town", 6], ["own", 5], ["brown", 4], ["table", 6], ["cable", 3], ["sun", 4], ["under", 3]],
  B: [["unable", 6], ["unpack", 4], ["undo", 5], ["put", 6], ["putting", 3], ["down", 5], ["downtown", 4], ["download", 5], ["readable", 4], ["doable", 3]],
};
const BY_WORD = "unputdownable";
const BY_MORPH = [2, 5, 9]; // un | put | down | able
function bytesMergeSeq(t, a, b) {
  const o = [];
  for (let i = 0; i < t.length; i++) {
    if (i + 1 < t.length && t[i] === a && t[i + 1] === b) {
      o.push(a + b);
      i++;
    } else o.push(t[i]);
  }
  return o;
}
function bytesTrain(corpus, M) {
  let words = corpus.map((wc) => ({ t: wc[0].split("").concat(["</w>"]), c: wc[1] }));
  const rules = [];
  for (let m = 0; m < M; m++) {
    const cnt = {};
    for (const w of words)
      for (let i = 0; i + 1 < w.t.length; i++) {
        const k = w.t[i] + " " + w.t[i + 1];
        cnt[k] = (cnt[k] || 0) + w.c;
      }
    let best = null;
    let bc = 0;
    for (const k in cnt) if (cnt[k] > bc || (cnt[k] === bc && k < best)) {
      best = k;
      bc = cnt[k];
    }
    if (!best) break;
    const ab = best.split(" ");
    rules.push([ab[0], ab[1], bc]);
    words = words.map((w) => ({ t: bytesMergeSeq(w.t, ab[0], ab[1]), c: w.c }));
  }
  return rules;
}
function bytesApply(rules, word) {
  let t = word.split("").concat(["</w>"]);
  for (const r of rules) t = bytesMergeSeq(t, r[0], r[1]);
  return t;
}
// where the token boundaries fall inside the word, and how many sit on morpheme boundaries
function bytesBoundaries(tokens) {
  const cuts = [];
  let pos = 0;
  for (let i = 0; i < tokens.length - 1; i++) {
    pos += tokens[i].replace("</w>", "").length;
    if (pos < BY_WORD.length) cuts.push(pos);
  }
  const onMorph = cuts.filter((c) => BY_MORPH.indexOf(c) >= 0).length;
  return { cuts: cuts, onMorph: onMorph, offMorph: cuts.length - onMorph };
}

// ---- screen 4: normalization rules on a small scraped corpus ----
const BY_CORPUS = [
  { s: "\u0645\u06CC\u200C\u0631\u0648\u0645", w: 0, std: true, how: "standard (ZWNJ)" },
  { s: "\u0645\u06CC \u0631\u0648\u0645", w: 0, std: false, how: "typed with a space" },
  { s: "\u0645\u064A\u200C\u0631\u0648\u0645", w: 0, std: false, how: "Arabic keyboard yeh" },
  { s: "\u06A9\u062A\u0627\u0628", w: 1, std: true, how: "standard" },
  { s: "\u0643\u062A\u0627\u0628", w: 1, std: false, how: "Arabic keyboard kaf" },
  { s: "\uFEDB\uFE98\uFE8E\uFE8F", w: 1, std: false, how: "copied from a PDF" },
  { s: "\u0628\u0647\u200C\u0631\u0648\u0632", w: 2, std: true, how: "standard (ZWNJ)" },
  { s: "\u0628\u0647\u0631\u0648\u0632", w: 3, std: true, how: "standard" },
];
const BY_WORDS = ["I go", "book", "up to date", "Behrouz (a name)"];
function bytesNormalize(s, cfg) {
  let t = s;
  if (cfg.uni === "NFC" || cfg.uni === "NFKC") t = t.normalize(cfg.uni);
  if (cfg.yeh) t = t.replace(/\u064A/g, "\u06CC");
  if (cfg.kaf) t = t.replace(/\u0643/g, "\u06A9");
  if (cfg.sp) t = t.replace(/(^|\s)(\u0645[\u06CC\u064A]) /g, "$1$2\u200C");
  if (cfg.delz) t = t.replace(/\u200C/g, "");
  return t;
}
function bytesCheck(cfg) {
  const out = BY_CORPUS.map((r) => bytesNormalize(r.s, cfg));
  const ids = [];
  const strOf = out.map((o) => {
    let k = ids.indexOf(o);
    if (k < 0) {
      ids.push(o);
      k = ids.length - 1;
    }
    return k;
  });
  const split = [];
  for (let w = 0; w < BY_WORDS.length; w++) {
    const set = [];
    BY_CORPUS.forEach((r, i) => {
      if (r.w === w && set.indexOf(strOf[i]) < 0) set.push(strOf[i]);
    });
    if (set.length > 1) split.push(w);
  }
  const shared = [];
  for (let k = 0; k < ids.length; k++) {
    const ws = [];
    BY_CORPUS.forEach((r, i) => {
      if (strOf[i] === k && ws.indexOf(r.w) < 0) ws.push(r.w);
    });
    if (ws.length > 1) shared.push(ws);
  }
  const damaged = BY_CORPUS.filter((r, i) => r.std && out[i] !== r.s).length;
  return { out: out, strOf: strOf, distinct: ids.length, split: split, shared: shared, damaged: damaged, pass: split.length === 0 && shared.length === 0 };
}
// node check (tools/sim.js):
//   all 48 settings -> only { uni: NFKC, yeh, kaf, sp, no delz } passes; start (all off) has 8 distinct strings.
// @sim-end

const BY_START = { uni: "none", yeh: false, kaf: false, sp: false, delz: false };
const BY_LEN_COLOR = [null, THEME.muted, THEME.accent, THEME.highlight, THEME.third];
const BY_WORD_MARK = ["\u25CF", "\u25A0", "\u25B2", "\u25C6"];
const BY_WORD_COLOR = [THEME.accent, THEME.third, THEME.highlight, THEME.warn];
const BY_PRESETS = [
  ["A \u0634 \uD83E\uDD89", "", "A \u0634 \uD83E\uDD89 (slide 39)"],
  ["\u0645\u06CC\u200C\u0631\u0648\u0645", "\u0645\u06CC\u200C\u0631\u0648\u0645", ""],
  ["\u0645\u06CC \u0631\u0648\u0645", "\u0645\u06CC \u0631\u0648\u0645", ""],
  ["\u0645\u06CC\u0631\u0648\u0645", "\u0645\u06CC\u0631\u0648\u0645", ""],
  ["\u06CC \u064A \u06A9 \u0643", "\u06CC \u064A \u06A9 \u0643", ""],
  ["\u0627\u062C\u062A\u0646\u0627\u0628\u200C\u0646\u0627\u067E\u0630\u06CC\u0631\u0645\u0646\u062F\u06CC", "\u0627\u062C\u062A\u0646\u0627\u0628\u200C\u0646\u0627\u067E\u0630\u06CC\u0631\u0645\u0646\u062F\u06CC", " (slide 4)"],
  ["Unputdownable", "", "Unputdownable (slide 2)"],
];
const BY_SLIDE41 = [
  ["\u0645\u06CC\u200C\u0631\u0648\u0645", "ZWNJ \u00B7 U+200C"],
  ["\u0645\u06CC \u0631\u0648\u0645", "SPACE \u00B7 U+0020"],
  ["\u0645\u06CC\u0631\u0648\u0645", "NO SEPARATOR"],
  ["\u06CC", "PERSIAN YEH"],
  ["\u064A", "ARABIC YEH"],
  ["\u06A9", "PERSIAN KAF"],
  ["\u0643", "ARABIC KAF"],
];
const BY_MAXTILES = 40;

// how to draw one code point so invisible ones can still be seen
function bytesGlyph(ch, cp) {
  if (cp === 0x200c) return { t: "ZWNJ", special: true };
  if (cp === 0x200d) return { t: "ZWJ", special: true };
  if (cp === 0x20) return { t: "␣", special: true };
  if (cp === 0x0a) return { t: "↵", special: true };
  if (cp < 0x20 || cp === 0x7f) return { t: "ctrl", special: true };
  if (cp >= 0xfe00 && cp <= 0xfe0f) return { t: "VS", special: true };
  if (/\p{M}/u.test(ch)) return { t: "◌" + ch, special: false };
  return { t: ch, special: false };
}
function bytesIsRtl(cp) {
  return (cp >= 0x0590 && cp <= 0x08ff) || (cp >= 0xfb1d && cp <= 0xfdff) || (cp >= 0xfe70 && cp <= 0xfeff);
}

// ---------- pieces ----------
function BytesTile({ glyph, special, rtl, line2, line3, edge, bad, dashed, small }) {
  return (
    <div
      style={{
        minWidth: small ? 34 : 50,
        padding: small ? "2px 4px 3px" : "4px 6px 5px",
        borderRadius: 4,
        border: "1px " + (dashed ? "dashed " : "solid ") + (bad ? THEME.warn : edge || THEME.grid),
        background: bad ? rgba(THEME.warn, 0.12) : THEME.bg,
        textAlign: "center",
        lineHeight: 1.25,
      }}
    >
      <div
        lang={rtl ? "fa" : undefined}
        style={{
          fontFamily: special ? MONO : rtl ? FA : SANS,
          fontSize: special ? 11 : small ? 15 : 19,
          color: special ? THEME.highlight : THEME.text,
          minHeight: small ? 18 : 25,
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
        }}
      >
        {special ? "⟨" + glyph + "⟩" : glyph}
      </div>
      {line2 ? <div style={{ fontFamily: MONO, fontSize: 10.5, color: THEME.muted, whiteSpace: "nowrap" }}>{line2}</div> : null}
      {line3 ? <div style={{ fontFamily: MONO, fontSize: 11, color: bad ? THEME.warn : THEME.text, whiteSpace: "nowrap" }}>{line3}</div> : null}
    </div>
  );
}
// screen 1: one input encoded with the chosen base alphabet
function BytesEncodedRow({ s, mode, extra, title }) {
  const cps = bytesCps(s);
  const ok = bytesCovers(mode, extra, s);
  const tiles = [];
  cps.slice(0, BY_MAXTILES).forEach((c, i) => {
    const g = bytesGlyph(c.ch, c.cp);
    const rtl = bytesIsRtl(c.cp);
    if (mode === "bytes") {
      c.bytes.forEach((b, j) => {
        tiles.push(<BytesTile key={i + "-" + j} glyph={j === 0 ? g.t : "·"} special={j === 0 && g.special} rtl={rtl} line2={"byte " + (j + 1) + "/" + c.bytes.length} line3={bytesHex(b)} edge={BY_LEN_COLOR[c.bytes.length]} dashed={j > 0} />);
      });
      return;
    }
    if (/\s/.test(c.ch)) {
      tiles.push(<BytesTile key={i} glyph="␣" special rtl={false} line2="word break" line3="" dashed />);
      return;
    }
    let id = BY_TOY.indexOf(c.ch);
    let known = id >= 0;
    if (known) id = id + 1;
    else if (mode === "seen" && extra.indexOf(c.ch) >= 0) {
      known = true;
      id = 11 + extra.indexOf(c.ch);
    } else if (mode === "unicode") {
      known = true;
      id = bytesU(c.cp);
    }
    tiles.push(<BytesTile key={i} glyph={g.t} special={g.special} rtl={rtl} line2={known ? "id" : "no base symbol"} line3={known ? String(id) : "✕"} bad={!known} />);
  });
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 8, margin: "0 0 8px", flexWrap: "wrap" }}>
      <div style={{ fontFamily: MONO, fontSize: 12, color: ok ? THEME.ok : THEME.warn, width: 112, flexShrink: 0 }}>
        {ok ? "◆ encodes" : "◇ fails"}
        <div style={{ color: THEME.muted, fontSize: 11 }}>{title}</div>
      </div>
      <div style={{ display: "flex", flexWrap: "wrap", gap: 4, minWidth: 0, flex: "1 1 200px" }}>
        {tiles}
        {cps.length > BY_MAXTILES ? <span style={{ fontFamily: MONO, fontSize: 12, color: THEME.muted, alignSelf: "center" }}>+{cps.length - BY_MAXTILES} more</span> : null}
      </div>
    </div>
  );
}
// screens 2-3: the byte inspector, one tile per code point
function BytesInspector({ text }) {
  const cps = bytesCps(text);
  return (
    <div style={{ display: "flex", flexWrap: "wrap", gap: 5 }}>
      {cps.slice(0, BY_MAXTILES).map((c, i) => {
        const g = bytesGlyph(c.ch, c.cp);
        return <BytesTile key={i} glyph={g.t} special={g.special} rtl={bytesIsRtl(c.cp)} line2={bytesU(c.cp)} line3={c.bytes.map(bytesHex).join(" ") + "  ·" + c.bytes.length + "B"} edge={BY_LEN_COLOR[c.bytes.length]} />;
      })}
      {cps.length > BY_MAXTILES ? <span style={{ fontFamily: MONO, fontSize: 12, color: THEME.muted, alignSelf: "center" }}>+{cps.length - BY_MAXTILES} more</span> : null}
      {cps.length === 0 ? <span style={{ fontFamily: MONO, fontSize: 12, color: THEME.muted }}>(empty: type something above)</span> : null}
    </div>
  );
}
function BytesTextBox({ value, onChange, placeholder, id }) {
  const change = useCallback((e) => onChange(e.target.value), [onChange]);
  return (
    <input
      id={id}
      type="text"
      value={value}
      onChange={change}
      maxLength={80}
      placeholder={placeholder}
      dir="auto"
      style={{ width: "100%", boxSizing: "border-box", fontFamily: FA + ", " + SANS, fontSize: 18, padding: "6px 10px", background: THEME.bg, color: THEME.text, border: "1px solid " + THEME.accent, borderRadius: 4, margin: "0 0 8px" }}
    />
  );
}
// one slide-41 string, compact: bytes grouped per code point, separators marked
function BytesSlide41Row({ s, label, onLoad }) {
  const cps = bytesCps(s);
  const n = cps.reduce((a, c) => a + c.bytes.length, 0);
  const click = useCallback(() => onLoad(s), [onLoad, s]);
  return (
    <button
      onClick={click}
      title="load into the inspector"
      style={{ display: "flex", alignItems: "center", gap: 10, width: "100%", textAlign: "left", background: THEME.bg, border: "1px solid " + THEME.grid, borderRadius: 4, padding: "4px 8px", margin: "0 0 4px", cursor: "pointer", color: THEME.text, flexWrap: "wrap" }}
    >
      <span lang="fa" dir="rtl" style={{ fontFamily: FA, fontSize: 18, width: 64, textAlign: "center" }}>
        {s}
      </span>
      <span style={{ fontFamily: MONO, fontSize: 10.5, color: THEME.muted, width: 104 }}>{label}</span>
      <span style={{ fontFamily: MONO, fontSize: 11.5, flex: "1 1 180px", minWidth: 0 }}>
        {cps.map((c, i) => {
          const sep = c.cp === 0x200c || c.cp === 0x20;
          return (
            <span key={i} style={{ color: sep ? THEME.highlight : THEME.text, borderBottom: sep ? "1px dashed " + THEME.highlight : "none", marginRight: 6 }}>
              {c.bytes.map(bytesHex).join(" ")}
            </span>
          );
        })}
      </span>
      <span style={{ fontFamily: MONO, fontSize: 11.5, color: THEME.muted, whiteSpace: "nowrap" }}>
        {cps.length} cp · {n} B
      </span>
    </button>
  );
}
function BytesRuleTable({ text }) {
  const counts = [0, 0, 0, 0, 0];
  for (const c of bytesCps(text)) counts[c.bytes.length]++;
  const rows = [
    [1, "U+0000 – U+007F", "0xxxxxxx", "A → 41"],
    [2, "U+0080 – U+07FF", "110xxxxx 10xxxxxx", "ش → D8 B4"],
    [3, "U+0800 – U+FFFF", "1110xxxx 10xxxxxx ×2", "ZWNJ → E2 80 8C"],
    [4, "U+10000 – U+10FFFF", "11110xxx 10xxxxxx ×3", "🦉 → F0 9F A6 89"],
  ];
  const cell = { padding: "3px 6px", borderBottom: "1px solid " + THEME.grid, whiteSpace: "nowrap" };
  return (
    <div style={{ overflowX: "auto" }}>
      <table style={{ borderCollapse: "collapse", fontFamily: MONO, fontSize: 12, width: "100%" }}>
        <thead>
          <tr style={{ color: THEME.muted, textAlign: "left" }}>
            <th style={cell}>bytes</th>
            <th style={cell}>code points</th>
            <th style={cell}>bit pattern</th>
            <th style={cell}>example</th>
            <th style={cell}>in your text</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r[0]}>
              <td style={Object.assign({}, cell, { color: BY_LEN_COLOR[r[0]] })}>{r[0]} B</td>
              <td style={cell}>{r[1]}</td>
              <td style={cell}>{r[2]}</td>
              <td style={cell}>{r[3]}</td>
              <td style={Object.assign({}, cell, { textAlign: "right" })}>{counts[r[0]]}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
function BytesSplit({ tokens }) {
  return (
    <div style={{ display: "flex", flexWrap: "wrap", gap: 4, margin: "4px 0 2px" }}>
      {tokens.map((t, i) => (
        <span key={i} style={{ fontFamily: MONO, fontSize: 15, padding: "3px 8px", borderRadius: 4, border: "1px solid " + THEME.accent, background: rgba(THEME.accent, 0.12) }}>
          {t}
        </span>
      ))}
    </div>
  );
}
// the word with its letters, cut by token boundaries; morpheme boundaries are marked above
function BytesCutRuler({ cuts }) {
  const letters = BY_WORD.split("");
  const W = 600;
  const step = (W - 40) / letters.length;
  const X = (k) => 20 + k * step;
  return (
    <svg viewBox={"0 0 " + W + " 92"} style={{ width: "100%", display: "block" }} role="img" aria-label="token boundaries versus morpheme boundaries in unputdownable">
      <SvgLabel x={4} y={14} color={THEME.highlight} size={11}>
        ▼ morpheme boundary (un | put | down | able)
      </SvgLabel>
      {BY_MORPH.map((m) => (
        <polygon key={"m" + m} points={X(m) - 6 + ",22 " + (X(m) + 6) + ",22 " + X(m) + ",31"} fill={THEME.highlight} />
      ))}
      {letters.map((ch, i) => (
        <text key={i} x={X(i) + step / 2} y={56} textAnchor="middle" fontFamily={MONO} fontSize={20} fill={THEME.text}>
          {ch}
        </text>
      ))}
      {cuts.map((c) => (
        <line key={"c" + c} x1={X(c)} x2={X(c)} y1={34} y2={66} stroke={BY_MORPH.indexOf(c) >= 0 ? THEME.ok : THEME.warn} strokeWidth={2.5} strokeDasharray={BY_MORPH.indexOf(c) >= 0 ? null : "4 3"} />
      ))}
      <SvgLabel x={4} y={86} color={THEME.muted} size={11}>
        │ token boundary: solid = on a morpheme boundary, dashed = inside a morpheme
      </SvgLabel>
    </svg>
  );
}
function BytesCorpusTable({ res }) {
  const cell = { padding: "4px 6px", borderBottom: "1px solid " + THEME.grid, verticalAlign: "middle" };
  return (
    <div style={{ overflowX: "auto" }}>
      <table style={{ borderCollapse: "collapse", width: "100%", fontSize: 13 }}>
        <thead>
          <tr style={{ fontFamily: MONO, fontSize: 11, color: THEME.muted, textAlign: "left" }}>
            <th style={cell}>means</th>
            <th style={cell}>as found</th>
            <th style={cell}>after your rules</th>
            <th style={cell}>code points after</th>
            <th style={cell}>string</th>
          </tr>
        </thead>
        <tbody>
          {BY_CORPUS.map((r, i) => {
            const out = res.out[i];
            const changed = out !== r.s;
            const k = res.strOf[i];
            const shared = res.shared.some((ws) => ws.indexOf(r.w) >= 0 && res.strOf.some((kk, j) => kk === k && BY_CORPUS[j].w !== r.w));
            return (
              <tr key={i}>
                <td style={Object.assign({}, cell, { fontFamily: MONO, fontSize: 12, color: BY_WORD_COLOR[r.w], whiteSpace: "nowrap" })}>
                  {BY_WORD_MARK[r.w]} {BY_WORDS[r.w]}
                </td>
                <td style={cell}>
                  <span lang="fa" dir="rtl" style={{ fontFamily: FA, fontSize: 18, whiteSpace: "nowrap" }}>
                    {r.s}
                  </span>
                  <div style={{ fontFamily: MONO, fontSize: 10.5, color: THEME.muted }}>{r.how}</div>
                </td>
                <td style={cell}>
                  <span lang="fa" dir="rtl" style={{ fontFamily: FA, fontSize: 18, whiteSpace: "nowrap", color: changed ? THEME.highlight : THEME.text }}>
                    {out}
                  </span>
                  <div style={{ fontFamily: MONO, fontSize: 10.5, color: THEME.muted }}>{changed ? "changed" : "unchanged"}</div>
                </td>
                <td style={Object.assign({}, cell, { fontFamily: MONO, fontSize: 10.5, color: THEME.muted, lineHeight: 1.4 })}>
                  {bytesCps(out).map((c, j) => {
                    const odd = c.cp === 0x200c || c.cp === 0x20 || c.cp === 0x064a || c.cp === 0x0643 || c.cp >= 0xfb50;
                    return (
                      <span key={j} style={{ color: odd ? THEME.highlight : THEME.muted, marginRight: 4, whiteSpace: "nowrap" }}>
                        {c.cp === 0x200c ? "ZWNJ" : c.cp === 0x20 ? "SPACE" : bytesU(c.cp).slice(2)}
                      </span>
                    );
                  })}
                  <div style={{ color: THEME.text }}>{bytesOf(out).length} bytes</div>
                </td>
                <td style={Object.assign({}, cell, { fontFamily: MONO, fontSize: 13, color: shared ? THEME.warn : THEME.text, whiteSpace: "nowrap" })}>
                  S{k + 1}
                  {shared ? " ✕" : ""}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

export default function BytesLesson({ initialScreen }) {
  const [screen, setScreen] = useState(initialScreen || 0);
  const [mode, setMode] = useState("toy");
  const [ptxt, setPtxt] = useState("");
  const [txt, setTxt] = useState("A ش 🦉");
  const [corpus, setCorpus] = useState("A");
  const [k, setK] = useState(0);
  const [cfg, setCfg] = useState(BY_START);

  const setUni = useCallback((v) => setCfg((c) => Object.assign({}, c, { uni: v })), []);
  const setYeh = useCallback((v) => setCfg((c) => Object.assign({}, c, { yeh: v })), []);
  const setKaf = useCallback((v) => setCfg((c) => Object.assign({}, c, { kaf: v })), []);
  const setSp = useCallback((v) => setCfg((c) => Object.assign({}, c, { sp: v })), []);
  const setDelz = useCallback((v) => setCfg((c) => Object.assign({}, c, { delz: v })), []);
  const resetCfg = useCallback(() => setCfg(BY_START), []);
  const pickCorpus = useCallback((v) => {
    setCorpus(v);
    setK(0);
  }, []);

  let question, stage, side;
  if (screen === 0) {
    const r = bytesPuzzle(mode, ptxt);
    question = (
      <>
        <b>Where we are:</b> lessons 3–4 built a BPE tokenizer on the toy alphabet d e i l n o r s t w (slides 25, 29). Lesson 4 encoded the unseen word <i>lowest</i> from known pieces (slide 31). <b>The puzzle:</b> can the same tokenizer encode <Fa>ژ</Fa> or 🦉 (slide 38)? Pick a base alphabet that encodes every input on this page, still works on inputs nobody has seen yet, and stays under {BY_BUDGET.toLocaleString("en-US")} rows in the embedding table.
      </>
    );
    stage = (
      <>
        <Label>inputs on this page</Label>
        {r.inputs.map((s, i) => (
          <BytesEncodedRow key={"p" + i} s={s} mode={mode} extra={r.extra} title={i === 0 ? "slide 31" : i < 3 ? "slide 38" : "your text"} />
        ))}
        <BytesTextBox id="by-ptxt" value={ptxt} onChange={setPtxt} placeholder="type your own input: Persian, English, emoji…" />
        <Label>tomorrow's inputs (not seen when the alphabet was chosen)</Label>
        {BY_TOMORROW.map((s, i) => (
          <BytesEncodedRow key={"t" + i} s={s} mode={mode} extra={r.extra} title="new input" />
        ))}
        <Note>Each tile is one base symbol with its ID. ✕ means the alphabet has no symbol for that character, so the input cannot be encoded at all: no merge rule can help, because merges only join symbols that exist.</Note>
      </>
    );
    side = (
      <>
        <Panel>
          <Label>base alphabet</Label>
          <Pills
            options={[
              ["toy", "toy: d e i l n o r s t w"],
              ["seen", "toy + every character on this page"],
              ["unicode", "every Unicode code point"],
              ["bytes", "the 256 byte values"],
            ]}
            value={mode}
            onPick={setMode}
          />
          <Readout
            rows={[
              ["base rows (incl. </w>)", r.rows.toLocaleString("en-US"), r.small ? THEME.text : THEME.warn],
              ["embedding params at d = 768", r.params.toLocaleString("en-US")],
              ["page inputs encoded", r.inputs.filter((s) => bytesCovers(mode, r.extra, s)).length + " / " + r.inputs.length],
              ["tomorrow's inputs encoded", BY_TOMORROW.filter((s) => bytesCovers(mode, r.extra, s)).length + " / " + BY_TOMORROW.length],
            ]}
          />
          <Goal met={r.pageOk}>every input on this page encodes</Goal>
          <Goal met={r.tomorrowOk}>tomorrow's inputs encode too</Goal>
          <Goal met={r.small}>base table ≤ {BY_BUDGET.toLocaleString("en-US")} rows</Goal>
        </Panel>
        <Note>An unseen word made of known characters is fine (lowest). An unseen character is a different problem: it has no base symbol (slide 38). In the byte option, why does <Fa>ش</Fa> become two tiles and 🦉 four?</Note>
      </>
    );
  } else if (screen === 1) {
    const cps = bytesCps(txt);
    const nb = cps.reduce((a, c) => a + c.bytes.length, 0);
    const has = [0, 1, 2, 3, 4].map((L) => cps.some((c) => c.bytes.length === L));
    question = (
      <>
        <b>Claim to test:</b> a byte-level tokenizer does not see letters. It first turns text into UTF-8 bytes, and those bytes are its base tokens (slide 39: A = 41, <Fa>ش</Fa> = D8 B4, 🦉 = F0 9F A6 89). Type or load text. Which characters take 1, 2, 3 or 4 bytes? Then click the three spellings of <Fa>{"می‌روم"}</Fa> from slide 41: does the tokenizer get the same bytes for all three?
      </>
    );
    stage = (
      <>
        <BytesTextBox id="by-txt" value={txt} onChange={setTxt} placeholder="type anything" />
        <Pills
          options={BY_PRESETS.map((p) => [
            p[0],
            <>
              {p[1] ? <Fa>{p[1]}</Fa> : null}
              {p[2]}
            </>,
          ])}
          value={txt}
          onPick={setTxt}
        />
        <BytesInspector text={txt} />
        <Note>One tile per code point, in memory order: left to right is the order the tokenizer reads, even for Persian. Tile edge colour and the "·nB" count both give the byte length. Invisible characters are shown in ⟨ ⟩.</Note>
        <Label>slide 41: similar text, different strings (click to inspect)</Label>
        {BY_SLIDE41.map((p) => (
          <BytesSlide41Row key={p[1]} s={p[0]} label={p[1]} onLoad={setTxt} />
        ))}
      </>
    );
    side = (
      <>
        <Panel>
          <Readout
            rows={[
              ["code points", String(cps.length)],
              ["UTF-8 bytes = base tokens", String(nb)],
              ["bytes per code point", cps.length ? fmt(nb / cps.length, 2) : "—"],
            ]}
          />
          <Goal met={has[1]}>a character that takes 1 byte</Goal>
          <Goal met={has[2]}>a character that takes 2 bytes</Goal>
          <Goal met={has[3]}>a character that takes 3 bytes</Goal>
          <Goal met={has[4]}>a character that takes 4 bytes</Goal>
        </Panel>
        <Note>The highlighted bytes in the slide-41 rows are the separator: ZWNJ is E2 80 8C (3 bytes), a space is 20 (1 byte), and the third spelling has none. Persian yeh <Fa>ی</Fa> is DB 8C, Arabic yeh <Fa>ي</Fa> is D9 8A. They look alike on screen; to the tokenizer they share no byte.</Note>
      </>
    );
  } else if (screen === 2) {
    const rules = bytesTrain(BY_CORPORA[corpus], 60);
    const kk = Math.min(k, rules.length);
    const tokens = bytesApply(rules.slice(0, kk), BY_WORD);
    const b = bytesBoundaries(tokens);
    const last = kk > 0 ? rules[kk - 1] : null;
    question = (
      <>
        <b>What you found:</b> UTF-8 writes every code point as 1 to 4 bytes, so 256 base values cover any input and nothing is ever unknown. A Persian letter costs 2 bytes, ZWNJ 3, an emoji 4. Look-alike strings are different bytes, so different tokens. But coverage ≠ understanding (slide 39). <b>One more question (slide 40):</b> you read <i>unputdownable</i> as un + put + down + able (slides 2–3). BPE only asks which adjacent pairs are frequent. Train it on two toy corpora: do its boundaries land on the morphemes?
      </>
    );
    stage = (
      <>
        <Eq size={13}>
          Text → UTF-8 bytes → BPE merges → token IDs (slide 39)
          <br />
          base vocabulary = 256 byte values; merges add longer pieces on top
        </Eq>
        <BytesRuleTable text={txt} />
        <Note>"In your text" counts the characters you left in the inspector on screen 2.</Note>
        <Label>BPE trained on toy corpus {corpus}, first {kk} merges, applied to unputdownable</Label>
        <BytesSplit tokens={tokens} />
        <BytesCutRuler cuts={b.cuts} />
      </>
    );
    side = (
      <>
        <Panel>
          <Label>training corpus (word × count, made up)</Label>
          <Pills
            options={[
              ["A", "A: input, output, town, own…"],
              ["B", "B: unable, undo, put, down…"],
            ]}
            value={corpus}
            onPick={pickCorpus}
          />
          <div style={{ fontFamily: MONO, fontSize: 11, color: THEME.muted, lineHeight: 1.5, margin: "0 0 6px" }}>{BY_CORPORA[corpus].map((wc) => wc[0] + "×" + wc[1]).join("  ")}</div>
          <Slider id="by-k" label="merges learned" value={kk} min={0} max={40} stepSize={1} onChange={setK} display={kk + (k >= rules.length ? " (all " + rules.length + ")" : "")} />
          <Readout
            rows={[
              ["last merge", last ? last[0] + " + " + last[1] + "  (count " + last[2] + ")" : "—"],
              ["tokens", String(tokens.length) + " (incl. </w>)"],
              ["cuts on a morpheme boundary", b.onMorph + " of 3 morpheme boundaries"],
              ["cuts inside a morpheme", String(b.offMorph), b.offMorph ? THEME.warn : THEME.text],
            ]}
          />
        </Panel>
        <Note>Corpus A ends at u | n | put | d | ow | n | able: <i>down</i> is cut into d + own, because own, town and brown made "ow" frequent. Corpus B ends at un | put | down | able. Same algorithm, different data, different boundaries. Neither run was told what a morpheme is (slide 40).</Note>
        <Note>
          <b>Next:</b> lesson 7 measures what bytes cost. A Persian letter is 2 bytes and an English one is 1, so the same sentence can need many more tokens in Persian (the "token tax", slides 42–48).
        </Note>
      </>
    );
  } else {
    const res = bytesCheck(cfg);
    question = (
      <>
        Slide 41 asks: should we normalize, and what might we lose? Here are eight strings scraped from the web, with what each one means. Turn on normalization rules (they run top to bottom) until the tokenizer sees <b>exactly one string per word</b>, without ever giving <b>two different words the same string</b>. <Fa>{"به‌روز"}</Fa> (up to date) and <Fa>بهروز</Fa> (the name Behrouz) differ only by a ZWNJ.
      </>
    );
    stage = (
      <>
        <BytesCorpusTable res={res} />
        <Note>Same S-number = byte-identical strings, so identical tokens. ✕ marks a string shared by two different words. Code points highlighted: separators, Arabic letters, and presentation forms (U+FBxx–U+FExx, what PDF text extraction often returns).</Note>
      </>
    );
    side = (
      <>
        <Panel edge={res.pass ? THEME.ok : THEME.grid}>
          <Label>1. Unicode normalization</Label>
          <Pills
            options={[
              ["none", "none"],
              ["NFC", "NFC"],
              ["NFKC", "NFKC"],
            ]}
            value={cfg.uni}
            onPick={setUni}
          />
          <Label>2–5. Persian rules</Label>
          <div>
            <Toggle label={<>Arabic yeh <Fa>{"\u064A"}</Fa> U+064A → Persian yeh <Fa>{"\u06CC"}</Fa> U+06CC</>} value={cfg.yeh} onChange={setYeh} />
            <Toggle label={<>Arabic kaf <Fa>{"\u0643"}</Fa> U+0643 → Persian kaf <Fa>{"\u06A9"}</Fa> U+06A9</>} value={cfg.kaf} onChange={setKaf} />
            <Toggle label={<>space after word-initial <Fa>{"\u0645\u06CC"}</Fa> → ZWNJ</>} value={cfg.sp} onChange={setSp} />
            <Toggle label="delete every ZWNJ" value={cfg.delz} onChange={setDelz} />
          </div>
          <Readout
            rows={[
              ["distinct strings", res.distinct + " (goal: 4 words → 4)", res.distinct === 4 ? THEME.text : THEME.warn],
              ["words still split", res.split.length ? res.split.map((w) => BY_WORDS[w]).join(", ") : "none", res.split.length ? THEME.warn : THEME.text],
              ["words merged together", res.shared.length ? res.shared.map((ws) => ws.map((w) => BY_WORDS[w]).join(" = ")).join("; ") : "none", res.shared.length ? THEME.warn : THEME.text],
              ["standard spellings changed", String(res.damaged)],
            ]}
          />
          <Goal met={res.split.length === 0}>every word is one string</Goal>
          <Goal met={res.shared.length === 0}>no two words share a string</Goal>
          <Btn onClick={resetCfg}>turn all rules off</Btn>
          <Hint
            hints={[
              "Take the split words one at a time. In the code-point column, find the code point that differs between two spellings of the same word: that tells you which rule it needs.",
              "The PDF copy uses presentation forms (U+FExx). NFKC maps them back to ordinary letters, but it returns the Arabic kaf U+0643: Unicode normalization alone does not do the Persian replacement (slide 41).",
              "Deleting every ZWNJ does unify the three spellings of می‌روم, but look at what it does to به‌روز.",
            ]}
          />
        </Panel>
      </>
    );
  }

  return <LessonFrame screen={screen} onScreen={setScreen} title="Bytes, Persian text, and meaning" question={question} stage={stage} side={side} />;
}
