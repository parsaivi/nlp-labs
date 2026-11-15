import { useState, useCallback, useMemo } from "react";
import { THEME, MONO } from "./common";

// Lesson 5 - Special tokens and vocabulary size (Tokenization, slides 33-37).
// Aha: some rows of the vocabulary are not word pieces. [BOS], [EOS], [PAD] are configured, get IDs at the end of
// the table and learned embeddings like any token. Padding makes a batch rectangular, but the filler token does
// not protect anything: the padding mask, a separate 0/1 array, is what keeps attention and the loss away from
// the filler. Every token, merged or special, is one row of the V × d embedding table, so more merges buy shorter
// sequences with more parameters, and the saving has to be measured on held-out text.
//
// Numbers from the slides: lesson 4's lowest -> [low, est</w>] -> [15, 13]; base IDs 0-10 and merges es=11 ... low=15
// (slide 29); [BOS] 16, [EOS] 17, 16 BPE + 2 special = V = 18 (slide 33); the special-token roles (slide 34); the
// gpt-oss / Harmony chat markers exactly as printed on slide 35; the batch "I read books [EOS]" / "Hi [EOS] [PAD] [PAD]"
// and its padding mask [1, 1, 0, 0] (slide 36); at d = 768, 16k -> 12.3M, 32k -> 24.6M, 64k -> 49.2M (slide 37).
// Teaching toys: the 2-D word embeddings in the batch puzzle and their attention outputs; the 30-sentence
// training corpus and 4 held-out sentences (made up for this lesson), on which real word-bounded BPE (the deck's
// algorithm and tie rule, slides 24-29) is run, so the length-vs-V curve and the challenge batch are computed, not
// drawn; d = 64 and the 6,400-parameter / 24-position budgets in the challenge.

// @sim-start
const SP_TRAIN = [
  "the students read books in the library every day",
  "the teacher reads a new book to the class every week",
  "students who read more books learn new words faster",
  "the library keeps old books and new books on every shelf",
  "the newest books stand on the lowest shelf near the door",
  "the oldest books stand on the widest shelf near the window",
  "a student reads the lower shelf first and the higher shelf later",
  "the teacher and the students talk about the books they read",
  "the class reads together and the teacher listens to every student",
  "every student keeps a list of the new words in the books",
  "the list of words grows longer every week",
  "students read slowly at first and faster later",
  "the teacher reads the newest book to the class",
  "the library opens early and closes late",
  "students borrow books and return them every week",
  "the widest window lets the light into the library",
  "the teacher writes the new words on the board",
  "students learn the words and read them again",
  "a good book teaches new words to every reader",
  "the readers talk about the book after the class",
  "the students opened their books and started reading",
  "the teacher walked to the board and started writing",
  "reading every day makes a reader faster",
  "the oldest reader in the class reads the longest books",
  "the students wanted newer books for the library",
  "the teacher asked the students about the words they wanted to learn",
  "writing new words helps students remember them",
  "the library worker counted the books on the shelf",
  "students played outside after the class ended",
  "the teacher smiled and the students laughed",
];
const SP_HELD = ["the readers borrowed the newest books", "hi", "teachers listened while students learned new words", "the lowest shelf holds older books"];
const SP_D = 64;
const SP_PARAM_BUDGET = 6400;
const SP_LEN_BUDGET = 24;
const SP_NSPECIAL = 3; // [BOS] [EOS] [PAD]

function spSplit(w) {
  return w.split("").concat(["</w>"]);
}
function spLess(a, b) {
  return a[0] < b[0] || (a[0] === b[0] && a[1] < b[1]);
}
function spMergeSeq(seq, a, b) {
  const out = [];
  let i = 0;
  while (i < seq.length) {
    if (i < seq.length - 1 && seq[i] === a && seq[i + 1] === b) {
      out.push(a + b);
      i += 2;
    } else {
      out.push(seq[i]);
      i++;
    }
  }
  return out;
}
// word-bounded BPE as on slides 24-29: weighted pair counts, ties by lexicographic pair order
function spTrain(lines, maxMerges) {
  const freq = {};
  let nWords = 0;
  for (const line of lines)
    for (const w of line.split(" ")) {
      freq[w] = (freq[w] || 0) + 1;
      nWords++;
    }
  const words = Object.keys(freq).sort();
  let segs = words.map(spSplit);
  const base = {};
  for (const s of segs) for (const ch of s) base[ch] = 1;
  const rules = [];
  const trainLen = [];
  const total = () => segs.reduce((s, sg, i) => s + sg.length * freq[words[i]], 0);
  trainLen.push(total());
  for (let m = 0; m < maxMerges; m++) {
    const c = {};
    for (let i = 0; i < words.length; i++) {
      const s = segs[i];
      for (let j = 0; j < s.length - 1; j++) {
        const k = s[j] + "\u0001" + s[j + 1];
        c[k] = (c[k] || 0) + freq[words[i]];
      }
    }
    let best = null;
    let bc = 0;
    for (const k in c) {
      const p = k.split("\u0001");
      if (c[k] > bc || (c[k] === bc && spLess(p, best))) {
        best = p;
        bc = c[k];
      }
    }
    if (!best) break;
    rules.push([best[0], best[1], bc]);
    segs = segs.map((s) => spMergeSeq(s, best[0], best[1]));
    trainLen.push(total());
  }
  return { rules: rules, trainLen: trainLen, base: Object.keys(base).sort(), nWords: nWords };
}
function spRank(rules, M) {
  const rank = {};
  for (let r = 0; r < M && r < rules.length; r++) rank[rules[r][0] + "\u0001" + rules[r][1]] = r;
  return rank;
}
// inference as in lesson 4: repeatedly merge the eligible pair with the earliest rank
function spApply(word, rank) {
  let s = spSplit(word);
  for (;;) {
    let br = Infinity;
    let bp = null;
    for (let j = 0; j < s.length - 1; j++) {
      const r = rank[s[j] + "\u0001" + s[j + 1]];
      if (r !== undefined && r < br) {
        br = r;
        bp = [s[j], s[j + 1]];
      }
    }
    if (!bp) break;
    s = spMergeSeq(s, bp[0], bp[1]);
  }
  return s;
}
function spEncodeLine(line, rank) {
  let out = [];
  for (const w of line.split(" ")) out = out.concat(spApply(w, rank));
  return out;
}
// everything the Explore and Challenge screens read, for M = 0 .. all merges
function spTable() {
  const T = spTrain(SP_TRAIN, 400);
  const heldWords = SP_HELD.join(" ").split(" ").length;
  const rows = [];
  for (let M = 0; M <= T.rules.length; M++) {
    const rank = spRank(T.rules, M);
    const enc = SP_HELD.map((l) => ["[BOS]"].concat(spEncodeLine(l, rank), ["[EOS]"]));
    const lens = enc.map((e) => e.length);
    const held = lens.reduce((a, b) => a + b, 0) - 2 * SP_HELD.length;
    rows.push({
      M: M,
      V: T.base.length + M + SP_NSPECIAL,
      trainPerWord: T.trainLen[M] / T.nWords,
      heldPerWord: held / heldWords,
      held: held,
      enc: enc,
      lens: lens,
      maxLen: Math.max.apply(null, lens),
    });
  }
  return { T: T, rows: rows };
}
function spChallenge(row) {
  const params = row.V * SP_D;
  return { params: params, okParams: params <= SP_PARAM_BUDGET, okLen: row.maxLen <= SP_LEN_BUDGET, ok: params <= SP_PARAM_BUDGET && row.maxLen <= SP_LEN_BUDGET };
}

// ---- the batch puzzle (slide 36): toy 2-D embeddings, one bidirectional attention layer, q = k = v = embedding
const SP_EMB = {
  I: [1.0, 0.3],
  read: [0.2, 1.0],
  books: [0.8, 0.8],
  Hi: [1.0, 0.5],
  "[EOS]": [-0.6, 0.7],
  "[PAD]": [0.4, -0.9],
};
const SP_ROW1 = ["I", "read", "books", "[EOS]"];
const SP_ROW2 = ["Hi", "[EOS]"];
// outputs at every position of one row; mask[j] = 0 hides key j (score + −∞)
function spAttend(tokens, mask) {
  return tokens.map((qt, i) => {
    if (!qt) return { w: tokens.map(() => 0), o: [NaN, NaN] };
    const q = SP_EMB[qt];
    const s = tokens.map((kt, j) => (!kt || !mask[j] ? -Infinity : dot(q, SP_EMB[kt])));
    const w = softmax(s);
    const o = [0, 0];
    tokens.forEach((kt, j) => {
      if (kt && w[j] > 0) {
        o[0] += w[j] * SP_EMB[kt][0];
        o[1] += w[j] * SP_EMB[kt][1];
      }
    });
    return { w: w, o: o };
  });
}
// how far the real positions' outputs move compared with running the sentence alone
function spShift(real, padded, mask) {
  const alone = spAttend(real, real.map(() => 1));
  const inBatch = spAttend(padded, mask);
  let worst = 0;
  for (let i = 0; i < real.length; i++) {
    const a = alone[i].o;
    const b = inBatch[i].o;
    const d = Math.hypot(a[0] - b[0], a[1] - b[1]);
    worst = isFinite(d) ? Math.max(worst, d) : Infinity;
  }
  return { worst: worst, alone: alone, inBatch: inBatch };
}
function spPuzzle(filler, mask1, mask2) {
  const row2 = SP_ROW2.concat([filler, filler]);
  const r1 = spShift(SP_ROW1, SP_ROW1, mask1);
  const r2 = spShift(SP_ROW2, row2, mask2);
  // empty cells (no filler chosen yet) are not positions at all, so they never count
  const row2Cells = filler ? mask2 : mask2.slice(0, SP_ROW2.length);
  const lossPos = mask1.reduce((a, b) => a + b, 0) + row2Cells.reduce((a, b) => a + b, 0);
  const lossOnFiller = !!filler && (mask2[2] + mask2[3] > 0);
  return {
    rect: !!filler,
    r1: r1,
    r2: r2,
    same: !!filler && r1.worst < 1e-9 && r2.worst < 1e-9,
    lossPos: lossPos,
    lossOk: !!filler && !lossOnFiller && lossPos === 6,
  };
}
// @sim-end


let SP_CACHE = null;
function spData() {
  if (!SP_CACHE) SP_CACHE = spTable();
  return SP_CACHE;
}
function spIsSpecial(t) {
  return !!t && (/^\[[A-Z]+\]$/.test(t) || /^<\|.*\|>$/.test(t));
}
// ---------- the batch: one row per sentence, and the padding mask drawn as a separate array ----------
// rows[r][j]: token or null (empty cell); mask[r][j]: 0/1; onMask(r, j) makes the mask cells clickable;
// weights[r][j]: attention weight from the row's first token (a bar inside the cell)
function SpBatch({ rows, mask, onMask, weights, ariaLabel }) {
  const n = rows.reduce((a, r) => Math.max(a, r.length), 0);
  const cw = 76;
  const ch = 44;
  const lw = 56;
  const mw = 34;
  const gap = 22;
  const maskX = lw + n * cw + gap;
  const W = maskX + n * mw + 6;
  const H = 24 + rows.length * (ch + 6);
  const click = useCallback(
    (e) => {
      if (!onMask) return;
      const g = e.target.closest ? e.target.closest("[data-m]") : null;
      if (!g) return;
      const rc = g.getAttribute("data-m").split(",");
      onMask(parseInt(rc[0], 10), parseInt(rc[1], 10));
    },
    [onMask]
  );
  return (
    <svg viewBox={"0 0 " + W + " " + H} onClick={click} style={{ width: "100%", maxWidth: W * 1.15, display: "block", background: THEME.bg, border: "1px solid " + THEME.grid, borderRadius: 6 }} role="img" aria-label={ariaLabel || "a batch of token sequences"}>
      <text x={lw} y={15} fontFamily={MONO} fontSize={11} fill={THEME.muted}>
        tokens
      </text>
      <text x={maskX} y={15} fontFamily={MONO} fontSize={11} fill={THEME.muted}>
        padding mask
      </text>
      {rows.map((row, r) => {
        const y = 22 + r * (ch + 6);
        const cells = [];
        for (let j = 0; j < n; j++) {
          const t = row[j];
          const x = lw + j * cw;
          const sp = spIsSpecial(t);
          const pad = t === "[PAD]";
          const masked = mask && mask[r] && mask[r][j] === 0;
          cells.push(
            <g key={j} opacity={masked ? 0.45 : 1}>
              <rect x={x + 1} y={y + 1} width={cw - 2} height={ch - 2} rx={3} fill={t ? (sp ? rgba(THEME.highlight, 0.13) : THEME.surface) : "transparent"} stroke={!t ? THEME.warn : sp ? THEME.highlight : THEME.grid} strokeDasharray={!t || pad ? "3 3" : undefined} strokeWidth={1} />
              <text x={x + cw / 2} y={y + ch / 2 + 1} textAnchor="middle" fontFamily={MONO} fontSize={13} fill={!t ? THEME.warn : sp ? THEME.highlight : THEME.text}>
                {t || "empty"}
              </text>
              {weights && weights[r] && t ? <rect x={x + 6} y={y + ch - 11} width={Math.max(0, (cw - 12) * (weights[r][j] || 0))} height={5} fill={THEME.third} /> : null}
            </g>
          );
        }
        const maskPart =
          mask && mask[r]
            ? mask[r].map((v, j) => (
                <g key={"m" + j} data-m={r + "," + j} style={{ cursor: onMask ? "pointer" : "default" }}>
                  <rect x={maskX + j * mw + 1} y={y + 6} width={mw - 2} height={ch - 12} rx={3} fill={v ? rgba(THEME.accent, 0.3) : THEME.bg} stroke={onMask ? THEME.accent : THEME.grid} strokeWidth={1} />
                  <text x={maskX + j * mw + mw / 2} y={y + ch / 2 + 5} textAnchor="middle" fontFamily={MONO} fontSize={14} fill={THEME.text}>
                    {v}
                  </text>
                </g>
              ))
            : null;
        return (
          <g key={r}>
            <text x={lw - 8} y={y + ch / 2 + 4} textAnchor="end" fontFamily={MONO} fontSize={11.5} fill={THEME.muted}>
              {"row " + (r + 1)}
            </text>
            {cells}
            {maskPart}
          </g>
        );
      })}
    </svg>
  );
}

// ---------- the challenge batch, turned on its side: one column per sentence, positions run downward ----------
// enc[c]: the tokens of sentence c with [BOS] and [EOS]; every column is padded to the longest; budget draws a line
function SpColumns({ enc, budget }) {
  const n = enc.reduce((a, e) => Math.max(a, e.length), 0);
  const rh = 16;
  const tw = 112;
  const mw = 22;
  const gap = 12;
  const lw = 30;
  const top = 36;
  const rowsShown = Math.max(n, budget);
  const colsW = lw + enc.length * (tw + mw + gap);
  const W = colsW + 64;
  const H = top + rowsShown * rh + 8;
  const yB = top + budget * rh;
  const over = n > budget;
  const ticks = [];
  for (let j = 0; j < rowsShown; j += 5) ticks.push(j);
  return (
    <svg viewBox={"0 0 " + W + " " + H} style={{ width: "100%", maxHeight: 640, display: "block", background: THEME.bg, border: "1px solid " + THEME.grid, borderRadius: 6 }} role="img" aria-label="the four held-out sentences as columns: tokens, padding and the padding mask">
      {ticks.map((j) => (
        <text key={"t" + j} x={lw - 6} y={top + j * rh + 12} textAnchor="end" fontFamily={MONO} fontSize={9.5} fill={THEME.muted}>
          {j + 1}
        </text>
      ))}
      {enc.map((e, c) => {
        const x0 = lw + c * (tw + mw + gap);
        const cells = [];
        for (let j = 0; j < n; j++) {
          const t = j < e.length ? e[j] : "[PAD]";
          const sp = spIsSpecial(t);
          const pad = j >= e.length;
          cells.push(
            <g key={j}>
              <rect x={x0} y={top + j * rh + 1} width={tw} height={rh - 2} rx={2} fill={sp ? rgba(THEME.highlight, pad ? 0.06 : 0.14) : THEME.surface} stroke={sp ? THEME.highlight : THEME.grid} strokeDasharray={pad ? "3 3" : undefined} strokeWidth={0.8} />
              <text x={x0 + 6} y={top + j * rh + 11.5} fontFamily={MONO} fontSize={10.5} fill={sp ? THEME.highlight : THEME.text} opacity={pad ? 0.7 : 1}>
                {t}
              </text>
            </g>
          );
        }
        const xm = x0 + tw + 3;
        return (
          <g key={c}>
            <text x={x0} y={14} fontFamily={MONO} fontSize={11} fill={THEME.muted}>
              {"row " + (c + 1) + " · " + e.length}
            </text>
            <text x={xm + mw / 2} y={28} textAnchor="middle" fontFamily={MONO} fontSize={9} fill={THEME.muted}>
              mask
            </text>
            {cells}
            <rect x={xm} y={top + 1} width={mw - 3} height={e.length * rh - 2} rx={2} fill={rgba(THEME.accent, 0.3)} stroke={THEME.accent} strokeWidth={0.8} />
            <text x={xm + (mw - 3) / 2} y={top + (e.length * rh) / 2 + 4} textAnchor="middle" fontFamily={MONO} fontSize={11} fill={THEME.text}>
              1
            </text>
            {n > e.length ? (
              <g>
                <rect x={xm} y={top + e.length * rh + 1} width={mw - 3} height={(n - e.length) * rh - 2} rx={2} fill={THEME.bg} stroke={THEME.grid} strokeWidth={0.8} />
                <text x={xm + (mw - 3) / 2} y={top + e.length * rh + ((n - e.length) * rh) / 2 + 4} textAnchor="middle" fontFamily={MONO} fontSize={11} fill={THEME.text}>
                  0
                </text>
              </g>
            ) : null}
          </g>
        );
      })}
      <line x1={lw - 4} x2={W - 4} y1={yB} y2={yB} stroke={over ? THEME.warn : THEME.ok} strokeWidth={1.6} strokeDasharray="6 4" />
      <SvgLabel x={W - 4} y={yB - 18} anchor="end" color={over ? THEME.warn : THEME.ok} size={10.5}>
        {(over ? "✕" : "✓") + " budget"}
      </SvgLabel>
      <SvgLabel x={W - 4} y={yB - 5} anchor="end" color={over ? THEME.warn : THEME.ok} size={10.5}>
        {budget + " tokens"}
      </SvgLabel>
    </svg>
  );
}

// ---------- tokens per word and table size against the number of merges ----------
function SpCurve({ rows, M }) {
  const W = 600;
  const pl = 52;
  const pr = 14;
  const top = { y0: 18, h: 170 };
  const bot = { y0: 228, h: 92 };
  const H = 350;
  const mMax = rows.length - 1;
  const X = (m) => pl + (m / mMax) * (W - pl - pr);
  const Yt = (v) => top.y0 + top.h - (v / 7) * top.h;
  const pMax = rows[mMax].V * SP_D;
  const Yb = (v) => bot.y0 + bot.h - (v / pMax) * bot.h;
  const path = (f, Y) => rows.map((r, i) => (i ? "L" : "M") + X(r.M).toFixed(1) + "," + Y(f(r)).toFixed(1)).join(" ");
  const cur = rows[M];
  const yt = [0, 1, 2, 3, 4, 5, 6, 7];
  const xt = [0, 50, 100, 150, 200];
  return (
    <svg viewBox={"0 0 " + W + " " + H} style={{ width: "100%", display: "block", background: THEME.bg, border: "1px solid " + THEME.grid, borderRadius: 6 }} role="img" aria-label="tokens per word and embedding parameters against the number of merges">
      {yt.map((v) => (
        <g key={"t" + v}>
          <line x1={pl} x2={W - pr} y1={Yt(v)} y2={Yt(v)} stroke={THEME.grid} strokeWidth={0.6} />
          <text x={pl - 6} y={Yt(v) + 4} textAnchor="end" fontFamily={MONO} fontSize={10.5} fill={THEME.muted}>
            {v}
          </text>
        </g>
      ))}
      <text x={pl} y={12} fontFamily={MONO} fontSize={11} fill={THEME.muted}>
        tokens per word
      </text>
      <path d={path((r) => r.trainPerWord, Yt)} fill="none" stroke={THEME.accent} strokeWidth={2} strokeDasharray="6 4" />
      <path d={path((r) => r.heldPerWord, Yt)} fill="none" stroke={THEME.highlight} strokeWidth={2.2} />
      <line x1={pl} x2={W - pr} y1={Yt(1)} y2={Yt(1)} stroke={THEME.muted} strokeWidth={0.8} strokeDasharray="2 3" />
      <SvgLabel x={X(4)} y={Yt(1) + 15} color={THEME.muted} size={10.5}>
        1 token per word: every word is one token
      </SvgLabel>
      <SvgLabel x={X(85)} y={Yt(rows[85].heldPerWord) - 12} color={THEME.highlight} size={11.5}>
        ◆ held-out sentences (never seen)
      </SvgLabel>
      <SvgLabel x={X(60)} y={Yt(rows[60].trainPerWord) + 17} color={THEME.accent} size={11.5}>
        ● training corpus (dashed)
      </SvgLabel>
      {/* bottom: the embedding table */}
      <line x1={pl} x2={W - pr} y1={bot.y0 + bot.h} y2={bot.y0 + bot.h} stroke={THEME.grid} strokeWidth={0.8} />
      <text x={pl - 6} y={Yb(0) + 4} textAnchor="end" fontFamily={MONO} fontSize={10.5} fill={THEME.muted}>
        0
      </text>
      <text x={pl - 6} y={Yb(pMax) + 4} textAnchor="end" fontFamily={MONO} fontSize={10.5} fill={THEME.muted}>
        {fmt(pMax / 1000, 1) + "k"}
      </text>
      <text x={pl} y={bot.y0 - 8} fontFamily={MONO} fontSize={11} fill={THEME.muted}>
        {"embedding table V × d (d = " + SP_D + ")"}
      </text>
      <path d={path((r) => r.V * SP_D, Yb)} fill="none" stroke={THEME.third} strokeWidth={2} />
      {xt.map((v) => (
        <text key={"x" + v} x={X(v)} y={H - 10} textAnchor="middle" fontFamily={MONO} fontSize={10.5} fill={THEME.muted}>
          {v}
        </text>
      ))}
      <text x={W - pr} y={H - 10} textAnchor="end" fontFamily={MONO} fontSize={10.5} fill={THEME.muted}>
        merges M
      </text>
      <line x1={X(M)} x2={X(M)} y1={top.y0 - 4} y2={bot.y0 + bot.h} stroke={THEME.text} strokeWidth={1} strokeDasharray="3 3" />
      <Marker x={X(M)} y={Yt(cur.trainPerWord)} shape="circle" color={THEME.accent} r={5} />
      <Marker x={X(M)} y={Yt(cur.heldPerWord)} shape="diamond" color={THEME.highlight} r={5} />
      <Marker x={X(M)} y={Yb(cur.V * SP_D)} shape="square" color={THEME.third} r={4.5} />
    </svg>
  );
}

// ---------- a token sequence as chips (Name screen) ----------
function SpChips({ tokens, ids, label }) {
  return (
    <div style={{ margin: "4px 0 10px" }}>
      {label ? <Label>{label}</Label> : null}
      <div style={{ display: "flex", flexWrap: "wrap", gap: 6 }}>
        {tokens.map((t, i) => {
          const sp = spIsSpecial(t);
          return (
            <div key={i} style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 2 }}>
              <span
                style={{
                  fontFamily: MONO,
                  fontSize: 14,
                  padding: "4px 9px",
                  borderRadius: 4,
                  border: "1px " + (t === "[PAD]" ? "dashed " : "solid ") + (sp ? THEME.highlight : THEME.grid),
                  background: sp ? rgba(THEME.highlight, 0.13) : THEME.bg,
                  color: sp ? THEME.highlight : THEME.text,
                }}
              >
                {t}
              </span>
              {ids ? <span style={{ fontFamily: MONO, fontSize: 11, color: THEME.muted }}>{ids[i] === null ? "" : "ID " + ids[i]}</span> : null}
            </div>
          );
        })}
      </div>
    </div>
  );
}

const SP_FILLERS = [
  ["none", "leave empty"],
  ["[PAD]", "[PAD]"],
  ["[EOS]", "[EOS]"],
  ["Hi", "Hi"],
  ["books", "books"],
];
const SP_VIEWS = [
  ["s33", "slide 33: IDs"],
  ["bert", "slides 34, 36: BERT"],
  ["chat", "slide 35: a chat"],
  ["batch", "slide 36: a batch"],
];
const SP_BIG = [
  ["16000", "16k"],
  ["32000", "32k"],
  ["64000", "64k"],
];

export default function SpecialLesson({ initialScreen }) {
  const [screen, setScreen] = useState(initialScreen || 0);
  const [filler, setFiller] = useState("none");
  const [mask1, setMask1] = useState([1, 1, 1, 1]);
  const [mask2, setMask2] = useState([1, 1, 1, 1]);
  const [mExp, setMExp] = useState(20);
  const [big, setBig] = useState("32000");
  const [view, setView] = useState("s33");
  const [mCh, setMCh] = useState(10);

  const flipMask = useCallback((r, j) => {
    const f = (m) => m.map((v, k) => (k === j ? 1 - v : v));
    if (r === 0) setMask1(f);
    else setMask2(f);
  }, []);
  const resetPuzzle = useCallback(() => {
    setFiller("none");
    setMask1([1, 1, 1, 1]);
    setMask2([1, 1, 1, 1]);
  }, []);
  const data = useMemo(spData, []);

  let question, stage, side;
  if (screen === 0) {
    const f = filler === "none" ? null : filler;
    const P = spPuzzle(f, mask1, mask2);
    const row2 = SP_ROW2.concat([f, f]);
    const hiA = P.r2.alone[0].o;
    const hiB = P.r2.inBatch[0].o;
    question = (
      <>
        <b>Where we are:</b> lesson 4 encoded the unseen word lowest as [low, est&lt;/w&gt;] → [15, 13] with the 16-token toy vocabulary. A model needs more than word pieces: slide 33 adds [BOS] = 16 and [EOS] = 17 to mark where a sequence starts and ends, so V = 18. Training also runs on batches, and sentences have different lengths (slide 36). <b>The puzzle:</b> row 2 has 2 tokens, but a batch is one 2 × 4 tensor. Fill its empty cells, then click the mask cells, so that every real token's output is exactly what it would be if its sentence ran alone. Does it matter which token you fill with?
      </>
    );
    stage = (
      <>
        <SpBatch rows={[SP_ROW1, row2]} mask={[mask1, mask2]} onMask={flipMask} weights={[P.r1.inBatch[0].w, P.r2.inBatch[0].w]} ariaLabel="slide 36's batch; click a mask cell to switch it between 1 and 0" />
        <Note>Each position's output is a softmax-weighted average of the positions in its row whose mask is 1 (one attention layer, lecture 3), using made-up 2-D embeddings. The teal bar in each cell is the weight that the row's first token (I or Hi) gives it. Word units are schematic, as on slide 36. The mask is its own array beside the tokens, not a token.</Note>
      </>
    );
    side = (
      <>
        <Panel>
          <Label>fill row 2's empty cells with</Label>
          <Pills options={SP_FILLERS} value={filler} onPick={setFiller} />
          <Readout
            rows={[
              ["batch shape", P.rect ? "2 × 4 tensor" : "lengths 4 and 2: no tensor", P.rect ? THEME.text : THEME.warn],
              ["row 1 outputs moved by", fmt(P.r1.worst, 3), P.r1.worst < 1e-9 ? THEME.ok : THEME.warn],
              ["row 2 outputs moved by", P.rect ? fmt(P.r2.worst, 3) : "—", P.rect && P.r2.worst < 1e-9 ? THEME.ok : THEME.warn],
              ["Hi alone → in the batch", "[" + fmt(hiA[0], 2) + ", " + fmt(hiA[1], 2) + "] → " + (isFinite(hiB[0]) ? "[" + fmt(hiB[0], 2) + ", " + fmt(hiB[1], 2) + "]" : "NaN")],
              ["positions counted in the loss", String(P.lossPos) + " (6 real tokens)"],
            ]}
          />
          <Goal met={P.rect}>the batch is a 2 × 4 rectangle</Goal>
          <Goal met={P.same}>every real token's output equals its output alone</Goal>
          <Goal met={P.lossOk}>the loss counts the 6 real tokens and nothing else</Goal>
          <Btn onClick={resetPuzzle}>start over</Btn>
        </Panel>
        {P.same && P.lossOk ? (
          <Note>Now switch the filler. The outputs stay put for every choice: once the mask is [1, 1, 0, 0], the filler is never read. [PAD] is the conventional filler, and it has a learned embedding like any other token (slide 33), so it would not be harmless on its own.</Note>
        ) : (
          <Note>"Moved by" is the largest distance between a real token's output in the batch and its output when the sentence runs by itself.</Note>
        )}
      </>
    );
  } else if (screen === 1) {
    const rows = data.rows;
    const M = Math.min(mExp, rows.length - 1);
    const r = rows[M];
    const rule = M > 0 ? data.T.rules[M - 1] : null;
    const libIdx = data.T.rules.findIndex((x) => x[0] + x[1] === "library</w>");
    const bigV = parseInt(big, 10);
    question = (
      <>
        <b>Claim to test:</b> every token, merged or special, is one row of the embedding table, so the table holds V × d numbers. Each merge adds one row and makes sequences shorter (slide 37). Here the deck's BPE (slides 24–29) is trained on a 30-sentence toy corpus and applied to 4 held-out sentences it never saw. Move the number of merges M. Where do more merges stop paying off on the held-out text, and why does the training text keep shrinking after that?
      </>
    );
    stage = (
      <>
        <SpCurve rows={rows} M={M} />
        <SpChips label={"held-out sentence 3 at M = " + M + " (" + (r.lens[2] - 2) + " tokens)"} tokens={r.enc[2].slice(1, -1)} />
      </>
    );
    side = (
      <>
        <Panel>
          <Slider id="sp-mexp" label="merges M" value={M} min={0} max={rows.length - 1} stepSize={1} onChange={setMExp} display={"M = " + M} />
          <Readout
            rows={[
              ["V = base + merges + special", data.T.base.length + " + " + M + " + " + SP_NSPECIAL + " = " + r.V],
              ["table V × d at d = " + SP_D, r.V * SP_D + " numbers", THEME.third],
              ["training: tokens per word", fmt(r.trainPerWord, 2), THEME.accent],
              ["held-out: tokens per word", fmt(r.heldPerWord, 2), THEME.highlight],
              ["merge " + M, rule ? rule[0] + " + " + rule[1] + " (seen " + rule[2] + "×)" : "none yet"],
            ]}
          />
        </Panel>
        <Panel>
          <Label>slide 37, real scale, d = 768</Label>
          <Pills options={SP_BIG} value={big} onPick={setBig} />
          <Readout rows={[["V × d", bigV.toLocaleString("en-US") + " × 768 = " + (bigV * 768).toLocaleString("en-US") + " ≈ " + fmt((bigV * 768) / 1e6, 1) + "M"]]} />
        </Panel>
        <Note>
          <b>Result:</b> the table grows by d numbers per merge, a straight line. Held-out length drops fast over the first 50 or so merges and then flattens, while the training text keeps falling toward 1 token per word. Late merges are seen only 1–3 times: they glue whole training words together (merge {libIdx + 1} makes library&lt;/w&gt;), which rarely helps new text. Length saved must be measured on held-out text (slide 37).
        </Note>
      </>
    );
  } else if (screen === 2) {
    let body;
    if (view === "s33") {
      body = (
        <>
          <SpChips label="slide 33: lowest, with boundary tokens" tokens={["[BOS]", "low", "est</w>", "[EOS]"]} ids={[16, 15, 13, 17]} />
          <Note>IDs 0–10 are the base symbols, 11–15 the five merges (slide 29). The two special tokens take the next IDs, 16 and 17. No merge produced them: they are configured before training, and their embeddings are learned with the model.</Note>
        </>
      );
    } else if (view === "bert") {
      body = (
        <>
          <SpChips label="BERT conventions, word units" tokens={["[CLS]", "I", "read", "[MASK]", "[SEP]"]} />
          <Note>[MASK] is an input token: it hides a word that BERT must predict (slide 36's "I read [MASK]"). [CLS] gives a summary position, [SEP] separates segments (slide 34). A decoder-only language model need not have any of these.</Note>
        </>
      );
    } else if (view === "chat") {
      body = (
        <>
          <SpChips label="gpt-oss / Harmony, message 1" tokens={["<|start|>", "user", "<|message|>", "Hi", "<|end|>"]} />
          <SpChips label="message 2" tokens={["<|start|>", "assistant", "<|message|>", "Hello!", "<|fim_suffix|>"]} />
          <Note>The orange markers have special IDs. "user" and "assistant" are ordinary header text. &lt;|end|&gt; ends a message and &lt;|fim_suffix|&gt; ends the assistant's turn, as printed on slide 35. Chat formats and stop markers vary by model.</Note>
        </>
      );
    } else {
      body = (
        <>
          <SpBatch rows={[["I", "read", "books", "[EOS]"], ["Hi", "[EOS]", "[PAD]", "[PAD]"]]} mask={[[1, 1, 1, 1], [1, 1, 0, 0]]} ariaLabel="slide 36's batch and padding mask" />
          <Note>The tokens and the mask are two different arrays. [PAD] fills the tensor. The mask [1, 1, 0, 0] is what keeps attention and the loss off the filler.</Note>
        </>
      );
    }
    question = (
      <>
        <b>What you found:</b> some tokens are not word pieces. [BOS], [EOS] and [PAD] are configured rows of the vocabulary with learned embeddings, so they count in V. Padding fills a batch to a rectangle, but only the padding mask, a separate 0/1 array, keeps attention and the loss away from the filler. And V sets the table size V × d, while more merges shorten sequences less and less on new text (slides 33–37). Pick a view to see where special tokens appear.
      </>
    );
    stage = (
      <>
        <Pills options={SP_VIEWS} value={view} onPick={setView} />
        {body}
        <Eq size={12.5}>
          V = 11 base + 5 merges + 2 special = 18 (slides 29, 33)
          <br />
          embedding parameters = V × d: 16,000 × 768 = 12,288,000 ≈ 12.3M (slide 37)
          <br />
          padding mask mⱼ = 1 if position j holds a real token, else 0
          <br />
          slide 36, row 2 "Hi [EOS] [PAD] [PAD]" → m = [1, 1, 0, 0]
          <br />
          attention score sᵢⱼ gets + 0 where mⱼ = 1 and + (−∞) where mⱼ = 0
          <br />
          loss = Σⱼ mⱼ ℓⱼ / Σⱼ mⱼ (padded targets drop out)
        </Eq>
      </>
    );
    side = (
      <>
        <Panel>
          <Label>common special tokens (slide 34)</Label>
          <Readout
            rows={[
              ["[BOS] / [EOS]", "start / end of a sequence"],
              ["[PAD]", "fills unused batch positions"],
              ["[UNK]", "an input the tokenizer cannot encode"],
              ["[MASK]", "hidden content to predict (BERT)"],
              ["[CLS] / [SEP]", "summary position / separator (BERT)"],
            ]}
          />
          <Note>The inventory and spellings depend on the model.</Note>
        </Panel>
        <Note>In a decoder with a causal mask, right padding sits after every real token, so attention never reaches it, but the padded targets must still be left out of the loss. Next, lesson 6: our base symbols were characters, so an unseen character has no token. Byte-level BPE starts from the 256 UTF-8 byte values instead (slides 38–41).</Note>
      </>
    );
  } else {
    const rows = data.rows;
    const M = Math.min(mCh, rows.length - 1);
    const r = rows[M];
    const C = spChallenge(r);
    const real = r.lens.reduce((a, b) => a + b, 0);
    const total = r.maxLen * r.enc.length;
    question = (
      <>
        You will train a small model with d = {SP_D} on batches like these 4 held-out sentences, each wrapped in [BOS] … [EOS] and right-padded to the longest row. Two budgets: the embedding table may hold at most {SP_PARAM_BUDGET.toLocaleString("en-US")} numbers, and a padded row may be at most {SP_LEN_BUDGET} tokens long. Choose where to stop merging.
      </>
    );
    stage = (
      <>
        <SpColumns enc={r.enc} budget={SP_LEN_BUDGET} />
        <Note>Each column is one sentence, its positions running downward, padded with [PAD] (dashed) to the longest column. Next to it is that row's padding mask. Orange cells are special tokens.</Note>
      </>
    );
    side = (
      <>
        <Panel edge={C.ok ? THEME.ok : THEME.grid}>
          <Slider id="sp-mch" label="merges M" value={M} min={0} max={rows.length - 1} stepSize={1} onChange={setMCh} display={"M = " + M} />
          <Readout
            rows={[
              ["V", data.T.base.length + " base + " + M + " merges + " + SP_NSPECIAL + " special = " + r.V],
              ["embedding table V × " + SP_D, r.V * SP_D + " / " + SP_PARAM_BUDGET, C.okParams ? THEME.ok : THEME.warn],
              ["longest padded row", r.maxLen + " / " + SP_LEN_BUDGET, C.okLen ? THEME.ok : THEME.warn],
              ["batch tensor", r.enc.length + " × " + r.maxLen + " = " + total + " positions"],
              ["real tokens · [PAD]", real + " · " + (total - real)],
            ]}
          />
          <Goal met={C.okParams}>table ≤ {SP_PARAM_BUDGET.toLocaleString("en-US")} numbers</Goal>
          <Goal met={C.okLen}>longest row ≤ {SP_LEN_BUDGET} tokens, [BOS] and [EOS] included</Goal>
          <Hint
            hints={[
              "The two budgets pull in opposite directions. Each merge adds one row of " + SP_D + " numbers to the table, and makes rows shorter. Find the first M where the longest row fits, and the last M where the table fits.",
              "The table budget is " + SP_PARAM_BUDGET + " = 100 × " + SP_D + ", so V can be at most 100. V counts the " + data.T.base.length + " base symbols and the " + SP_NSPECIAL + " special tokens as well as the merges. The longest row is held-out sentence 3.",
            ]}
          />
        </Panel>
        <Note>The short row is mostly [PAD]: every padded position costs compute in the batch, even though the mask keeps it out of attention and the loss.</Note>
      </>
    );
  }

  return <LessonFrame screen={screen} onScreen={setScreen} title="Special tokens and vocabulary size" question={question} stage={stage} side={side} />;
}
