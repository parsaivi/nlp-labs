import { useState, useCallback, useMemo } from "react";
import { THEME, MONO } from "./common";

// Lesson 4 - BPE inference: apply rules by rank (Tokenization, slides 30-32).
// Aha: at inference BPE counts nothing. The saved list is frozen, and its order is the whole algorithm:
// among the adjacent pairs that have a rule, merge the one with the earliest rank, and repeat. The rank is
// training-corpus frequency frozen into an order; how often a pair occurs in the NEW text plays no part.
// Rank only changes the result when two eligible pairs compete for the same symbol.
//
// Numbers from the slides: the training corpus low, lower, newest, widest (slide 25); the five ordered rules
// 1 e+s, 2 es+t, 3 est+</w>, 4 l+o, 5 lo+w and the IDs </w>, d, e, i, l, n, o, r, s, t, w = 0-10, es = 11,
// est = 12, est</w> = 13, lo = 14, low = 15 (slide 29); the inference procedure (slide 30); the worked example
// lowest -> l o w e s t </w> -> ranks 1, 2, 3, 4, 5 -> [low, est</w>] -> [15, 13], with (l, o) eligible from the
// start but waiting for earlier ranks (slides 31-32). All reproduced by apTrace below.
// Teaching toys (not on the slides): the sixth rule w + e -> we on the Explore screen (the pair (w, e) does
// appear on slide 26 with count 8, but it never became a rule), the extra rule o + w -> ow, the challenge text
// "lower newest lowest tow" with its target segmentations, and the "sorted by count in this text" start order.
// Toy tokens such as we and ow have no ID on the slides, so they are shown with "?".

// @sim-start
const AP_BASE = ["</w>", "d", "e", "i", "l", "n", "o", "r", "s", "t", "w"];
const AP_EOW = "</w>";
const AP_RULES = [
  ["e", "s"],
  ["es", "t"],
  ["est", "</w>"],
  ["l", "o"],
  ["lo", "w"],
];
const AP_ID = { "</w>": 0, d: 1, e: 2, i: 3, l: 4, n: 5, o: 6, r: 7, s: 8, t: 9, w: 10, es: 11, est: 12, "est</w>": 13, lo: 14, low: 15 };
const AP_TOY = ["w", "e"];
const AP_CH_RULES = [
  ["e", "s"],
  ["es", "t"],
  ["est", "</w>"],
  ["l", "o"],
  ["lo", "w"],
  ["w", "e"],
  ["o", "w"],
];
const AP_CH_WORDS = ["lower", "newest", "lowest", "tow"];
const AP_CH_TARGET = [
  ["lo", "we", "r", "</w>"],
  ["n", "e", "w", "est</w>"],
  ["low", "est</w>"],
  ["t", "ow", "</w>"],
];
// one intended answer: slide 29's order with w + e slotted after est</w> and o + w last
const AP_CH_GOOD = [0, 1, 2, 5, 3, 4, 6];

function apSplit(word) {
  return word.split("").concat([AP_EOW]);
}
function apKey(a, b) {
  return a + "\u0001" + b;
}
function apRankMap(rules) {
  const m = {};
  rules.forEach((r, i) => {
    m[apKey(r[0], r[1])] = i + 1;
  });
  return m;
}
// eligible pairs: positions whose adjacent pair has a saved rule, with that rule's rank (1 = earliest)
function apEligible(toks, rmap) {
  const out = [];
  for (let i = 0; i < toks.length - 1; i++) {
    const r = rmap[apKey(toks[i], toks[i + 1])];
    if (r) out.push({ i: i, rank: r });
  }
  return out;
}
function apEarliest(el) {
  let best = null;
  for (const e of el) if (!best || e.rank < best.rank) best = e;
  return best;
}
// replace every adjacent (a, b), left to right, by the token ab
function apMerge(toks, a, b) {
  const o = [];
  for (let i = 0; i < toks.length; i++) {
    if (i < toks.length - 1 && toks[i] === a && toks[i + 1] === b) {
      o.push(a + b);
      i++;
    } else o.push(toks[i]);
  }
  return o;
}
// deterministic BPE inference (as in subword-nmt apply_bpe.py): merge the eligible pair with the earliest rank, repeat
function apTrace(word, rules) {
  const rmap = apRankMap(rules);
  let toks = apSplit(word);
  const steps = [{ toks: toks, rank: 0, pair: null, eligible: apEligible(toks, rmap) }];
  for (let guard = 0; guard < 64; guard++) {
    const best = apEarliest(apEligible(toks, rmap));
    if (!best) break;
    const pair = rules[best.rank - 1];
    toks = apMerge(toks, pair[0], pair[1]);
    steps.push({ toks: toks, rank: best.rank, pair: pair, eligible: apEligible(toks, rmap) });
  }
  return steps;
}
function apFinal(word, rules) {
  const t = apTrace(word, rules);
  return t[t.length - 1].toks;
}
function apSame(a, b) {
  return a.length === b.length && a.every((v, i) => v === b[i]);
}
// how often a rule's pair occurs in the plain characters of the challenge text
function apCount(rule) {
  let c = 0;
  for (const w of AP_CH_WORDS) {
    const t = apSplit(w);
    for (let i = 0; i < t.length - 1; i++) if (t[i] === rule[0] && t[i + 1] === rule[1]) c++;
  }
  return c;
}
// the challenge's starting order: most frequent pair in this text first (ties keep list order)
function apStartOrder() {
  const idx = AP_CH_RULES.map((r, i) => i);
  return idx.sort((a, b) => apCount(AP_CH_RULES[b]) - apCount(AP_CH_RULES[a]) || a - b);
}
function apChallenge(order) {
  const rules = order.map((k) => AP_CH_RULES[k]);
  const traces = AP_CH_WORDS.map((w) => apTrace(w, rules));
  const outs = traces.map((t) => t[t.length - 1].toks);
  const ok = outs.map((o, i) => apSame(o, AP_CH_TARGET[i]));
  return { traces: traces, outs: outs, ok: ok, pass: ok.every(Boolean) };
}
function apPerms(n) {
  if (n <= 1) return [[0]];
  const out = [];
  for (const p of apPerms(n - 1)) for (let k = 0; k <= p.length; k++) out.push(p.slice(0, k).concat([n - 1]).concat(p.slice(k)));
  return out;
}
// Tuning (node, tools/sim.js): lowest under slide 29's rules -> [low, est</w>] -> [15, 13]; the five slide rules give
// the same tokens in all 120 orders for every word tried (no two rules share a symbol); start order
// w+e, o+w, e+s, l+o, es+t, est+</w>, lo+w fails newest and lowest; AP_CH_GOOD passes; 504 of 5040 orders pass
// (10%); the nearest passing order is 3 adjacent swaps from the start.
// @sim-end

const AP_PERM5 = apPerms(5);
const AP_PERM6 = apPerms(6);
const AP_LETTERS = AP_BASE.slice(1);

function apRuleText(r) {
  return r[0] + " + " + r[1] + " → " + r[0] + r[1];
}
function apIdOf(t) {
  if (AP_ID[t] !== undefined) return String(AP_ID[t]);
  if (t.length === 1 && AP_LETTERS.indexOf(t) < 0) return "✕";
  return "?";
}
function apClean(s) {
  return s.toLowerCase().replace(/\s+/g, "").slice(0, 14);
}
function apBadChars(word) {
  const out = [];
  for (const ch of word.split("")) if (AP_LETTERS.indexOf(ch) < 0 && out.indexOf(ch) < 0) out.push(ch);
  return out;
}
function apDecode(ids) {
  const inv = {};
  for (const k of Object.keys(AP_ID)) inv[AP_ID[k]] = k;
  return ids.map((i) => inv[i]).join("").split(AP_EOW).join(" ");
}
function apRulesWithToy(toyRank) {
  if (!toyRank) return AP_RULES;
  const R = AP_RULES.slice();
  R.splice(toyRank - 1, 0, AP_TOY);
  return R;
}

// ---------- pieces ----------
function ApplyTile({ t }) {
  const eow = t === AP_EOW;
  const bad = t.length === 1 && !eow && AP_LETTERS.indexOf(t) < 0;
  const merged = t.length > 1 && !eow;
  const toy = merged && AP_ID[t] === undefined;
  let edge = THEME.grid;
  let fill = THEME.bg;
  let col = THEME.text;
  if (eow) col = THEME.muted;
  if (merged) {
    edge = toy ? THEME.third : THEME.accent;
    fill = rgba(toy ? THEME.third : THEME.accent, 0.14);
  }
  if (bad) {
    edge = THEME.warn;
    fill = rgba(THEME.warn, 0.12);
  }
  return (
    <span
      style={{
        display: "inline-block",
        fontFamily: bad ? FA : MONO,
        fontSize: eow ? 13 : 16,
        lineHeight: "24px",
        padding: "3px 8px",
        minWidth: 16,
        textAlign: "center",
        borderRadius: 4,
        border: "1px " + (eow ? "dashed " : "solid ") + edge,
        background: fill,
        color: col,
        whiteSpace: "nowrap",
      }}
    >
      {bad ? "✕ " : ""}
      {t}
    </span>
  );
}
function ApplyPairBtn({ i, rank, star, onPair, showRank }) {
  const click = useCallback(() => onPair(i), [onPair, i]);
  const label = showRank ? "r" + rank + (star ? "★" : "") : "＋";
  const col = star ? THEME.highlight : THEME.accent;
  return (
    <button
      onClick={onPair ? click : undefined}
      disabled={!onPair}
      title={showRank ? "rule rank " + rank : "merge this pair"}
      style={{
        fontFamily: MONO,
        fontSize: 11.5,
        padding: "2px 4px",
        margin: "0 2px",
        borderRadius: 10,
        cursor: onPair ? "pointer" : "default",
        border: "1px " + (star ? "solid " : "dashed ") + col,
        background: star ? rgba(THEME.highlight, 0.16) : "transparent",
        color: star ? THEME.highlight : THEME.accent,
        lineHeight: "16px",
      }}
    >
      {label}
    </button>
  );
}
// one token sequence; between two tokens a marker when the pair has a saved rule
function ApplyRow({ toks, rmap, onPair, showRank, starOn }) {
  const el = rmap ? apEligible(toks, rmap) : [];
  const best = starOn ? apEarliest(el) : null;
  const byPos = {};
  for (const e of el) byPos[e.i] = e.rank;
  const items = [];
  toks.forEach((t, i) => {
    items.push(<ApplyTile key={"t" + i} t={t} />);
    if (i < toks.length - 1) {
      if (byPos[i]) items.push(<ApplyPairBtn key={"p" + i} i={i} rank={byPos[i]} star={!!best && best.i === i} onPair={onPair} showRank={showRank} />);
      else items.push(<span key={"p" + i} style={{ display: "inline-block", width: 6 }} />);
    }
  });
  return <div style={{ display: "flex", flexWrap: "wrap", alignItems: "center", rowGap: 6, padding: "4px 0" }}>{items}</div>;
}
// the saved, ordered rule list; `hot` = rank just applied, `elig` = ranks eligible now
function ApplyRules({ rules, hot, elig, toyKey }) {
  return (
    <div style={{ fontFamily: MONO, fontSize: 13, lineHeight: 1.75 }}>
      {rules.map((r, i) => {
        const rank = i + 1;
        const isHot = hot === rank;
        const isEl = elig && elig.indexOf(rank) >= 0;
        const toy = toyKey && r[0] + r[1] === toyKey;
        const tok = r[0] + r[1];
        return (
          <div key={tok} style={{ display: "flex", justifyContent: "space-between", gap: 8, padding: "0 6px", borderRadius: 3, background: isHot ? rgba(THEME.highlight, 0.16) : "transparent", color: isEl || isHot ? THEME.text : THEME.muted }}>
            <span>
              <span style={{ color: isHot ? THEME.highlight : THEME.accent }}>{rank}</span> {apRuleText(r)}
              {toy ? <span style={{ color: THEME.third }}> (toy)</span> : null}
            </span>
            <span>
              {isHot ? "applied ★ " : isEl ? "eligible ▪ " : ""}ID {apIdOf(tok)}
            </span>
          </div>
        );
      })}
    </div>
  );
}
function ApplyTrace({ steps, upto }) {
  const last = upto === undefined ? steps.length - 1 : upto;
  return (
    <div style={{ fontFamily: MONO, fontSize: 13.5, lineHeight: 1.7, background: THEME.bg, border: "1px solid " + THEME.grid, borderRadius: 6, padding: "6px 10px", overflowX: "auto" }}>
      {steps.slice(0, last + 1).map((s, k) => (
        <div key={k} style={{ display: "flex", gap: 12, whiteSpace: "nowrap", color: k === last ? THEME.text : THEME.muted }}>
          <span style={{ minWidth: 150, color: k === last ? THEME.highlight : THEME.muted }}>{s.pair ? "rank " + s.rank + ": " + s.pair[0] + " + " + s.pair[1] : "start"}</span>
          <span>{s.toks.join(" ")}</span>
        </div>
      ))}
      {last === steps.length - 1 ? <div style={{ color: THEME.muted }}>stop: no adjacent pair has a rule</div> : null}
    </div>
  );
}
function ApplyWordBox({ value, onChange, id }) {
  const change = useCallback((e) => onChange(e.target.value), [onChange]);
  return (
    <label htmlFor={id} style={{ display: "block", margin: "2px 0 6px" }}>
      <Label>a word (letters d e i l n o r s t w)</Label>
      <input
        id={id}
        type="text"
        value={value}
        onChange={change}
        maxLength={20}
        spellCheck={false}
        style={{ width: "100%", boxSizing: "border-box", fontFamily: MONO, fontSize: 16, padding: "6px 8px", background: THEME.bg, color: THEME.text, border: "1px solid " + THEME.accent, borderRadius: 4 }}
      />
    </label>
  );
}
function ApplyIdChip({ id, tok, onPick }) {
  const click = useCallback(() => onPick(id), [onPick, id]);
  return (
    <button
      onClick={click}
      style={{ fontFamily: MONO, fontSize: 12, padding: "3px 6px", borderRadius: 4, cursor: "pointer", border: "1px solid " + (tok.length > 1 && tok !== AP_EOW ? THEME.accent : THEME.grid), background: THEME.bg, color: THEME.text }}
    >
      <span style={{ color: THEME.muted }}>{id}</span> {tok}
    </button>
  );
}
function ApplyOrderRow({ pos, n, rule, count, onMove }) {
  const up = useCallback(() => onMove(pos, -1), [onMove, pos]);
  const down = useCallback(() => onMove(pos, 1), [onMove, pos]);
  const toy = AP_RULES.every((r) => r[0] + r[1] !== rule[0] + rule[1]);
  const small = { fontFamily: MONO, fontSize: 12, width: 28, padding: "2px 0", borderRadius: 4, border: "1px solid " + THEME.accent, background: "transparent", color: THEME.accent, cursor: "pointer" };
  const off = { fontFamily: MONO, fontSize: 12, width: 28, padding: "2px 0", borderRadius: 4, border: "1px solid " + THEME.grid, background: "transparent", color: THEME.muted, cursor: "default" };
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 6, fontFamily: MONO, fontSize: 13, padding: "3px 0", borderBottom: "1px solid " + rgba(THEME.grid, 0.6) }}>
      <span style={{ color: THEME.accent, width: 18, textAlign: "right" }}>{pos + 1}</span>
      <span style={{ flex: "1 1 auto", color: THEME.text }}>
        {apRuleText(rule)}
        {toy ? <span style={{ color: THEME.third }}> (toy)</span> : null}
      </span>
      <span style={{ color: THEME.muted, width: 34, textAlign: "right", whiteSpace: "nowrap" }} title="occurrences of this pair in the text">{count}×</span>
      <button onClick={up} disabled={pos === 0} style={pos === 0 ? off : small} aria-label={"move " + rule.join("+") + " up"}>
        ▲
      </button>
      <button onClick={down} disabled={pos === n - 1} style={pos === n - 1 ? off : small} aria-label={"move " + rule.join("+") + " down"}>
        ▼
      </button>
    </div>
  );
}

export default function ApplyLesson({ initialScreen }) {
  const [screen, setScreen] = useState(initialScreen || 0);
  const [pz, setPz] = useState({ toks: apSplit("lowest"), log: [] });
  const [rawWord, setRawWord] = useState("lowest");
  const [toyRank, setToyRank] = useState("0");
  const [k, setK] = useState(0);
  const [ids, setIds] = useState([15, 13]);
  const [order, setOrder] = useState(apStartOrder());

  const word = apClean(rawWord);
  const rules = useMemo(() => apRulesWithToy(parseInt(toyRank, 10)), [toyRank]);
  const rmap = useMemo(() => apRankMap(rules), [rules]);
  const steps = useMemo(() => apTrace(word, rules), [word, rules]);
  const kk = Math.min(k, steps.length - 1);

  const pzMap = useMemo(() => apRankMap(AP_RULES), []);
  const onPzPair = useCallback(
    (i) =>
      setPz((P) => {
        const a = P.toks[i];
        const b = P.toks[i + 1];
        const rank = pzMap[apKey(a, b)];
        if (!rank) return P;
        const elig = apEligible(P.toks, pzMap).map((e) => e.rank);
        return { toks: apMerge(P.toks, a, b), log: P.log.concat([{ a: a, b: b, rank: rank, elig: elig }]) };
      }),
    [pzMap]
  );
  const resetPz = useCallback(() => setPz({ toks: apSplit("lowest"), log: [] }), []);
  const onWord = useCallback((v) => {
    setRawWord(v);
    setK(0);
  }, []);
  const onToy = useCallback((v) => {
    setToyRank(v);
    setK(0);
  }, []);
  const stepFwd = useCallback(() => setK((v) => v + 1), []);
  const stepBack = useCallback(() => setK((v) => Math.max(0, v - 1)), []);
  const runAll = useCallback(() => setK(99), []);
  const restart = useCallback(() => setK(0), []);
  const addId = useCallback((id) => setIds((L) => (L.length >= 12 ? L : L.concat([id]))), []);
  const clearIds = useCallback(() => setIds([]), []);
  const undoId = useCallback(() => setIds((L) => L.slice(0, -1)), []);
  const onMove = useCallback(
    (pos, dir) =>
      setOrder((O) => {
        const j = pos + dir;
        if (j < 0 || j >= O.length) return O;
        const n = O.slice();
        const t = n[pos];
        n[pos] = n[j];
        n[j] = t;
        return n;
      }),
    []
  );
  const resetOrder = useCallback(() => setOrder(apStartOrder()), []);

  // how many different final segmentations the current rule set gives over every possible ordering
  const variety = useMemo(() => {
    const perms = rules.length === 6 ? AP_PERM6 : AP_PERM5;
    const seen = {};
    for (const p of perms) seen[apFinal(word, p.map((q) => rules[q])).join(" ")] = 1;
    return { n: Object.keys(seen).length, of: perms.length, list: Object.keys(seen) };
  }, [word, rules]);

  let question, stage, side;
  if (screen === 0) {
    const el = apEligible(pz.toks, pzMap);
    const done = el.length === 0;
    const inOrder = pz.log.every((e) => e.rank === Math.min.apply(null, e.elig));
    question = (
      <>
        <b>Where we are:</b> lesson 3 trained BPE on low, lower, newest and widest, and saved five ordered merge rules and 16 token IDs (slide 29). Now a word arrives that training never saw: <b>lowest</b>. Training is over, so nothing is counted and no rule is added (slide 30). <b>The puzzle:</b> turn lowest into token IDs using only the saved list. A ＋ sits between two symbols that have a saved rule; click one to merge that pair. Which pair should go first, and does it matter?
      </>
    );
    stage = (
      <>
        <Label>the word, split into base symbols plus the end-of-word marker</Label>
        <Panel>
          <ApplyRow toks={pz.toks} rmap={pzMap} onPair={onPzPair} showRank={false} />
        </Panel>
        <Label>your merges</Label>
        <div style={{ fontFamily: MONO, fontSize: 13, lineHeight: 1.7, minHeight: 24 }}>
          {pz.log.length === 0 ? <span style={{ color: THEME.muted }}>none yet</span> : null}
          {pz.log.map((e, i) => {
            const first = Math.min.apply(null, e.elig);
            const ok = e.rank === first;
            return (
              <div key={i} style={{ color: ok ? THEME.text : THEME.warn }}>
                {ok ? "✓ " : "✗ "}
                {e.a} + {e.b} is rule {e.rank}. Eligible then: rules {e.elig.slice().sort().join(", ")}.{" "}
                {ok ? "The tokenizer also picks rule " + first + "." : "The tokenizer would pick rule " + first + " first."}
              </div>
            );
          })}
        </div>
        {done ? (
          <Panel edge={THEME.highlight} style={{ marginTop: 8 }}>
            <Readout
              rows={[
                ["your tokens", "[" + pz.toks.join(", ") + "]"],
                ["IDs (slide 29)", "[" + pz.toks.map(apIdOf).join(", ") + "]"],
                ["the tokenizer's tokens (slide 32)", "[low, est</w>] → [15, 13]"],
              ]}
            />
          </Panel>
        ) : null}
      </>
    );
    side = (
      <>
        <Panel>
          <Label>saved rules, in training order (slide 29)</Label>
          <ApplyRules rules={AP_RULES} elig={el.map((e) => e.rank)} />
        </Panel>
        <Panel>
          <Goal met={done}>no adjacent pair has a saved rule any more</Goal>
          <Goal met={done && inOrder}>every merge took the eligible rule with the earliest rank</Goal>
          <Btn onClick={resetPz}>start lowest again</Btn>
        </Panel>
        <Note>Try it twice: once starting with l + o, once in rank order. Compare the final tokens. If the order changes nothing here, why does slide 30 insist on the earliest rank?</Note>
      </>
    );
  } else if (screen === 1) {
    const cur = steps[kk];
    const bad = apBadChars(word);
    const fin = steps[steps.length - 1].toks;
    question = (
      <>
        <b>Claim to test:</b> inference repeats one move. Among the adjacent pairs that have a saved rule, merge the one with the earliest rank, and stop when no pair has a rule (slide 30). Type a word and step through it; ★ marks the pair that goes next. First, can you find a word whose final tokens change if the five rules are put in a different order? Then add a sixth rule, w + e → we, and move its rank. Which of lowest, lower and newest change, and at which ranks?
      </>
    );
    stage = (
      <>
        <Panel>
          <Label>
            step {kk} of {steps.length - 1}
            {cur.pair ? " · applied rank " + cur.rank + ": " + cur.pair[0] + " + " + cur.pair[1] : ""}
          </Label>
          <ApplyRow toks={cur.toks} rmap={rmap} showRank starOn />
          <div style={{ display: "flex", gap: 6, flexWrap: "wrap", marginTop: 6 }}>
            <Btn onClick={restart} disabled={kk === 0}>
              ⟲ start
            </Btn>
            <Btn onClick={stepBack} disabled={kk === 0}>
              ◀ back
            </Btn>
            <Btn onClick={stepFwd} disabled={kk >= steps.length - 1} strong>
              merge ★ ▶
            </Btn>
            <Btn onClick={runAll} disabled={kk >= steps.length - 1}>
              run to the end
            </Btn>
          </div>
        </Panel>
        <ApplyTrace steps={steps} upto={kk} />
        {bad.length ? (
          <div style={{ fontFamily: MONO, fontSize: 12.5, color: THEME.warn, marginTop: 6 }}>
            ✕ {bad.map((c) => "“" + c + "”").join(" ")} {bad.length > 1 ? "are" : "is"} not among the 11 base symbols, so there is no ID for it. Lesson 6 fixes this with bytes.
          </div>
        ) : null}
      </>
    );
    side = (
      <>
        <Panel>
          <ApplyWordBox id="ap-word" value={rawWord} onChange={onWord} />
          <Label>sixth rule w + e → we (toy)</Label>
          <Pills
            options={[
              ["0", "off"],
              ["1", "rank 1"],
              ["2", "2"],
              ["3", "3"],
              ["4", "4"],
              ["5", "5"],
              ["6", "6"],
            ]}
            value={toyRank}
            onPick={onToy}
          />
          <ApplyRules rules={rules} hot={cur.rank || null} elig={cur.eligible.map((e) => e.rank)} toyKey={toyRank !== "0" ? "we" : null} />
        </Panel>
        <Panel>
          <Readout
            rows={[
              ["final tokens", "[" + fin.join(", ") + "]"],
              ["IDs", "[" + fin.map(apIdOf).join(", ") + "]"],
              ["different results over all " + variety.of + " rule orders", String(variety.n), variety.n > 1 ? THEME.highlight : THEME.text],
            ]}
          />
          {variety.n > 1 ? <Note>The orders give: {variety.list.map((s) => "[" + s.split(" ").join(", ") + "]").join(" or ")}.</Note> : null}
        </Panel>
        <Note>
          ? = a toy token with no ID on slide 29. Try slowest, widest, tow, or a Persian letter such as <Fa>ک</Fa>.
        </Note>
      </>
    );
  } else if (screen === 2) {
    const fin = steps[steps.length - 1].toks;
    question = (
      <>
        <b>What you found:</b> inference applies the saved rules by rank, earliest first, until no adjacent pair has a rule. Nothing in the new text is counted. Rank decides the result only when two eligible pairs share a symbol, as (w, e) and (e, s) share the e of lowest: the earlier rule takes it. With slide 29's five rules no two ever compete, so every order gives the same tokens. An unseen word still comes out as known pieces, and the IDs are looked up at the end (slides 30–32).
      </>
    );
    stage = (
      <>
        <ApplyTrace steps={steps} />
        <Eq size={12.5}>
          tokens ← characters(word) + &lt;/w&gt;
          <br />
          repeat: E ← {"{"} adjacent pairs (a, b) that have a saved rule {"}"}
          <br />
          &nbsp;&nbsp;if E is empty: stop
          <br />
          &nbsp;&nbsp;(a, b) ← argmin over E of rank(a, b) ; replace every adjacent a b by ab
          <br />
          IDs ← vocabulary lookup of tokens
        </Eq>
        <Readout
          rows={[
            [word || "(empty)", "[" + fin.join(", ") + "] → [" + fin.map(apIdOf).join(", ") + "]"],
            ["slides 31–32", "lowest → [low, est</w>] → [15, 13]"],
          ]}
        />
        <Note>Rank and ID are different numbers. Rank is the order in which training learned a rule; the ID is the token's row in the embedding matrix (lesson 1). Rule 1 makes es, whose ID is 11.</Note>
      </>
    );
    side = (
      <>
        <Panel>
          <ApplyWordBox id="ap-word3" value={rawWord} onChange={onWord} />
        </Panel>
        <Panel>
          <Label>decoding: click IDs to build a sequence</Label>
          <div style={{ display: "flex", flexWrap: "wrap", gap: 4 }}>
            {Object.keys(AP_ID).map((t) => (
              <ApplyIdChip key={t} id={AP_ID[t]} tok={t} onPick={addId} />
            ))}
          </div>
          <Readout
            rows={[
              ["IDs", "[" + ids.join(", ") + "]"],
              ["joined", ids.length ? apDecode(ids).split(" ").join("␣") : "—"],
              ["text", ids.length ? "“" + apDecode(ids).trim() + "”" : "—", THEME.highlight],
            ]}
          />
          <div style={{ display: "flex", gap: 6, marginTop: 6 }}>
            <Btn onClick={undoId} disabled={!ids.length}>
              undo
            </Btn>
            <Btn onClick={clearIds} disabled={!ids.length}>
              clear
            </Btn>
          </div>
        </Panel>
        <Note>
          &lt;/w&gt; is not a letter of the text. Decoding joins the tokens and turns each &lt;/w&gt; into a space, so [15, 13] gives back “lowest” (slide 32). Next lesson, <a href="#special" style={{ color: THEME.accent }}>special tokens</a>, adds IDs that no merge ever produces, such as [BOS] = 16 and [EOS] = 17 (slide 33).
        </Note>
      </>
    );
  } else {
    const res = apChallenge(order);
    question = (
      <>
        A tokenizer has seven rules: slide 29's five plus two toy rules, w + e → we and o + w → ow. On the text <b>lower newest lowest tow</b> it must output the target tokens shown for each word. Someone ranked the rules by how often each pair appears in this text, most frequent first. Re-rank the rules with ▲ ▼ until all four words come out as their targets.
      </>
    );
    stage = (
      <>
        {AP_CH_WORDS.map((w, i) => {
          const tr = res.traces[i];
          return (
            <Panel key={w} edge={res.ok[i] ? THEME.ok : THEME.grid} style={{ marginBottom: 8 }}>
              <div style={{ display: "flex", justifyContent: "space-between", gap: 8, flexWrap: "wrap" }}>
                <Goal met={res.ok[i]}>
                  {w} → target [{AP_CH_TARGET[i].join(", ")}]
                </Goal>
              </div>
              <ApplyRow toks={res.outs[i]} />
              <div style={{ fontFamily: MONO, fontSize: 12, color: THEME.muted }}>
                merges in order: {tr.length > 1 ? tr.slice(1).map((s) => s.pair[0] + "+" + s.pair[1]).join(" → ") : "none"}
              </div>
            </Panel>
          );
        })}
      </>
    );
    side = (
      <>
        <Panel edge={res.pass ? THEME.ok : THEME.grid}>
          <Label>rank · rule · times the pair occurs in this text</Label>
          {order.map((ri, pos) => (
            <ApplyOrderRow key={ri} pos={pos} n={order.length} rule={AP_CH_RULES[ri]} count={apCount(AP_CH_RULES[ri])} onMove={onMove} />
          ))}
          <div style={{ marginTop: 8 }}>
            <Goal met={res.pass}>all four words match their targets ({res.ok.filter(Boolean).length} / 4)</Goal>
            <Btn onClick={resetOrder}>back to the count order</Btn>
          </div>
          <Hint
            hints={[
              "Start with a word that fails and read its merges. Find the first merge that takes a symbol the target puts in a different token.",
              "When two eligible rules want the same symbol, the one with the earlier rank gets it. The count column never enters the algorithm, so a pair seen twice may need to outrank a pair seen three times.",
            ]}
          />
        </Panel>
      </>
    );
  }

  return <LessonFrame screen={screen} onScreen={setScreen} title="BPE inference: apply rules by rank" question={question} stage={stage} side={side} />;
}
