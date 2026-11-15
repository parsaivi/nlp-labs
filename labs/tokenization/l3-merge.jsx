import { useState, useCallback } from "react";
import { THEME, MONO } from "./common";

// Lesson 3 - BPE training: merge the most frequent pair (Tokenization, slides 22-29).
// Aha: BPE training is greedy counting. Every adjacent pair is counted over occurrences, each word weighted by
// its frequency; the winner is glued in every word, the rule is saved, and the counts are recomputed. So the
// vocabulary is whatever the corpus repeats most: est (in newest x6 + widest x3 = 9) beats low (low x5 +
// lower x2 = 7), even though low is a whole word. The output is the base symbols, one new token per merge,
// and the merge order.
//
// Numbers from the slides: the corpus low x5, lower x2, newest x6, widest x3 and M = 5 (slide 25); the tie rule
// "lexicographic pair order" (slide 25); the five merges and their counts 9, 9, 9, 7, 7 with the ties named on
// slides 26-28; the pair counts on slide 26 (6+3 = 9, 8, 7); 11 base symbols + 5 learned = 16 tokens and the
// illustrative IDs </w>, d, e, i, l, n, o, r, s, t, w = 0-10, es = 11, est = 12, est</w> = 13, lo = 14,
// low = 15 (slide 29). All reproduced by mgTrain below.
// Teaching toys (not on the slides): merges 6-10 of the same corpus (Name screen, M slider up to 10), every
// frequency the learner sets with the sliders, and the challenge goal (learn low</w> and est within M = 5).

// @sim-start
const MG_WORDS = ["low", "lower", "newest", "widest"];
const MG_SLIDE_F = [5, 2, 6, 3];
const MG_SLIDE_M = 5;
const MG_EOW = "</w>";
function mgSplit(w) {
  return w.split("").concat([MG_EOW]);
}
// lexicographic pair order: compare the first symbols as strings, then the second ones ("<" sorts before letters)
function mgPairLess(a1, b1, a2, b2) {
  return a1 < a2 || (a1 === a2 && b1 < b2);
}
// every adjacent pair, its count summed over occurrences x word frequency, and the per-word terms of that sum
function mgCount(segs, freqs) {
  const map = {};
  const list = [];
  for (let wi = 0; wi < segs.length; wi++) {
    const s = segs[wi];
    for (let i = 0; i + 1 < s.length; i++) {
      const key = s[i] + " " + s[i + 1];
      let e = map[key];
      if (!e) {
        e = { a: s[i], b: s[i + 1], key: key, count: 0, terms: [] };
        map[key] = e;
        list.push(e);
      }
      e.count += freqs[wi];
      let t = null;
      for (const x of e.terms) if (x.w === wi) t = x;
      if (t) t.n += 1;
      else e.terms.push({ w: wi, n: 1, f: freqs[wi] });
    }
  }
  list.sort((p, q) => q.count - p.count || (mgPairLess(p.a, p.b, q.a, q.b) ? -1 : 1));
  return list.filter((e) => e.count > 0);
}
// merge the non-overlapping occurrences of (a, b), left to right
function mgMerge(s, a, b) {
  const out = [];
  for (let i = 0; i < s.length; i++) {
    if (i + 1 < s.length && s[i] === a && s[i + 1] === b) {
      out.push(a + b);
      i++;
    } else out.push(s[i]);
  }
  return out;
}
function mgTrain(words, freqs, M) {
  let segs = words.map(mgSplit);
  const steps = [];
  for (let k = 0; k < M; k++) {
    const counts = mgCount(segs, freqs);
    if (!counts.length) break;
    const best = counts[0];
    const before = segs;
    segs = segs.map((s) => mgMerge(s, best.a, best.b));
    steps.push({ counts: counts, best: best, before: before, after: segs, token: best.a + best.b });
  }
  return { steps: steps, segs: segs, start: words.map(mgSplit) };
}
function mgBase(words) {
  const set = {};
  for (const w of words) for (const s of mgSplit(w)) set[s] = 1;
  return Object.keys(set).sort();
}
function mgVocab(words, steps) {
  return mgBase(words).concat(steps.map((s) => s.token));
}
function mgTerms(e) {
  const parts = e.terms.map((t) => (t.n > 1 ? t.n + "×" + t.f : String(t.f)));
  return parts.length > 1 ? parts.join("+") + " = " + e.count : String(e.count);
}
// challenge: with M = 5 and only the four frequencies changed, learn the whole word low</w> AND the piece est
const MG_CH_M = 5;
const MG_CH_START = [5, 2, 6, 3];
function mgChallenge(f) {
  const toks = mgTrain(MG_WORDS, f, MG_CH_M).steps.map((s) => s.token);
  const low = toks.indexOf("low</w>") >= 0;
  const est = toks.indexOf("est") >= 0;
  return { toks: toks, low: low, est: est, ok: low && est };
}
// @sim-end

const MG_EXPLORE_ID = "t3";

// positions (word, index) covered by the non-overlapping occurrences of (a, b), left to right, as in mgMerge
function mgOccur(segs, a, b) {
  const hit = {};
  for (let wi = 0; wi < segs.length; wi++) {
    const s = segs[wi];
    for (let i = 0; i + 1 < s.length; i++) {
      if (s[i] === a && s[i + 1] === b) {
        hit[wi + ":" + i] = 1;
        hit[wi + ":" + (i + 1)] = 1;
        i++;
      }
    }
  }
  return hit;
}
function mgPairText(a, b) {
  return "(" + a + ", " + b + ")";
}

// one token box. base symbols: dashed border; learned tokens: solid violet border; the token made by the
// latest merge: apricot fill and a ★; tokens about to be merged next: apricot underline
function MgTok({ text, learned, fresh, next }) {
  const eow = text.length >= 4 && text.slice(-4) === MG_EOW;
  const body = eow ? text.slice(0, -4) : text;
  return (
    <span
      style={{
        display: "inline-block",
        fontFamily: MONO,
        fontSize: 14,
        lineHeight: 1.3,
        padding: "3px 6px",
        borderRadius: 4,
        border: "1px " + (learned ? "solid " + (fresh ? THEME.highlight : THEME.accent) : "dashed " + THEME.grid),
        background: fresh ? rgba(THEME.highlight, 0.18) : learned ? rgba(THEME.accent, 0.12) : THEME.bg,
        color: THEME.text,
        fontWeight: learned ? 600 : 400,
        boxShadow: next ? "inset 0 -3px 0 " + THEME.highlight : "none",
        whiteSpace: "nowrap",
      }}
    >
      {fresh ? <span style={{ color: THEME.highlight, fontWeight: 400 }}>★</span> : null}
      {body}
      {eow ? <span style={{ color: THEME.muted, fontWeight: 400 }}>{MG_EOW}</span> : null}
    </span>
  );
}
// a clickable gap between two symbols (Puzzle screen)
function MgGap({ id, on, onPick }) {
  const click = useCallback(() => onPick(id), [onPick, id]);
  return (
    <button
      onClick={click}
      aria-pressed={on}
      aria-label="guess this pair"
      style={{
        fontFamily: MONO,
        fontSize: 12,
        width: 15,
        height: 26,
        padding: 0,
        margin: "0 1px",
        borderRadius: 9,
        cursor: "pointer",
        border: "1px " + (on ? "solid " + THEME.highlight : "dashed " + THEME.grid),
        background: on ? rgba(THEME.highlight, 0.25) : "transparent",
        color: on ? THEME.highlight : THEME.muted,
      }}
    >
      {on ? "★" : "+"}
    </button>
  );
}
// the corpus: one row per word, its frequency, and its current segmentation
function MgCorpus({ segs, freqs, fresh, next, pick, onPick }) {
  const occ = next ? mgOccur(segs, next.a, next.b) : {};
  return (
    <div style={{ background: THEME.bg, border: "1px solid " + THEME.grid, borderRadius: 6, padding: "8px 10px" }}>
      <div style={{ display: "grid", gridTemplateColumns: "auto 1fr", gap: "8px 10px", alignItems: "center" }}>
        <div style={{ fontFamily: MONO, fontSize: 11, color: THEME.muted, letterSpacing: 1 }}>WORD × f</div>
        <div style={{ fontFamily: MONO, fontSize: 11, color: THEME.muted, letterSpacing: 1 }}>CURRENT TOKENS</div>
        {segs.map((s, wi) => (
          <React.Fragment key={MG_WORDS[wi]}>
            <div style={{ fontFamily: MONO, fontSize: 13.5, color: THEME.text, whiteSpace: "nowrap" }}>
              {MG_WORDS[wi]} <span style={{ color: THEME.highlight }}>×{freqs[wi]}</span>
            </div>
            <div style={{ display: "flex", flexWrap: "wrap", gap: onPick ? 0 : 4, alignItems: "center", minWidth: 0 }}>
              {s.map((t, i) => (
                <React.Fragment key={i}>
                  {onPick && i > 0 ? <MgGap id={wi + ":" + i} on={pick === wi + ":" + i} onPick={onPick} /> : null}
                  <MgTok text={t} learned={t.length > 1 && t !== MG_EOW} fresh={!!fresh && t === fresh} next={!!occ[wi + ":" + i]} />
                </React.Fragment>
              ))}
            </div>
          </React.Fragment>
        ))}
      </div>
    </div>
  );
}
// pair counts, sorted by count and then by lexicographic pair order (the order BPE breaks ties in)
function MgPairTable({ counts, limit, best, mine }) {
  const rows = counts.slice(0, limit || counts.length);
  const top = counts.length ? counts[0].count : 1;
  return (
    <div>
      <div style={{ display: "grid", gridTemplateColumns: "auto minmax(40px, 1fr) auto", gap: "3px 8px", alignItems: "center", fontFamily: MONO, fontSize: 13 }}>
        {rows.map((e) => {
          const isBest = !!best && e.key === best.key;
          const tie = !!best && !isBest && e.count === best.count;
          const isMine = !!mine && e.key === mine;
          return (
            <React.Fragment key={e.key}>
              <span style={{ color: isBest ? THEME.highlight : THEME.text, whiteSpace: "nowrap", paddingLeft: 4, borderLeft: "3px solid " + (isBest ? THEME.highlight : "transparent") }}>{mgPairText(e.a, e.b)}</span>
              <span style={{ position: "relative", height: 16, background: THEME.bg, border: "1px solid " + THEME.grid, borderRadius: 2 }}>
                <span style={{ position: "absolute", left: 0, top: 0, bottom: 0, width: (100 * e.count) / top + "%", background: isBest ? THEME.highlight : isMine ? THEME.third : THEME.accent, opacity: 0.75 }} />
              </span>
              <span style={{ whiteSpace: "nowrap", color: THEME.text, fontVariantNumeric: "tabular-nums" }}>
                {mgTerms(e)}
                {isBest ? <span style={{ color: THEME.highlight }}> ★ merge</span> : null}
                {tie ? <span style={{ color: THEME.muted }}> tie, later in order</span> : null}
                {isMine ? <span style={{ color: THEME.third }}> ◀ your pick</span> : null}
              </span>
            </React.Fragment>
          );
        })}
      </div>
      {limit && counts.length > limit ? <div style={{ fontFamily: MONO, fontSize: 12, color: THEME.muted, padding: "2px 4px" }}>+ {counts.length - limit} more pairs with smaller counts</div> : null}
    </div>
  );
}
// the saved, ordered merge rules; optional count and the runner-up it beat
function MgRules({ steps, upto, showCount, empty }) {
  const n = upto === undefined ? steps.length : Math.min(upto, steps.length);
  if (n === 0) return <div style={{ fontFamily: MONO, fontSize: 13, color: THEME.muted }}>{empty || "no rules yet"}</div>;
  return (
    <div style={{ fontFamily: MONO, fontSize: 13, lineHeight: 1.65 }}>
      {steps.slice(0, n).map((s, k) => {
        const ru = s.counts[1];
        return (
          <div key={k} style={{ display: "flex", flexWrap: "wrap", gap: "0 10px" }}>
            <span style={{ color: k === n - 1 ? THEME.highlight : THEME.accent, minWidth: 18 }}>{k + 1}</span>
            <span style={{ color: THEME.text }}>
              {s.best.a} + {s.best.b} → <b>{s.token}</b>
            </span>
            {showCount ? (
              <span style={{ color: THEME.muted }}>
                {mgTerms(s.best)}
                {ru ? (ru.count === s.best.count ? " · tie with " : " · beat ") + mgPairText(ru.a, ru.b) + " " + ru.count : ""}
              </span>
            ) : null}
          </div>
        );
      })}
    </div>
  );
}
function MgVocab({ vocab, nBase }) {
  return (
    <div style={{ display: "flex", flexWrap: "wrap", gap: 5 }}>
      {vocab.map((t, i) => (
        <span key={i} style={{ display: "inline-flex", alignItems: "baseline", gap: 4 }}>
          <span style={{ fontFamily: MONO, fontSize: 11, color: THEME.muted }}>{i}</span>
          <MgTok text={t} learned={i >= nBase} fresh={false} next={false} />
        </span>
      ))}
    </div>
  );
}

export default function MergeLesson({ initialScreen }) {
  const [screen, setScreen] = useState(initialScreen || 0);
  // puzzle
  const [pick, setPick] = useState(null);
  const [showAll, setShowAll] = useState(false);
  // explore
  const [freqs, setFreqs] = useState(MG_SLIDE_F);
  const [M, setM] = useState(MG_SLIDE_M);
  const [k, setK] = useState(0);
  // name
  const [nM, setNM] = useState(MG_SLIDE_M);
  // challenge
  const [cf, setCf] = useState(MG_CH_START);

  const setF = useCallback((i, v) => setFreqs((F) => F.map((x, j) => (j === i ? v : x))), []);
  const setF0 = useCallback((v) => setF(0, v), [setF]);
  const setF1 = useCallback((v) => setF(1, v), [setF]);
  const setF2 = useCallback((v) => setF(2, v), [setF]);
  const setF3 = useCallback((v) => setF(3, v), [setF]);
  const setC = useCallback((i, v) => setCf((F) => F.map((x, j) => (j === i ? v : x))), []);
  const setC0 = useCallback((v) => setC(0, v), [setC]);
  const setC1 = useCallback((v) => setC(1, v), [setC]);
  const setC2 = useCallback((v) => setC(2, v), [setC]);
  const setC3 = useCallback((v) => setC(3, v), [setC]);
  const flipAll = useCallback(() => setShowAll((v) => !v), []);

  const trace = mgTrain(MG_WORDS, freqs, M);
  const maxK = trace.steps.length;
  const kk = Math.min(k, maxK);
  const stepFwd = useCallback(() => setK((v) => Math.min(Math.min(v, maxK) + 1, maxK)), [maxK]);
  const stepBack = useCallback(() => setK((v) => Math.max(0, Math.min(v, maxK) - 1)), [maxK]);
  const runAll = useCallback(() => setK(maxK), [maxK]);
  const resetEx = useCallback(() => {
    setFreqs(MG_SLIDE_F);
    setM(MG_SLIDE_M);
    setK(0);
  }, []);
  const resetCh = useCallback(() => setCf(MG_CH_START), []);

  let question, stage, side;
  if (screen === 0) {
    const start = MG_WORDS.map(mgSplit);
    const counts = mgCount(start, MG_SLIDE_F);
    let mine = null;
    if (pick) {
      const wi = parseInt(pick.split(":")[0], 10);
      const i = parseInt(pick.split(":")[1], 10);
      const key = start[wi][i - 1] + " " + start[wi][i];
      for (const e of counts) if (e.key === key) mine = e;
    }
    const top = counts[0].count;
    const types = mine ? mine.terms.length : 0;
    question = (
      <>
        <b>Where we are:</b> lesson 2 showed the trade-off. Word tokens turn unseen words into [UNK]; character tokens make sequences long, and attention costs O(n²d). Slide 21 asks which units belong in the vocabulary. Sennrich, Haddow and Birch (ACL 2016) answered with byte pair encoding, a compression method adapted to subwords: start from characters and keep gluing the most frequent pair of neighbours (slides 22–24). <b>The puzzle:</b> here is slide 25's corpus, each word split into characters plus the end-of-word marker &lt;/w&gt;. Which pair does BPE glue first? Click a + between two symbols.
      </>
    );
    stage = (
      <>
        <MgCorpus segs={start} freqs={MG_SLIDE_F} pick={pick} onPick={setPick} />
        {showAll ? (
          <div style={{ marginTop: 10 }}>
            <Label>every adjacent pair, counted over the whole corpus</Label>
            <MgPairTable counts={counts} mine={mine ? mine.key : null} />
          </div>
        ) : null}
      </>
    );
    side = (
      <>
        <Panel>
          <Label>your guess</Label>
          {mine ? (
            <Readout
              rows={[
                ["pair", mgPairText(mine.a, mine.b), THEME.third],
                ["appears in", mine.terms.map((t) => MG_WORDS[t.w] + " ×" + t.f).join(", ")],
                ["word types containing it", String(types)],
                ["count in the corpus", mgTerms(mine), mine.count === top ? THEME.ok : THEME.text],
              ]}
            />
          ) : (
            <Note>No guess yet. Click a + in the corpus.</Note>
          )}
          <Goal met={!!mine && mine.count === top}>your pair has the highest count in the corpus</Goal>
          <Toggle label="show every pair count" value={showAll} onChange={flipAll} />
        </Panel>
        <Note>
          {!mine
            ? "A pair's count is how many times its two symbols sit side by side in the corpus, so a word that appears 6 times adds 6."
            : mine.count === top
            ? "Three pairs tie at 9 = 6 + 3: (e, s), (s, t) and (t, </w>), all from newest ×6 and widest ×3. low ×5 + lower ×2 gives (l, o) only 7. Which of the three tied pairs goes first? That is the next screen."
            : "Some pair sits side by side more often than " + mine.count + " times. Count occurrences, not word types: the frequencies are the weights."}
        </Note>
      </>
    );
  } else if (screen === 1) {
    const segs = kk === 0 ? trace.start : trace.steps[kk - 1].after;
    const nextStep = kk < maxK ? trace.steps[kk] : null;
    const counts = nextStep ? nextStep.counts : mgCount(segs, freqs);
    const fresh = kk > 0 ? trace.steps[kk - 1].token : null;
    const ties = nextStep ? counts.filter((e) => e.count === nextStep.best.count).length : 0;
    question = (
      <>
        <b>Claim to test (slide 24):</b> count every adjacent pair, weighted by word frequency; merge the most frequent pair in every word; save the rule; recount; repeat up to M times. A tie goes to the pair that comes first in lexicographic order, comparing the first symbols as strings, then the second ones. Press <i>merge</i> five times and compare with slides 26–28. Then reset and raise the frequency of low: from which value is lo merged before es, and why that value?
      </>
    );
    stage = (
      <>
        <MgCorpus segs={segs} freqs={freqs} fresh={fresh} next={nextStep ? nextStep.best : null} />
        <div style={{ marginTop: 10 }}>
          <Label>{nextStep ? "pair counts now: merge " + (kk + 1) + " picks the ★ row" : "budget used: " + kk + " of M = " + M + " merges done"}</Label>
          <MgPairTable counts={counts} limit={7} best={nextStep ? nextStep.best : null} />
        </div>
      </>
    );
    side = (
      <>
        <Panel>
          <div style={{ display: "flex", gap: 6, flexWrap: "wrap", marginBottom: 8 }}>
            <Btn onClick={stepBack} disabled={kk === 0}>
              ◂ undo
            </Btn>
            <Btn onClick={stepFwd} disabled={kk >= maxK} strong>
              merge {kk < maxK ? kk + 1 : ""} ▸
            </Btn>
            <Btn onClick={runAll} disabled={kk >= maxK}>
              run all M
            </Btn>
            <Btn onClick={resetEx}>reset</Btn>
          </div>
          <Readout
            rows={[
              ["merges done", kk + " of M = " + M],
              ["next merge", nextStep ? mgPairText(nextStep.best.a, nextStep.best.b) + " → " + nextStep.token : "none (budget used)", nextStep ? THEME.highlight : THEME.muted],
              ["pairs tied at the top", nextStep ? String(ties) : "—"],
              ["vocabulary size", "11 + " + kk + " = " + (11 + kk)],
            ]}
          />
        </Panel>
        <Panel>
          <Label>saved rules, in order</Label>
          <MgRules steps={trace.steps} upto={kk} showCount />
        </Panel>
        <Panel>
          <Label>word frequencies f and budget M</Label>
          <Slider id={MG_EXPLORE_ID + "-f0"} label="f(low)" value={freqs[0]} min={1} max={10} stepSize={1} onChange={setF0} display={String(freqs[0])} />
          <Slider id={MG_EXPLORE_ID + "-f1"} label="f(lower)" value={freqs[1]} min={1} max={10} stepSize={1} onChange={setF1} display={String(freqs[1])} />
          <Slider id={MG_EXPLORE_ID + "-f2"} label="f(newest)" value={freqs[2]} min={1} max={10} stepSize={1} onChange={setF2} display={String(freqs[2])} />
          <Slider id={MG_EXPLORE_ID + "-f3"} label="f(widest)" value={freqs[3]} min={1} max={10} stepSize={1} onChange={setF3} display={String(freqs[3])} />
          <Slider id={MG_EXPLORE_ID + "-m"} label="merge budget M" value={M} min={1} max={10} stepSize={1} onChange={setM} display={"M = " + M} />
        </Panel>
        <Note>(l, o) and (o, w) occur in exactly the same words, so they always tie; lexicographic order picks (l, o) (slide 28). Each merge changes the counts around it, which is why BPE recounts after every merge.</Note>
      </>
    );
  } else if (screen === 2) {
    const t = mgTrain(MG_WORDS, MG_SLIDE_F, nM);
    const base = mgBase(MG_WORDS);
    const vocab = mgVocab(MG_WORDS, t.steps);
    question = (
      <>
        <b>What you found:</b> BPE training is greedy counting. Each pair's count is summed over every occurrence, so a word seen 6 times adds 6. The most frequent pair is merged in every word, ties go to the first pair in lexicographic order, the rule is saved, and the counts are recomputed. Training returns the base symbols, one new token per merge, and the merge order (slides 24, 29). est came before low because 6 + 3 = 9 beats 5 + 2 = 7. Move M and watch the vocabulary grow.
      </>
    );
    stage = (
      <>
        <Eq size={12.5}>
          count(a, b) = Σ_w f(w) · n_w(a, b)
          <br />
          merge k: (a*, b*) = argmax count(a, b); ties → first (a, b) in string order
          <br />
          save rule k: a* + b* → a*b*, apply it in every word, recount
          <br />
          |V| = |base symbols| + M = {base.length} + {t.steps.length} = {vocab.length}
        </Eq>
        <Label>token vocabulary with IDs (base symbols sorted, then one ID per merge)</Label>
        <MgVocab vocab={vocab} nBase={base.length} />
        <div style={{ marginTop: 10 }}>
          <Label>ordered merge rules</Label>
          <MgRules steps={t.steps} showCount empty="M = 0: only the base symbols" />
        </div>
      </>
    );
    side = (
      <>
        <Panel>
          <Slider id="t3-nm" label="merge budget M" value={nM} min={0} max={10} stepSize={1} onChange={setNM} display={"M = " + nM} />
          <Readout
            rows={[
              ["f(w)", "frequency of word w"],
              ["n_w(a, b)", "times a, b are adjacent in w"],
              ["base symbols", base.length + " (characters + </w>)"],
              ["learned tokens", String(t.steps.length)],
              ["vocabulary size", String(vocab.length), nM === MG_SLIDE_M ? THEME.ok : THEME.text],
            ]}
          />
        </Panel>
        <Note>At M = 5 this is slide 29: 11 + 5 = 16 tokens, es = 11 … low = 15 (the IDs are illustrative). "String order" puts &lt; before letters, so &lt;/w&gt; gets ID 0. Merges 6–10 continue the same rule: whole frequent words such as newest&lt;/w&gt; become single tokens.</Note>
        <Note>
          Keep both the vocabulary and the merge order. Next lesson, <a href="#apply" style={{ color: THEME.accent }}>BPE inference</a>, freezes these rules and applies them by rank to words training never saw (slides 30–32).
        </Note>
      </>
    );
  } else {
    const t = mgTrain(MG_WORDS, cf, MG_CH_M);
    const r = mgChallenge(cf);
    const fresh = t.steps.length ? t.steps[t.steps.length - 1].token : null;
    question = (
      <>
        Same four words, same budget M = {MG_CH_M}. Change only the word frequencies (1 to 10) so that training learns both the whole word <b>low&lt;/w&gt;</b> as one token and the piece <b>est</b>. With slide 25's frequencies, low&lt;/w&gt; is not learned until merge 9.
      </>
    );
    stage = (
      <>
        <MgCorpus segs={t.segs} freqs={cf} fresh={fresh} />
        <div style={{ marginTop: 10 }}>
          <Label>the {MG_CH_M} rules training saves with these frequencies</Label>
          <MgRules steps={t.steps} showCount />
        </div>
      </>
    );
    side = (
      <>
        <Panel edge={r.ok ? THEME.ok : THEME.grid}>
          <Slider id="t3-c0" label="f(low)" value={cf[0]} min={1} max={10} stepSize={1} onChange={setC0} display={String(cf[0])} />
          <Slider id="t3-c1" label="f(lower)" value={cf[1]} min={1} max={10} stepSize={1} onChange={setC1} display={String(cf[1])} />
          <Slider id="t3-c2" label="f(newest)" value={cf[2]} min={1} max={10} stepSize={1} onChange={setC2} display={String(cf[2])} />
          <Slider id="t3-c3" label="f(widest)" value={cf[3]} min={1} max={10} stepSize={1} onChange={setC3} display={String(cf[3])} />
          <Goal met={r.low}>low&lt;/w&gt; is a learned token</Goal>
          <Goal met={r.est}>est is a learned token</Goal>
          <Btn onClick={resetCh}>back to slide 25's frequencies</Btn>
          <Hint
            hints={[
              "Count the merges each target needs. Each merge glues two neighbours, so l o w </w> → low</w> takes 3 merges and e s t → est takes 2. That is all of M = 5: est</w> or any other token must not win a merge.",
              "Once low exists, (low, </w>) occurs only in the word low, because in lower low is followed by e. So its count is f(low) alone, and it must beat the est pairs, which count f(newest) + f(widest). The est pairs in turn must beat lower's own pairs such as (e, r).",
            ]}
          />
        </Panel>
        <Note>The rule list shows each merge's count and the pair it beat or tied with. Read it to see which merge went where you did not want it.</Note>
      </>
    );
  }

  return <LessonFrame screen={screen} onScreen={setScreen} title="BPE training: merge the most frequent pair" question={question} stage={stage} side={side} />;
}
