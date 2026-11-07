import { useState, useMemo, useCallback } from "react";
import { THEME, MONO } from "./common";

// Lesson 6 - Unseen words from known pieces (Word Embedding, slides 26-28).
// Aha: Word2Vec has no vector for a word it never saw. FastText builds every word's vector as the sum
// of its character n-gram vectors, so an unseen word inherits meaning from the pieces it shares with known
// words. The pieces are not morphemes, and one string still gets one vector, whatever it means in context.

// @sim-start
const KNOWN = ["play", "player", "played", "cat", "act", "category", "catalog", "نانوا", "نان", "کتاب", "خانه", "walk", "talk"];
const TESTS = [
  ["playing", ["play", "player", "played"]],
  ["cats", ["cat"]],
  ["acts", ["act"]],
  ["نانوایی", ["نانوا"]],
  ["کتابها", ["کتاب"]],
  ["walking", ["walk"]],
];
const D = 48;
const GRAM_BUDGET = 6.5;
const VCACHE = {};
// every n-gram gets a fixed random vector, so only shared pieces can make two words similar
function gramVec(g) {
  if (!VCACHE[g]) {
    const r = mulberry32(hashStr(g));
    const z = new Array(D);
    for (let i = 0; i < D; i++) z[i] = randn(r);
    VCACHE[g] = z;
  }
  return VCACHE[g];
}
function grams(w, nmin, nmax) {
  const s = "<" + w + ">";
  const ch = Array.from(s);
  const out = [];
  for (let n = nmin; n <= nmax; n++) for (let i = 0; i + n <= ch.length; i++) {
    const g = ch.slice(i, i + n).join("");
    if (g !== s) out.push(g);
  }
  return out;
}
// a known word also has a vector for the whole word "<w>"; an unseen word does not
function ftVec(w, nmin, nmax, known) {
  const gs = grams(w, nmin, nmax);
  if (known) gs.push("<" + w + ">");
  const v = new Array(D).fill(0);
  for (const g of gs) {
    const z = gramVec(g);
    for (let i = 0; i < D; i++) v[i] += z[i];
  }
  return v;
}
function nearestKnown(w, nmin, nmax) {
  const v = ftVec(w, nmin, nmax, KNOWN.indexOf(w) >= 0);
  if (norm(v) < 1e-9) return [];
  return KNOWN.filter((k) => k !== w)
    .map((k) => [k, cosine(v, ftVec(k, nmin, nmax, true))])
    .sort((a, b) => b[1] - a[1]);
}
function avgGrams(nmin, nmax) {
  const all = KNOWN.concat(TESTS.map((t) => t[0]));
  return all.reduce((s, w) => s + grams(w, nmin, nmax).length, 0) / all.length;
}
// @sim-end

const QUICK = ["playing", "cats", "acts", "نانوایی", "کتابها", "walking", "catalogs", "talked"];
const FA_RE = /[؀-ۿ]/;

function TextIn({ value, onChange }) {
  const change = useCallback((e) => onChange(e.target.value.trim().slice(0, 18)), [onChange]);
  return (
    <input
      value={value}
      onChange={change}
      aria-label="word to look up"
      dir="auto"
      style={{ width: "100%", fontFamily: FA_RE.test(value) ? FA : MONO, fontSize: 18, padding: "6px 10px", background: THEME.bg, color: THEME.text, border: "1px solid " + THEME.accent, borderRadius: 4 }}
    />
  );
}
function GramChips({ list, shared }) {
  return (
    <div style={{ display: "flex", flexWrap: "wrap", gap: 5, direction: "ltr" }}>
      {list.map((g, i) => {
        const on = shared && shared.has(g);
        return (
          <span key={i} lang={FA_RE.test(g) ? "fa" : undefined} style={{ fontFamily: FA_RE.test(g) ? FA : MONO, fontSize: 13.5, padding: "1px 7px", borderRadius: 4, border: "1px " + (on ? "solid " + THEME.highlight : "dashed " + THEME.grid), background: on ? rgba(THEME.highlight, 0.16) : THEME.bg, color: on ? THEME.text : THEME.muted }}>
            {g}
            {on ? " ✓" : ""}
          </span>
        );
      })}
    </div>
  );
}

export default function SubwordLesson({ initialScreen }) {
  const [screen, setScreen] = useState(initialScreen || 0);
  const [word, setWord] = useState("playing");
  const [nmin, setNmin] = useState(3);
  const [nmax, setNmax] = useState(6);
  const [sense, setSense] = useState("آب");
  const [cmin, setCmin] = useState(3);
  const [cmax, setCmax] = useState(6);

  const setMin = useCallback((v) => {
    setNmin(v);
    setNmax((m) => Math.max(m, v));
  }, []);
  const setMax = useCallback((v) => {
    setNmax(v);
    setNmin((m) => Math.min(m, v));
  }, []);
  const setCMin = useCallback((v) => {
    setCmin(v);
    setCmax((m) => Math.max(m, v));
  }, []);
  const setCMax = useCallback((v) => {
    setCmax(v);
    setCmin((m) => Math.min(m, v));
  }, []);

  const known = KNOWN.indexOf(word) >= 0;
  const quick = (
    <div style={{ display: "flex", flexWrap: "wrap", gap: 5, margin: "6px 0" }}>
      {QUICK.map((q) => (
        <Chip key={q} value={q} on={q === word} onPick={setWord}>
          {q}
        </Chip>
      ))}
    </div>
  );
  const vocabBox = (
    <div style={{ display: "flex", flexWrap: "wrap", gap: 5 }}>
      {KNOWN.map((k) => (
        <span key={k} lang={FA_RE.test(k) ? "fa" : undefined} style={{ fontFamily: FA_RE.test(k) ? FA : MONO, fontSize: 13.5, padding: "1px 7px", border: "1px solid " + (k === word ? THEME.ok : THEME.grid), borderRadius: 4, background: THEME.bg }}>
          {k}
        </span>
      ))}
    </div>
  );

  let question, stage, side;
  if (screen === 0) {
    const g3 = grams(word, 3, 3);
    const rows = KNOWN.map((k) => {
      const kg = new Set(grams(k, 3, 3));
      return { label: k, fa: FA_RE.test(k), value: g3.filter((g) => kg.has(g)).length, color: THEME.accent };
    }).sort((x, y) => y.value - x.value);
    question = (
      <>
        <b>Where we are:</b> Word2Vec and the vectors from lessons 3–5 keep one vector per word in a fixed vocabulary. <b>The puzzle:</b> the vocabulary below knows <i>play</i> but never saw <i>playing</i>, and it knows <Fa>نانوا</Fa> but not <Fa>نانوایی</Fa> (slide 26). Type or pick a word and look it up. What vector does an unseen word get? Could the character pieces it shares with known words help?
      </>
    );
    stage = (
      <>
        <Panel>
          <TextIn value={word} onChange={setWord} />
          {quick}
          <div style={{ fontFamily: MONO, fontSize: 15, margin: "8px 0", color: known ? THEME.ok : THEME.warn }}>{word ? (known ? "◆ in the vocabulary: it has a vector" : "✕ not in the vocabulary: no word-level vector") : ""}</div>
          <Label>the fixed Word2Vec vocabulary</Label>
          {vocabBox}
        </Panel>
        <Label>3-letter pieces of «{word}» shared with each known word</Label>
        <Bars rows={rows.slice(0, 6)} range={[0, Math.max(3, g3.length)]} digits={0} labelW={90} ariaLabel="shared 3-grams" />
      </>
    );
    side = (
      <>
        <Panel>
          <Label>{"pieces of <" + word + "> (3 characters, < and > mark the word's edges)"}</Label>
          <GramChips list={g3} shared={new Set(KNOWN.reduce((acc, k) => acc.concat(grams(k, 3, 3)), []))} />
        </Panel>
        <Note>✓ marks a piece that some known word also has. Word-level lookup throws all of that away: an unseen form simply has no vector.</Note>
      </>
    );
  } else if (screen === 1) {
    const nn = nearestKnown(word, nmin, nmax);
    const top = nn[0];
    const shared = top ? new Set(grams(top[0], nmin, nmax)) : null;
    question = (
      <>
        <b>Claim to test:</b> FastText represents a word by its character n-grams, learns one vector per n-gram, and sums them (slide 27). Here every n-gram has a random vector, so two words can only be similar through shared pieces. Pick a word and the n-gram lengths. Which range lets <i>playing</i> find <i>play</i>? What goes wrong with single characters (try <i>cats</i> and <i>acts</i>), and what goes wrong with long pieces?
      </>
    );
    stage = (
      <>
        <Panel>
          <TextIn value={word} onChange={setWord} />
          {quick}
          <Label>
            {"n-grams of <" + word + ">, n = " + nmin + "…" + nmax}
            {top ? " · ✓ shared with «" + top[0] + "»" : ""}
          </Label>
          <GramChips list={grams(word, nmin, nmax)} shared={shared} />
        </Panel>
        <Label>closest known words, cosine of the summed vectors</Label>
        <Bars rows={nn.slice(0, 6).map((p) => ({ label: p[0], fa: FA_RE.test(p[0]), value: p[1], color: THEME.accent }))} range={[-0.2, 1]} labelW={90} ariaLabel="nearest known words" />
      </>
    );
    side = (
      <>
        <Panel>
          <Slider id="l6-min" label="shortest n-gram" value={nmin} min={1} max={6} stepSize={1} onChange={setMin} display={"n ≥ " + nmin} />
          <Slider id="l6-max" label="longest n-gram" value={nmax} min={1} max={6} stepSize={1} onChange={setMax} display={"n ≤ " + nmax} />
          <Readout
            rows={[
              ["n-grams in this word", String(grams(word, nmin, nmax).length)],
              ["closest known word", top ? top[0] + " (" + fmt(top[1], 2) + ")" : "—"],
            ]}
          />
        </Panel>
        <Note>FastText's default is n = 3…6. With n = 1, cats and acts have the same letters, so they get the same vector. Also try catalogs: pieces like «cat» are not always morphemes (slide 27).</Note>
      </>
    );
  } else if (screen === 2) {
    const sv = ftVec("شیر", 3, 6, true);
    const contexts = ["آب", "جنگل", "صبحانه"];
    question = (
      <>
        <b>What you found:</b> a word's vector is the sum of its n-gram vectors, so an unseen word shares meaning with known words in proportion to the pieces they share. Short pieces collide (anagrams), long pieces stop being shared, and 3…6 is a workable middle. But the vector depends only on the spelling. Pick a context for <Fa>شیر</Fa> (slide 28).
      </>
    );
    stage = (
      <>
        <Panel>
          <div lang="fa" dir="rtl" style={{ fontFamily: FA, fontSize: 26, textAlign: "center", margin: "4px 0 8px" }}>
            <span style={{ color: THEME.highlight }}>شیر</span> {sense}
          </div>
          <Label>{"n-grams of <شیر> (the same for every context)"}</Label>
          <GramChips list={grams("شیر", 3, 6).concat(["<شیر>"])} />
          <div style={{ height: 8 }} />
          <Bars rows={contexts.map((c) => ({ label: "شیر in «شیر " + c + "»", fa: true, value: cosine(sv, ftVec("شیر", 3, 6, true)), color: c === sense ? THEME.highlight : THEME.accent }))} range={[0, 1]} digits={3} labelW={150} ariaLabel="cosine between شیر in each context and شیر in the chosen context" />
        </Panel>
        <Eq size={12.5}>
          G_w = n-grams of {"<w>"} for n = 3…6, plus {"<w>"} itself
          <br />
          v_w = Σ over g ∈ G_w of z_g
          <br />
          s(w, c) = Σ over g ∈ G_w of z_gᵀ u_c
        </Eq>
      </>
    );
    side = (
      <>
        <Panel>
          <Label>context</Label>
          <Pills
            options={[
              ["آب", "شیر آب · tap"],
              ["جنگل", "شیر جنگل · lion"],
              ["صبحانه", "شیر صبحانه · milk"],
            ]}
            value={sense}
            onPick={setSense}
          />
          <Readout
            rows={[
              ["n-grams used for شیر", String(grams("شیر", 3, 6).length + 1)],
              ["cosine with the other contexts", "1.000"],
            ]}
          />
        </Panel>
        <Note>Bojanowski et al. 2017 (slide 27). Subwords fix unseen forms, not ambiguity: Word2Vec and FastText give شیر one vector in every sentence (slide 28). Telling the senses apart needs a model that reads the context. That is what attention does, in the next lab.</Note>
      </>
    );
  } else {
    const res = TESTS.map(([w, good]) => {
      const nn = nearestKnown(w, cmin, cmax);
      return { w, top: nn[0] ? nn[0][0] : "—", c: nn[0] ? nn[0][1] : 0, ok: nn[0] ? good.indexOf(nn[0][0]) >= 0 : false };
    });
    const allOk = res.every((r) => r.ok);
    const avg = avgGrams(cmin, cmax);
    const cheap = avg <= GRAM_BUDGET;
    question = (
      <>
        FastText stores one vector for every distinct n-gram, so fewer n-grams per word means a smaller table. Pick the n-gram range so that all six unseen words find their own relative as the closest known word, using at most {GRAM_BUDGET} n-grams per word on average.
      </>
    );
    stage = (
      <>
        <Bars rows={res.map((r) => ({ label: r.w + " → " + r.top, fa: FA_RE.test(r.w), value: r.c, color: r.ok ? THEME.ok : THEME.warn, mark: r.ok ? "✓" : "✕" }))} range={[-0.2, 1]} labelW={170} ariaLabel="closest known word for each unseen word" />
        <Panel style={{ marginTop: 8 }}>
          <Label>known vocabulary</Label>
          {vocabBox}
        </Panel>
      </>
    );
    side = (
      <>
        <Panel edge={allOk && cheap ? THEME.ok : THEME.grid}>
          <Slider id="l6-cmin" label="shortest n-gram" value={cmin} min={1} max={6} stepSize={1} onChange={setCMin} display={"n ≥ " + cmin} />
          <Slider id="l6-cmax" label="longest n-gram" value={cmax} min={1} max={6} stepSize={1} onChange={setCMax} display={"n ≤ " + cmax} />
          <Readout
            rows={[
              ["n-grams per word (average)", fmt(avg, 1), cheap ? THEME.ok : THEME.text],
              ["unseen words placed right", res.filter((r) => r.ok).length + " / " + res.length],
            ]}
          />
          <Goal met={allOk}>all six unseen words find their relative</Goal>
          <Goal met={cheap}>at most {GRAM_BUDGET} n-grams per word on average</Goal>
          <Hint
            hints={[
              "The default 3…6 gets every word right but uses about 14 n-grams per word. A single length uses far fewer. Which single lengths still work?",
              "n = 1 merges anagrams (cats and acts), and n ≥ 4 is too long for short words like cat. Try a single n of 2 or 3.",
            ]}
          />
        </Panel>
      </>
    );
  }

  return <LessonFrame screen={screen} onScreen={setScreen} title="Unseen words from known pieces" question={question} stage={stage} side={side} />;
}
