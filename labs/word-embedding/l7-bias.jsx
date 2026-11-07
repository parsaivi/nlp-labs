import { useState, useMemo, useCallback } from "react";
import { THEME, MONO } from "./common";

// Lesson 7 - Vectors inherit associations (Word Embedding, slides 31-35).
// Aha: WEAT turns "does this embedding associate jobs with gender?" into numbers: a per-word score
// (mean cosine to male terms minus mean cosine to female terms), a gap between two groups of words,
// and a permutation test asking how often a random regrouping gives a gap that large. With 4 words
// per group, the Persian fastText gap (+0.034) is not distinguishable from chance (p = 0.34).

// @sim-start
// slide 33 and 34: fastText Persian Wikipedia, score = mean cos to {مرد، پسر، پدر، برادر} − mean cos to {زن، دختر، مادر، خواهر}
const DATA = {
  occ: {
    name: "occupations (slide 33)",
    groups: ["technical jobs", "education & care"],
    words: [
      ["مهندس", 0.044],
      ["برنامه‌نویس", -0.012],
      ["مکانیک", 0.021],
      ["فیزیکدان", 0.072],
      ["پرستار", -0.067],
      ["معلم", 0.017],
      ["مربی", 0.07],
      ["مددکار", -0.032],
    ],
  },
  adj: {
    name: "adjectives (slide 34)",
    groups: ["agency traits", "warmth & sensitivity"],
    words: [
      ["قوی", 0.046],
      ["شجاع", 0.043],
      ["مستقل", 0.012],
      ["منطقی", 0.013],
      ["مهربان", 0.025],
      ["دلسوز", 0.049],
      ["حساس", -0.009],
      ["عاطفی", -0.041],
    ],
  },
};
const GAP = 0.0343;
const SPREAD = 0.0425;
const ALPHA = 0.05;
const PERMS = 4000;

function combos(n, k) {
  const out = [];
  const rec = (s, cur) => {
    if (cur.length === k) {
      out.push(cur.slice());
      return;
    }
    for (let i = s; i < n; i++) {
      cur.push(i);
      rec(i + 1, cur);
      cur.pop();
    }
  };
  rec(0, []);
  return out;
}
// exact test: every way to split the 8 words into two groups of 4 (70 splits)
function exactSplits(scores, inX) {
  const n = scores.length;
  const k = inX.filter(Boolean).length;
  if (k === 0 || k === n) return { obs: 0, diffs: [], p: 1 };
  const mX = mean(scores.filter((s, i) => inX[i]));
  const mY = mean(scores.filter((s, i) => !inX[i]));
  const obs = mX - mY;
  const diffs = combos(n, k).map((c) => mean(c.map((i) => scores[i])) - mean(scores.filter((s, i) => c.indexOf(i) < 0)));
  const p = diffs.filter((d) => Math.abs(d) >= Math.abs(obs) - 1e-12).length / diffs.length;
  return { obs, diffs, p };
}
function invNorm(p) {
  const t = Math.sqrt(-2 * Math.log(Math.min(p, 1 - p)));
  const x = t - (2.515517 + 0.802853 * t + 0.010328 * t * t) / (1 + 1.432788 * t + 0.189269 * t * t + 0.001308 * t * t * t);
  return p < 0.5 ? -x : x;
}
// hypothetical: n words per group with the slide's gap and within-group spread, random-regrouping test
function bigGroupTest(n) {
  const z = [];
  for (let i = 0; i < n; i++) z.push(invNorm((i + 0.5) / n));
  const all = z.map((x) => GAP / 2 + SPREAD * x).concat(z.map((x) => -GAP / 2 + SPREAD * x));
  const tot = all.reduce((a, b) => a + b, 0);
  const rng = mulberry32(1000 + n);
  const idx = all.map((_, i) => i);
  const diffs = [];
  let hit = 0;
  for (let r = 0; r < PERMS; r++) {
    for (let i = idx.length - 1; i > 0; i--) {
      const j = Math.floor(rng() * (i + 1));
      const t = idx[i];
      idx[i] = idx[j];
      idx[j] = t;
    }
    let sa = 0;
    for (let i = 0; i < n; i++) sa += all[idx[i]];
    const d = sa / n - (tot - sa) / n;
    diffs.push(d);
    if (Math.abs(d) >= GAP - 1e-12) hit++;
  }
  return { diffs, p: hit / PERMS, obs: GAP };
}
// @sim-end

function Hist({ diffs, obs, range }) {
  const W = 600;
  const H = 170;
  const B = 31;
  const lo = range[0];
  const hi = range[1];
  const counts = new Array(B).fill(0);
  for (const d of diffs) counts[clamp(Math.floor(((d - lo) / (hi - lo)) * B), 0, B - 1)]++;
  const mx = Math.max(1, ...counts);
  const X = (v) => 30 + ((clamp(v, lo, hi) - lo) / (hi - lo)) * (W - 60);
  const bw = (W - 60) / B;
  return (
    <svg viewBox={"0 0 " + W + " " + H} style={{ width: "100%", display: "block" }} role="img" aria-label="Gaps from random regroupings, with the observed gap marked">
      {counts.map((c, i) => {
        const x0 = 30 + i * bw;
        const mid = lo + ((i + 0.5) / B) * (hi - lo);
        const far = Math.abs(mid) >= Math.abs(obs);
        return <rect key={i} x={x0 + 1} y={H - 28 - (c / mx) * (H - 50)} width={bw - 2} height={(c / mx) * (H - 50)} fill={far ? THEME.highlight : THEME.accent} opacity={far ? 0.95 : 0.6} />;
      })}
      <line x1={30} x2={W - 30} y1={H - 28} y2={H - 28} stroke={THEME.grid} />
      {[obs, -obs].map((o, i) => (
        <g key={i}>
          <line x1={X(o)} x2={X(o)} y1={14} y2={H - 24} stroke={THEME.text} strokeDasharray={i ? "3 4" : "none"} strokeWidth={1.6} />
          <SvgLabel x={X(o)} y={12} anchor="middle" size={11}>
            {i ? "−observed" : "observed " + sgn(o, 3)}
          </SvgLabel>
        </g>
      ))}
      {[lo, 0, hi].map((t) => (
        <SvgLabel key={t} x={X(t)} y={H - 10} anchor="middle" color={THEME.muted} size={10.5}>
          {sgn(t, 2)}
        </SvgLabel>
      ))}
    </svg>
  );
}
function plane2(w, man, woman) {
  return cosine(w, man) - cosine(w, woman);
}
// the s = 0 line: directions at equal angles to man and woman
function bisector(P) {
  const a = P(0, 0);
  const b = P(3, 3);
  return (
    <g>
      <line x1={a[0]} y1={a[1]} x2={b[0]} y2={b[1]} stroke={THEME.text} strokeDasharray="6 5" opacity={0.7} />
      <SvgLabel x={b[0] - 6} y={b[1] + 16} anchor="end" color={THEME.muted} size={11}>
        s = 0
      </SvgLabel>
    </g>
  );
}
const MAN = [2.2, 1.0];
const WOMAN = [1.0, 2.2];

export default function BiasLesson({ initialScreen }) {
  const [screen, setScreen] = useState(initialScreen || 0);
  const [nurse, setNurse] = useState([0.9, 1.9]);
  const [eng, setEng] = useState([2.0, 1.1]);
  const [ds, setDs] = useState("occ");
  const [inX, setInX] = useState({ occ: [true, true, true, true, false, false, false, false], adj: [true, true, true, true, false, false, false, false] });
  const [mine, setMine] = useState(false);
  const [n, setN] = useState(4);

  const onDrag = useCallback((id, x, y) => {
    if (id === "nurse") setNurse([x, y]);
    if (id === "eng") setEng([x, y]);
  }, []);
  const flip = useCallback(
    (i) =>
      setInX((S) => {
        const cur = S[ds].slice();
        cur[i] = !cur[i];
        return Object.assign({}, S, { [ds]: cur });
      }),
    [ds]
  );
  const resetGroups = useCallback(() => setInX((S) => Object.assign({}, S, { [ds]: [true, true, true, true, false, false, false, false] })), [ds]);

  const D = DATA[ds];
  const scores = D.words.map((w) => w[1]);
  const grp = inX[ds];
  const big = useMemo(() => bigGroupTest(n), [n]);
  const prev = useMemo(() => (n > 4 ? bigGroupTest(n - 1) : null), [n]);

  let question, stage, side;
  if (screen === 0) {
    const sN = plane2(nurse, MAN, WOMAN);
    const sE = plane2(eng, MAN, WOMAN);
    question = (
      <>
        <b>Where we are:</b> lessons 1–6 showed that embeddings learn whatever the training text says about which words go together. Training text also carries patterns from society (slide 31). <b>The puzzle:</b> one way to measure an association is a word's score s(w) = cos(w, man) − cos(w, woman). In this sketch, drag <i>nurse</i> and <i>engineer</i>. Where must a word sit to score 0? And does a score of +0.2 show that anyone believes engineers are men?
      </>
    );
    stage = (
      <WordPlane
        xr={[-0.5, 3]}
        yr={[-0.5, 3]}
        ticks={0.5}
        points={[
          { id: "man", label: "man (male term)", x: MAN[0], y: MAN[1], color: THEME.accent, shape: "circle", arrow: true },
          { id: "woman", label: "woman (female term)", x: WOMAN[0], y: WOMAN[1], color: THEME.highlight, shape: "square", arrow: true },
          { id: "nurse", label: "nurse  s = " + sgn(sN, 2), x: nurse[0], y: nurse[1], color: THEME.third, shape: "diamond", drag: true, arrow: true },
          { id: "eng", label: "engineer  s = " + sgn(sE, 2), x: eng[0], y: eng[1], color: THEME.third, shape: "triangle", drag: true, arrow: true },
        ]}
        extra={bisector}
        onDrag={onDrag}
        ariaLabel="Illustrative 2-D sketch of gender association"
      />
    );
    side = (
      <>
        <Panel>
          <Readout
            rows={[
              ["s(nurse)", sgn(sN, 3), sN > 0 ? THEME.accent : THEME.highlight],
              ["s(engineer)", sgn(sE, 3), sE > 0 ? THEME.accent : THEME.highlight],
              ["gap s(engineer) − s(nurse)", sgn(sE - sN, 3)],
            ]}
          />
        </Panel>
        <Note>This sketch is illustrative, not measured (slide 31). s depends only on direction: a word scores 0 anywhere on the dashed line that splits the angle between man and woman.</Note>
      </>
    );
  } else if (screen === 1) {
    const mX = mean(scores.filter((s, i) => grp[i]));
    const mY = mean(scores.filter((s, i) => !grp[i]));
    question = (
      <>
        <b>Claim to test:</b> WEAT compares two groups of target words. Each word's score is its mean cosine to four male terms minus its mean cosine to four female terms, and the test statistic is the gap between the two groups' mean scores (slide 32). These are real fastText Persian Wikipedia scores (slides 33–34). Which words carry the gap? Click words to move them between groups. Does the gap depend on a single word?
      </>
    );
    stage = (
      <Bars
        rows={D.words.map((w, i) => ({ label: w[0], fa: true, value: w[1], color: grp[i] ? THEME.accent : THEME.highlight, mark: grp[i] ? "● " + D.groups[0] : "■ " + D.groups[1] }))}
        range={[-0.1, 0.1]}
        digits={3}
        labelW={100}
        ariaLabel="per-word association scores"
        valueFmt={(v) => sgn(v, 3)}
      />
    );
    side = (
      <>
        <Panel>
          <Pills
            options={[
              ["occ", "occupations"],
              ["adj", "adjectives"],
            ]}
            value={ds}
            onPick={setDs}
          />
          <Label>click to move a word to the other group</Label>
          <div dir="rtl" style={{ display: "flex", flexWrap: "wrap", gap: 5, marginBottom: 8 }}>
            {D.words.map((w, i) => (
              <Chip key={i} value={i} on={grp[i]} onPick={flip} mark={grp[i] ? "●" : "■"}>
                {w[0]}
              </Chip>
            ))}
          </div>
          <Readout
            rows={[
              ["● " + D.groups[0] + " mean", sgn(mX, 3)],
              ["■ " + D.groups[1] + " mean", sgn(mY, 3)],
              ["gap", isFinite(mX - mY) ? sgn(mX - mY, 3) : "—"],
            ]}
          />
          <Btn onClick={resetGroups}>slide's grouping</Btn>
        </Panel>
        <Note>Male terms: مرد، پسر، پدر، برادر. Female terms: زن، دختر، مادر، خواهر. Note مربی (+0.070) and دلسوز (+0.049): they sit in the "care" and "warmth" groups but score male-associated, which shrinks the gap.</Note>
      </>
    );
  } else if (screen === 2) {
    const use = mine ? grp : [true, true, true, true, false, false, false, false];
    const T = exactSplits(scores, use);
    question = (
      <>
        <b>What you found:</b> the slide's grouping gives a gap of {sgn(exactSplits(scores, [true, true, true, true, false, false, false, false]).obs, 3)}, but single words move it a lot. Is that gap more than chance? The permutation test splits the same 8 scores into two groups of 4 in all 70 possible ways and counts how often the gap is at least as large. That fraction is the p-value.
      </>
    );
    stage = (
      <>
        <Hist diffs={T.diffs} obs={T.obs} range={[-0.07, 0.07]} />
        <Eq size={12.5}>
          s(w, A, B) = mean over a ∈ A of cos(w, a) − mean over b ∈ B of cos(w, b)
          <br />
          gap = mean over x ∈ X of s(x) − mean over y ∈ Y of s(y)
          <br />
          p = fraction of the 70 regroupings with |gap| ≥ observed = {fmt(T.p, 2)}
        </Eq>
      </>
    );
    side = (
      <>
        <Panel>
          <Pills
            options={[
              ["occ", "occupations"],
              ["adj", "adjectives"],
            ]}
            value={ds}
            onPick={setDs}
          />
          <Toggle label="use my grouping from Explore" value={mine} onChange={setMine} />
          <Readout
            rows={[
              ["observed gap", sgn(T.obs, 3)],
              ["regroupings as extreme (apricot)", Math.round(T.p * T.diffs.length) + " / " + T.diffs.length],
              ["two-sided p", fmt(T.p, 2)],
            ]}
          />
        </Panel>
        <Note>
          Slides 32–34 report p = 0.34 for occupations and 0.37 for adjectives: exploratory, small sets, not evidence of an association. An association is not proof of anyone's belief, and harm depends on the application (slide 31). Static vectors are still fast and useful when the trade-off fits the task (slides 35–38).
        </Note>
      </>
    );
  } else {
    const ok = big.p < ALPHA;
    const okMin = ok && (n === 4 || (prev && prev.p >= ALPHA));
    question = (
      <>
        The slide's occupation test had 4 words per group, a gap of +0.034 and p = 0.34. Suppose you collected more occupation words per group, with the same gap and the same spread of scores within each group. Find the smallest group size at which the permutation test gives p &lt; 0.05.
      </>
    );
    stage = (
      <>
        <Hist diffs={big.diffs} obs={GAP} range={[-0.07, 0.07]} />
        <Note>Bars: the gap from {PERMS.toLocaleString("en-US")} random regroupings of the {2 * n} hypothetical words. Apricot bars are at least as extreme as the observed +0.034.</Note>
      </>
    );
    side = (
      <>
        <Panel edge={okMin ? THEME.ok : THEME.grid}>
          <Slider id="l7-n" label="words per group" value={n} min={4} max={30} stepSize={1} onChange={setN} display={"n = " + n} />
          <Readout
            rows={[
              ["two-sided p at n = " + n, fmt(big.p, 3), ok ? THEME.ok : THEME.text],
              ["regroupings as extreme", Math.round(big.p * PERMS) + " / " + PERMS],
            ]}
          />
          <Goal met={ok}>p &lt; 0.05</Goal>
          <Goal met={okMin}>… and one word fewer per group would not be enough</Goal>
          <Hint
            hints={[
              "The observed gap stays fixed at +0.034. What happens to the spread of the random-regrouping gaps as each group gets more words?",
              "The spread of a mean shrinks like 1/√n. The gap needs to sit about two spreads out. Look between n = 10 and n = 20, and check the size just below your answer.",
            ]}
          />
        </Panel>
        <Note>Hypothetical scores, quantile-spaced with the slide's gap (+0.034) and within-group spread (0.042). With 4 words per group even a real gap would rarely reach significance. That is why the slide calls its result exploratory.</Note>
      </>
    );
  }

  return <LessonFrame screen={screen} onScreen={setScreen} title="Vectors inherit associations" question={question} stage={stage} side={side} />;
}
