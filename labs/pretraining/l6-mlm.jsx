import { useState, useCallback, useMemo } from "react";
import { THEME, MONO } from "./common";

// Lesson 6 - BERT's pretraining: masked tokens and next sentences (Pretraining, slides 61-75).
// Aha: an encoder sees both sides, so its game is "recover a hidden token from its context" (MLM), not
// "predict the next one". 15% of positions become targets; of those 80% show [MASK], 10% a random token and
// 10% the original. Because some targets look untouched and some visible tokens are wrong, the model cannot
// tell which positions will be scored and must build a good vector for every token. NSP adds a second head on
// [CLS]; both losses train one encoder in the same step.

// @sim-start
// the share of input positions in each state, for selection rate s and split (mask, random, keep)
function maskStats(s, pm, pr) {
  const pk = 1 - pm - pr;
  const maskShown = s * pm;
  const randomShown = s * pr;
  const keptTarget = s * pk;
  const looksNormal = 1 - maskShown;
  return {
    maskShown,
    randomShown,
    keptTarget,
    plain: 1 - s,
    // of the visible ordinary tokens: how many are secretly targets, how many are wrong
    hiddenTarget: looksNormal > 0 ? (keptTarget + randomShown) / looksNormal : 0,
    wrong: looksNormal > 0 ? randomShown / looksNormal : 0,
  };
}
// @sim-end

const CANDS = ["drink", "like", "spill", "love", "eat", "chase", "sleep"];
// toy "ideal model" guesses for: the cats ___ milk
const VIEWS = {
  none: [0.06, 0.08, 0.03, 0.06, 0.08, 0.06, 0.08],
  left: [0.08, 0.1, 0.02, 0.06, 0.12, 0.15, 0.25],
  right: [0.35, 0.1, 0.15, 0.05, 0.03, 0.01, 0.0],
  both: [0.6, 0.12, 0.08, 0.06, 0.03, 0.02, 0.0],
};
function effChoices(ps) {
  const rest = Math.max(0, 1 - ps.reduce((a, b) => a + b, 0));
  // "other" is spread over 20 more words, so it counts as real uncertainty
  let H = 0;
  for (const p of ps) if (p > 0) H -= p * Math.log(p);
  if (rest > 0) H -= rest * Math.log(rest / 20);
  return Math.exp(H);
}
const SENT = ["the", "old", "dog", "slept", "near", "the", "warm", "fire", "while", "rain", "fell", "on", "the", "roof", "all", "night"];
const RANDOM_WORDS = ["table", "blue", "seven", "jump", "river", "paper"];
function drawMask(seed, s, pm, pr) {
  const rng = mulberry32(seed);
  return SENT.map((w, i) => {
    if (rng() >= s) return { w, kind: "plain", shown: w };
    const u = rng();
    if (u < pm) return { w, kind: "mask", shown: "[MASK]" };
    if (u < pm + pr) return { w, kind: "random", shown: RANDOM_WORDS[Math.floor(rng() * RANDOM_WORDS.length)] };
    return { w, kind: "keep", shown: w };
  });
}
function simulate(n, s, pm, pr) {
  const rng = mulberry32(7);
  const c = { mask: 0, random: 0, keep: 0, plain: 0 };
  for (let i = 0; i < n; i++) {
    if (rng() >= s) c.plain++;
    else {
      const u = rng();
      if (u < pm) c.mask++;
      else if (u < pm + pr) c.random++;
      else c.keep++;
    }
  }
  return c;
}
const KIND_MARK = { mask: "M", random: "R", keep: "K", plain: "" };
function TokenBox({ t, reveal }) {
  const isT = t.kind !== "plain";
  return (
    <div style={{ border: "1px " + (reveal && isT ? "solid " + THEME.highlight : "solid " + THEME.grid), borderRadius: 4, padding: "3px 7px", background: t.kind === "mask" ? rgba(THEME.accent, 0.18) : THEME.bg, textAlign: "center" }}>
      <div style={{ fontFamily: MONO, fontSize: 13.5, color: t.kind === "random" && reveal ? THEME.warn : THEME.text }}>{t.shown}</div>
      {reveal ? <div style={{ fontFamily: MONO, fontSize: 10.5, color: isT ? THEME.highlight : THEME.muted }}>{isT ? KIND_MARK[t.kind] + " → " + t.w : "·"}</div> : null}
    </div>
  );
}
// the challenge input: original "the dog chased the ball [SEP] it ran very fast"
const CH = [
  { shown: "[CLS]", special: true },
  { shown: "the" },
  { shown: "dog" },
  { shown: "[MASK]", target: "chased" },
  { shown: "the" },
  { shown: "ball", target: "ball" },
  { shown: "[SEP]", special: true },
  { shown: "it" },
  { shown: "ran" },
  { shown: "table", target: "very" },
  { shown: "fast" },
  { shown: "[SEP]", special: true },
];
function MarkToken({ i, t, on, onPick, reveal }) {
  const click = useCallback(() => onPick(i), [onPick, i]);
  return (
    <button
      onClick={click}
      aria-pressed={on}
      style={{
        fontFamily: MONO,
        fontSize: 13.5,
        padding: "4px 8px",
        borderRadius: 4,
        cursor: "pointer",
        border: "1px " + (on ? "solid " + THEME.highlight : "dashed " + THEME.grid),
        background: on ? rgba(THEME.highlight, 0.16) : THEME.bg,
        color: t.special ? THEME.muted : THEME.text,
        textAlign: "center",
      }}
    >
      <div>{t.shown}</div>
      <div style={{ fontSize: 10.5, color: reveal && t.target ? THEME.highlight : THEME.muted }}>{on ? "◆ scored?" : "◇"}{reveal && t.target ? " · target: " + t.target : ""}</div>
    </button>
  );
}

export default function MlmLesson({ initialScreen }) {
  const [screen, setScreen] = useState(initialScreen || 0);
  const [left, setLeft] = useState(true);
  const [right, setRight] = useState(false);
  const [seen, setSeen] = useState({ left: true });
  const [sel, setSel] = useState(0.15);
  const [pm, setPm] = useState(0.8);
  const [pr, setPr] = useState(0.1);
  const [seed, setSeed] = useState(3);
  const [simN, setSimN] = useState(0);
  const [pMlm, setPMlm] = useState(0.2);
  const [pNsp, setPNsp] = useState(0.8);
  const [isNext, setIsNext] = useState(true);
  const [marks, setMarks] = useState(CH.map(() => false));

  const view = left && right ? "both" : left ? "left" : right ? "right" : "none";
  const onLeft = useCallback((v) => {
    setLeft(v);
    setSeen((S) => Object.assign({}, S, { [v && right ? "both" : v ? "left" : right ? "right" : "none"]: true }));
  }, [right]);
  const onRight = useCallback((v) => {
    setRight(v);
    setSeen((S) => Object.assign({}, S, { [left && v ? "both" : left ? "left" : v ? "right" : "none"]: true }));
  }, [left]);
  const onPm = useCallback((v) => {
    setPm(v);
    setPr((r) => Math.min(r, 1 - v));
  }, []);
  const onPr = useCallback((v) => setPr(Math.min(v, 1 - pm)), [pm]);
  const redraw = useCallback(() => setSeed((s) => s + 1), []);
  const runSim = useCallback(() => setSimN(100000), []);
  const flip = useCallback((i) => setMarks((M) => M.map((v, j) => (j === i ? !v : v))), []);
  const clearMarks = useCallback(() => setMarks(CH.map(() => false)), []);
  const drawn = useMemo(() => drawMask(seed, sel, pm, pr), [seed, sel, pm, pr]);
  const sim = useMemo(() => (simN ? simulate(simN, sel, pm, pr) : null), [simN, sel, pm, pr]);

  let question, stage, side;
  if (screen === 0) {
    const ps = VIEWS[view];
    const other = Math.max(0, 1 - ps.reduce((a, b) => a + b, 0));
    question = (
      <>
        <b>Where we are:</b> lesson 5's model reads a fixed window of past words. Transformers can read the whole sentence, and BERT keeps only the Transformer's encoder, where every token may attend to every other (slides 61–65). <b>The puzzle:</b> a next-token model guesses a word from its left side only. A masked model hides a word and may use both sides (slide 22). Show or hide each side of <span style={{ fontFamily: MONO }}>the cats ___ milk</span>. How many plausible words are left in each case?
      </>
    );
    stage = (
      <>
        <Panel>
          <div style={{ fontFamily: MONO, fontSize: 18, textAlign: "center", letterSpacing: 0.5, padding: "6px 0" }}>
            <span style={{ color: left ? THEME.text : THEME.grid }}>the cats</span> <span style={{ color: THEME.highlight }}>[MASK]</span> <span style={{ color: right ? THEME.text : THEME.grid }}>milk</span>
          </div>
        </Panel>
        <Bars rows={CANDS.map((w, i) => ({ label: w, value: ps[i], color: w === "drink" ? THEME.highlight : THEME.accent })).concat([{ label: "20 others", value: other, color: THEME.grid }])} range={[0, 1]} digits={2} labelW={90} rowH={21} ariaLabel="guesses for the hidden word" />
      </>
    );
    side = (
      <>
        <Panel>
          <Toggle label="left context: the cats" value={left} onChange={onLeft} />
          <Toggle label="right context: milk" value={right} onChange={onRight} />
          <Readout
            rows={[
              ["P(drink)", fmt(ps[0], 2)],
              ["effective number of choices", fmt(effChoices(ps), 1), THEME.highlight],
            ]}
          />
          <Goal met={!!seen.left && !!seen.both}>compare left-only (what a next-token model gets) with both sides</Goal>
        </Panel>
        <Note>"Effective number of choices" is the perplexity of the guess (lesson 4). Toy probabilities from an imagined ideal model, not a measured BERT.</Note>
      </>
    );
  } else if (screen === 1) {
    const st = maskStats(sel, pm, pr);
    const nTarget = drawn.filter((t) => t.kind !== "plain").length;
    question = (
      <>
        <b>Claim to test:</b> BERT selects 15% of positions as targets, and the target is always the original token. Of the selected, 80% are replaced by [MASK], 10% by a random token and 10% left unchanged (slide 70). Draw maskings and simulate many positions. Then set the split to 100% [MASK]: what could the model then assume about every token it can read?
      </>
    );
    stage = (
      <>
        <Panel>
          <Label>what BERT reads (one draw) · under each: M / R / K = masked, random, kept → original target</Label>
          <div style={{ display: "flex", flexWrap: "wrap", gap: 5 }}>
            {drawn.map((t, i) => (
              <TokenBox key={i} t={t} reveal />
            ))}
          </div>
          <div style={{ display: "flex", gap: 6, marginTop: 8 }}>
            <Btn onClick={redraw}>draw another masking</Btn>
            <Btn onClick={runSim} strong>
              simulate 100,000 positions
            </Btn>
          </div>
        </Panel>
        <Bars
          rows={[
            { label: "[MASK] shown", value: st.maskShown, color: THEME.accent, mark: sim ? "sim " + fmt((100 * sim.mask) / simN, 2) + "%" : "" },
            { label: "random token", value: st.randomShown, color: THEME.warn, mark: sim ? "sim " + fmt((100 * sim.random) / simN, 2) + "%" : "" },
            { label: "kept, still a target", value: st.keptTarget, color: THEME.highlight, mark: sim ? "sim " + fmt((100 * sim.keep) / simN, 2) + "%" : "" },
            { label: "not a target", value: st.plain, color: THEME.grid, mark: sim ? "sim " + fmt((100 * sim.plain) / simN, 2) + "%" : "" },
          ]}
          range={[0, 1]}
          valueFmt={(v) => fmt(v * 100, 1) + "%"}
          labelW={150}
          ariaLabel="share of positions in each state"
        />
      </>
    );
    side = (
      <>
        <Panel>
          <Slider id="m6-s" label="selected as targets" value={sel} min={0.05} max={0.5} stepSize={0.01} onChange={setSel} display={fmt(sel * 100, 0) + "%"} />
          <Slider id="m6-m" label="of those: [MASK]" value={pm} min={0} max={1} stepSize={0.01} onChange={onPm} display={fmt(pm * 100, 0) + "%"} />
          <Slider id="m6-r" label="of those: random token" value={pr} min={0} max={1} stepSize={0.01} onChange={onPr} display={fmt(pr * 100, 0) + "%"} />
          <Readout
            rows={[
              ["of those: kept", fmt((1 - pm - pr) * 100, 0) + "%"],
              ["targets in this draw", nTarget + " / " + SENT.length],
              ["targets per 512-token input", fmt(sel * 512, 0)],
              ["a readable token is secretly a target", fmt(st.hiddenTarget * 100, 2) + "%", THEME.highlight],
              ["a readable token is wrong", fmt(st.wrong * 100, 2) + "%", THEME.warn],
            ]}
          />
        </Panel>
        <Note>Only targets contribute to the MLM loss (slide 69), so about 15% of positions teach the model anything per pass. With 100% [MASK], every readable token is guaranteed correct and never scored. The model could then ignore them, and fine-tuning text has no [MASK] at all.</Note>
      </>
    );
  } else if (screen === 2) {
    const lM = -Math.log(pMlm);
    const lN = -Math.log(pNsp);
    question = (
      <>
        <b>What you found:</b> MLM picks targets the model cannot spot in advance, so every token's vector has to stay informative. Original BERT adds a second game, next sentence prediction: is B the real continuation of A (50%) or a random sentence (50%)? Its head reads the [CLS] vector (slides 71–72). Both losses are added and train the one encoder in the same step (slides 73–75).
      </>
    );
    stage = (
      <>
        <svg viewBox="0 0 600 230" style={{ width: "100%", display: "block" }} role="img" aria-label="one encoder, two heads">
          {["[CLS]", "the", "cat", "[MASK]", "[SEP]", isNext ? "it" : "the", isNext ? "is" : "train", isNext ? "sleepy" : "left", "[SEP]"].map((w, i) => (
            <g key={i}>
              <rect x={14 + i * 64} y={190} width={58} height={28} rx={4} fill={w === "[MASK]" ? rgba(THEME.accent, 0.2) : THEME.bg} stroke={THEME.grid} />
              <SvgLabel x={43 + i * 64} y={209} anchor="middle" size={11.5}>
                {w}
              </SvgLabel>
              <line x1={43 + i * 64} x2={43 + i * 64} y1={190} y2={160} stroke={THEME.grid} />
            </g>
          ))}
          <rect x={14} y={110} width={570} height={50} rx={6} fill={THEME.surface} stroke={THEME.accent} strokeWidth={1.5} />
          <SvgLabel x={300} y={140} anchor="middle">
            one shared BERT encoder · one forward pass
          </SvgLabel>
          <Arrow x1={43} y1={110} x2={43} y2={66} color={THEME.third} />
          <rect x={10} y={30} width={150} height={34} rx={4} fill={THEME.bg} stroke={THEME.third} />
          <SvgLabel x={85} y={52} anchor="middle" size={11.5} color={THEME.third}>
            {"NSP head d → 2"}
          </SvgLabel>
          <SvgLabel x={85} y={20} anchor="middle" size={11.5}>
            {"P(" + (isNext ? "IsNext" : "NotNext") + ") = " + fmt(pNsp, 2)}
          </SvgLabel>
          <Arrow x1={235} y1={110} x2={235} y2={66} color={THEME.highlight} />
          <rect x={190} y={30} width={150} height={34} rx={4} fill={THEME.bg} stroke={THEME.highlight} />
          <SvgLabel x={265} y={52} anchor="middle" size={11.5} color={THEME.highlight}>
            {"MLM head d → |V|"}
          </SvgLabel>
          <SvgLabel x={265} y={20} anchor="middle" size={11.5}>
            {"P(sleeps) = " + fmt(pMlm, 2)}
          </SvgLabel>
          <SvgLabel x={370} y={52} size={12}>
            {"loss = " + fmt(lM, 3) + " + " + fmt(lN, 3) + " = " + fmt(lM + lN, 3)}
          </SvgLabel>
        </svg>
        <Eq size={13}>
          L = L_MLM + L_NSP
          <br />
          L_MLM = −(1/|T|) Σ_(i∈T) ln p(xᵢ | corrupted input)
          <br />
          L_NSP = −ln p(label | h_[CLS])
        </Eq>
      </>
    );
    side = (
      <>
        <Panel>
          <Toggle label="B is the real next sentence (IsNext)" value={isNext} onChange={setIsNext} />
          <Slider id="m6-pm" label="MLM head: P(sleeps) at [MASK]" value={pMlm} min={0.01} max={1} stepSize={0.01} onChange={setPMlm} />
          <Slider id="m6-pn" label={"NSP head: P(" + (isNext ? "IsNext" : "NotNext") + ") at [CLS]"} value={pNsp} min={0.01} max={1} stepSize={0.01} onChange={setPNsp} />
          <Readout
            rows={[
              ["MLM cross-entropy", fmt(lM, 3)],
              ["NSP cross-entropy", fmt(lN, 3)],
              ["total loss", fmt(lM + lN, 3), THEME.highlight],
            ]}
          />
        </Panel>
        <Note>Slide 75: −ln(0.2) ≈ 1.609 and −ln(0.8) ≈ 0.223. Backpropagation sends both gradients into the same encoder, then one optimizer step updates the encoder and both heads. Targets come from the text and its document order, so no human labels are needed.</Note>
      </>
    );
  } else {
    const ordinary = CH.map((t, i) => i).filter((i) => !CH[i].special);
    const g1 = marks[3];
    const g2 = marks[9];
    const g3 = ordinary.every((i) => marks[i]);
    const g4 = CH.every((t, i) => !t.special || !marks[i]);
    const all = g1 && g2 && g3 && g4;
    question = (
      <>
        You are the encoder. This input came out of BERT's masking recipe, built from a real two-sentence document. The MLM loss will be computed at some positions, but nobody tells you which. Mark every position whose output vector you must make good at predicting its original token. Which positions can you safely ignore?
      </>
    );
    stage = (
      <>
        <Panel>
          <div style={{ display: "flex", flexWrap: "wrap", gap: 6 }}>
            {CH.map((t, i) => (
              <MarkToken key={i} i={i} t={t} on={marks[i]} onPick={flip} reveal={all} />
            ))}
          </div>
          <div style={{ marginTop: 8 }}>
            <Btn onClick={clearMarks}>clear marks</Btn>
          </div>
        </Panel>
        {all ? <Note>Revealed: the real targets were chased ([MASK]), very (replaced by "table") and ball, which was selected and left unchanged. Nothing in the input told "ball" apart from "dog".</Note> : null}
      </>
    );
    side = (
      <>
        <Panel edge={all ? THEME.ok : THEME.grid}>
          <Goal met={g1}>the [MASK] position</Goal>
          <Goal met={g2}>the token that does not fit</Goal>
          <Goal met={g3}>every position that could be a hidden target</Goal>
          <Goal met={g4}>no position that can never be an MLM target</Goal>
          <Hint hints={["10% of the targets were left unchanged. Could you tell such a token apart from one that was never selected?", "Any ordinary token may be a target, so every one needs a good contextual vector. [CLS] and [SEP] are added by the input format and are never MLM targets."]} />
        </Panel>
        <Note>Devlin et al. (2019, App. A.1) give this reason for the 80/10/10 split: the encoder does not know which words it will be asked to predict, or which were replaced, so it keeps a contextual representation of every input token.</Note>
      </>
    );
  }

  return <LessonFrame screen={screen} onScreen={setScreen} title="BERT's pretraining: masked tokens and next sentences" question={question} stage={stage} side={side} />;
}
