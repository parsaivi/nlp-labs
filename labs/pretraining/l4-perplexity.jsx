import { useState, useCallback } from "react";
import { THEME, MONO } from "./common";

// Lesson 4 - Which model predicts better? (Pretraining, slides 40-43).
// Aha: a sequence probability shrinks with length, so compare models per token: cross-entropy is the
// average surprise -ln p at the actual next token, and perplexity exp(H) is the effective number of equally
// likely choices. The worst-predicted token dominates, and the comparison is only fair on the same text
// with the same tokenizer.

// @sim-start
const MODEL_A = [0.625, 0.6, 0.4, 1];
const MODEL_B = [0.5, 0.5, 0.25, 0.8];
const TARGETS = ["the", "dog", "runs", "[EOS]"];
function xent(ps) {
  let s = 0;
  for (const p of ps) s -= Math.log(p);
  return s / ps.length;
}
function ppl(ps) {
  return Math.exp(xent(ps));
}
// @sim-end

function SurpriseBars({ ps, labels, color }) {
  return (
    <Bars
      rows={ps.map((p, i) => ({ label: labels[i], value: -Math.log(p), color: color || THEME.accent, mark: "p = " + fmt(p, 3) }))}
      range={[0, 3.2]}
      digits={3}
      labelW={70}
      ariaLabel="surprise −ln p for each target token"
    />
  );
}

export default function PerplexityLesson({ initialScreen }) {
  const [screen, setScreen] = useState(initialScreen || 0);
  const [n, setN] = useState(4);
  const [p, setP] = useState(0.5);
  const [hitA, setHitA] = useState(false);
  const [hitC, setHitC] = useState(false);
  const [ps, setPs] = useState(MODEL_A.slice());
  const [nT, setNT] = useState(4);
  const [preset, setPreset] = useState("A");
  const [base, setBase] = useState("e");
  const [tok, setTok] = useState("word");
  const [bs, setBs] = useState(MODEL_B.slice());

  const onN = useCallback((v) => {
    setN(v);
  }, []);
  const check = useCallback(
    (nn, pp) => {
      const P = Math.pow(pp, nn);
      if (nn === 4 && Math.abs(P - 0.15) < 0.006) setHitA(true);
      if (nn === 12 && Math.abs(P - 0.001) < 0.00012) setHitC(true);
    },
    []
  );
  const onNc = useCallback(
    (v) => {
      onN(v);
      check(v, p);
    },
    [onN, check, p]
  );
  const onPc = useCallback(
    (v) => {
      setP(v);
      check(n, v);
    },
    [check, n]
  );
  const setI = useCallback((k, v) => {
    setPs((P) => P.map((x, i) => (i === k ? v : x)));
    setPreset("");
  }, []);
  const s0 = useCallback((v) => setI(0, v), [setI]);
  const s1 = useCallback((v) => setI(1, v), [setI]);
  const s2 = useCallback((v) => setI(2, v), [setI]);
  const s3 = useCallback((v) => setI(3, v), [setI]);
  const pickPreset = useCallback((k) => {
    setPreset(k);
    if (k === "A") {
      setPs(MODEL_A.slice());
      setNT(4);
    } else if (k === "B") {
      setPs(MODEL_B.slice());
      setNT(4);
    } else if (k === "U") {
      setPs([0.25, 0.25, 0.25, 0.25]);
      setNT(4);
    } else {
      setPs([0.5, 0.25, 0.125, 1]);
      setNT(3);
    }
  }, []);
  const setB = useCallback((k, v) => setBs((P) => P.map((x, i) => (i === k ? v : x))), []);
  const b0 = useCallback((v) => setB(0, v), [setB]);
  const b1 = useCallback((v) => setB(1, v), [setB]);
  const b2 = useCallback((v) => setB(2, v), [setB]);
  const b3 = useCallback((v) => setB(3, v), [setB]);
  const resetB = useCallback(() => setBs(MODEL_B.slice()), []);

  let question, stage, side;
  if (screen === 0) {
    const P = Math.pow(p, n);
    const pts = [];
    for (let k = 1; k <= 40; k++) pts.push([k, Math.log10(Math.pow(p, k))]);
    question = (
      <>
        <b>Where we are:</b> lesson 3 scored sentences with a bigram model. To choose between two language models, score the same held-out text with both: higher probability is better (slide 40). <b>The puzzle:</b> model A gives a 4-token text probability 0.150. Model C gives a 12-token text probability 0.001. Which model predicts better? Use the sliders: a model that gives every token the same probability p, on a text of N tokens.
      </>
    );
    stage = (
      <>
        <LinePlot
          series={[{ label: "log₁₀ P(text) for this p", color: THEME.accent, shape: "circle", pts: pts.filter((q, i) => i % 3 === 0 || q[0] === n) }]}
          xr={[1, 40]}
          yr={[-12, 0]}
          xticks={[1, 4, 12, 20, 30, 40]}
          yticks={[[0, "1"], [-3, "10⁻³"], [-6, "10⁻⁶"], [-9, "10⁻⁹"], [-12, "10⁻¹²"]]}
          xlabel="text length N (tokens)"
          ylabel="P(text)"
          vline={n}
          H={250}
          legend={false}
          ariaLabel="sequence probability against text length"
        />
        <Note>Same model, same skill at every token: the text probability still falls by a factor p with every extra token.</Note>
      </>
    );
    side = (
      <>
        <Panel>
          <Slider id="p4-n" label="text length N" value={n} min={1} max={40} stepSize={1} onChange={onNc} display={"N = " + n} />
          <Slider id="p4-p" label="probability of each correct token p" value={p} min={0.05} max={1} stepSize={0.001} onChange={onPc} display={fmt(p, 3)} />
          <Readout
            rows={[
              ["P(text) = pᴺ", P < 1e-4 ? P.toExponential(2) : fmt(P, 4)],
              ["per-token probability p", fmt(p, 3)],
            ]}
          />
          <Goal met={hitA}>match model A: N = 4 and P(text) ≈ 0.150</Goal>
          <Goal met={hitC}>match model C: N = 12 and P(text) ≈ 0.001</Goal>
          {hitA && hitC ? <Note>Model A behaves like p ≈ 0.62 per token, model C like p ≈ 0.56. Model C's text was simply longer. Comparing raw text probabilities across lengths was unfair.</Note> : null}
        </Panel>
      </>
    );
  } else if (screen === 1) {
    const use = ps.slice(0, nT);
    const H = xent(use);
    question = (
      <>
        <b>Claim to test:</b> to compare per token, average the surprise −ln p at each actual next token (slide 41). Exponentiate it to get perplexity, read as "the model is as unsure as if it were choosing among PPL equally likely tokens" (slide 42). Load the presets, then drag one probability toward 0. Which token decides the score?
      </>
    );
    stage = (
      <>
        <Pills
          options={[
            ["A", "model A (slide 40)"],
            ["B", "model B (slide 40)"],
            ["U", "4 equal choices"],
            ["S", "1/2, 1/4, 1/8 (slide 43)"],
          ]}
          value={preset}
          onPick={pickPreset}
        />
        <SurpriseBars ps={use} labels={nT === 4 ? TARGETS : ["target 1", "target 2", "target 3"]} />
        <Eq size={13.5}>
          H = −(1/{nT}) · [{use.map((q) => "ln " + fmt(q, 3)).join(" + ")}] = {fmt(H, 3)} nats/token
          <br />
          PPL = e^H = {fmt(Math.exp(H), 3)}
        </Eq>
      </>
    );
    side = (
      <>
        <Panel>
          {[s0, s1, s2, s3].slice(0, nT).map((fn, i) => (
            <Slider key={i} id={"p4-s" + i} label={"P(" + (nT === 4 ? TARGETS[i] : "target " + (i + 1)) + " | prefix)"} value={ps[i]} min={0.001} max={1} stepSize={0.001} onChange={fn} display={fmt(ps[i], 3)} />
          ))}
          <Readout
            rows={[
              ["cross-entropy H", fmt(H, 3) + " nats"],
              ["perplexity", fmt(Math.exp(H), 3)],
            ]}
          />
        </Panel>
        <Note>Slide 41: A has H ≈ 0.474, B has H ≈ 0.749. Lower is better. A token with p = 0.001 alone adds 6.9 / N to H.</Note>
      </>
    );
  } else if (screen === 2) {
    const A = MODEL_A;
    const total = A.reduce((a, b) => a * b, 1);
    const nChar = "the dog runs".length + 1;
    const charP = Math.pow(total, 1 / nChar);
    const use = tok === "word" ? { n: 4, H: xent(A) } : { n: nChar, H: -Math.log(charP) };
    const unit = base === "e" ? 1 : 1 / Math.log(2);
    question = (
      <>
        <b>What you found:</b> the text probability depends on length, so we average per token. The average surprise is the cross-entropy H, and PPL = e^H = P(text)^(−1/N), the effective number of choices per step. One badly predicted token dominates. Now check what the number depends on: the log base, and the tokenizer.
      </>
    );
    stage = (
      <>
        <Eq size={14}>
          H = −(1/N) Σₜ ln P(xₜ | x_&lt;t) · PPL = e^H = (∏ₜ P(xₜ | x_&lt;t))^(−1/N)
          <br />
          slide 43: (1/2 · 1/4 · 1/8)^(−1/3) = 64^(1/3) = 4
        </Eq>
        <Panel>
          <Label>same model A, same text "the dog runs [EOS]", P(text) = 0.150</Label>
          <Readout
            rows={[
              ["tokens N", String(use.n)],
              ["H (" + (base === "e" ? "nats" : "bits") + " per token)", fmt(use.H * unit, 3)],
              ["PPL", fmt(Math.exp(use.H), 3), THEME.highlight],
            ]}
          />
        </Panel>
      </>
    );
    side = (
      <>
        <Panel>
          <Label>log base for H</Label>
          <Pills
            options={[
              ["e", "natural log (nats)"],
              ["2", "log₂ (bits)"],
            ]}
            value={base}
            onPick={setBase}
          />
          <Label>tokenizer</Label>
          <Pills
            options={[
              ["word", "word tokens (4)"],
              ["char", "characters (" + nChar + ")"],
            ]}
            value={tok}
            onPick={setTok}
          />
        </Panel>
        <Note>The base changes H but not PPL (2^bits = e^nats). The tokenizer changes PPL itself: the same text probability spread over 13 character targets looks like a far smaller "number of choices". So compare only with the same held-out text and tokenizer, and the same context length; a lower PPL also does not guarantee better task performance (slide 43).</Note>
      </>
    );
  } else {
    const pa = ppl(MODEL_A);
    const pb = ppl(bs);
    const changed = bs.filter((v, i) => Math.abs(v - MODEL_B[i]) > 1e-9).length;
    const g1 = pb <= pa + 1e-9;
    const g2 = changed <= 1;
    question = (
      <>
        Model B loses to model A on "the dog runs [EOS]" (PPL {fmt(ppl(MODEL_B), 3)} against {fmt(pa, 3)}). You can retrain B to improve its probability for <b>one</b> target token, and leave the other three as they are. Make B's perplexity at most A's. Which token can do it, and how high must its probability go?
      </>
    );
    stage = (
      <>
        <Label>model B, surprise per token</Label>
        <SurpriseBars ps={bs} labels={TARGETS} color={THEME.third} />
        <Label>model A, for comparison</Label>
        <SurpriseBars ps={MODEL_A} labels={TARGETS} />
      </>
    );
    side = (
      <>
        <Panel edge={g1 && g2 ? THEME.ok : THEME.grid}>
          {[b0, b1, b2, b3].map((fn, i) => (
            <Slider key={i} id={"p4-b" + i} label={"B: P(" + TARGETS[i] + ")" + (Math.abs(bs[i] - MODEL_B[i]) > 1e-9 ? " ✎" : "")} value={bs[i]} min={0.01} max={1} stepSize={0.005} onChange={fn} display={fmt(bs[i], 3)} />
          ))}
          <Readout
            rows={[
              ["PPL of B", fmt(pb, 3), g1 ? THEME.ok : THEME.text],
              ["PPL of A", fmt(pa, 3)],
              ["tokens changed", String(changed), g2 ? THEME.text : THEME.warn],
            ]}
          />
          <Goal met={g1}>PPL(B) ≤ PPL(A)</Goal>
          <Goal met={g2}>only one token changed</Goal>
          <Btn onClick={resetB}>reset B</Btn>
          <Hint hints={["Equal perplexity on the same 4 tokens means equal text probability. B's product is 0.05, A's is 0.15. By what factor must one of B's probabilities grow?", "A factor of 3 must still leave a probability ≤ 1. Only one of B's four tokens is low enough for that."]} />
        </Panel>
      </>
    );
  }

  return <LessonFrame screen={screen} onScreen={setScreen} title="Which model predicts better? Cross-entropy and perplexity" question={question} stage={stage} side={side} />;
}
