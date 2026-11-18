import { useState, useCallback } from "react";
import { THEME, MONO } from "./common";

// Lesson 7 - What grew: parameters, tokens, compute (GPT, slides 42, 55, 64, 72-75).
// Aha: a GPT's size follows from two numbers. Each block holds about 12 d² weights (4 d² in attention,
// 8 d² in the MLP), so N ≈ 12 L d² plus the embeddings. Plug in GPT-1, GPT-2 and GPT-3 and you get 117M,
// 1.5B and 175B. Training compute is about 6 N D FLOPs for D tokens; for GPT-3 that is ≈ 3.1 × 10²³,
// ≈ 355 V100-years, ≈ $4.6M at $1.50 per GPU-hour.

// @sim-start
const VOCAB = 50257;
function paramsOf(L, d, V, ctx) {
  const attn = 4 * d * d * L;
  const mlp = 8 * d * d * L;
  const emb = V * d;
  const pos = ctx * d;
  return { attn, mlp, emb, pos, total: attn + mlp + emb + pos };
}
const V100_FLOPS = 28e12; // effective FLOP/s assumed per V100
const SECONDS_PER_YEAR = 365.25 * 24 * 3600;
function trainCost(N, D, flops, pricePerHour) {
  const C = 6 * N * D;
  const gpuSeconds = C / flops;
  return { C, gpuYears: gpuSeconds / SECONDS_PER_YEAR, dollars: (gpuSeconds / 3600) * pricePerHour };
}
// @sim-end

const PRESETS = {
  gpt1: { name: "GPT-1", L: 12, d: 768, V: 40478, ctx: 512, reported: "≈117M" },
  gpt2: { name: "GPT-2 XL", L: 48, d: 1600, V: VOCAB, ctx: 1024, reported: "≈1.5B" },
  gpt3: { name: "GPT-3 175B", L: 96, d: 12288, V: VOCAB, ctx: 2048, reported: "175B" },
};
function big(n) {
  if (n >= 1e9) return fmt(n / 1e9, 2) + "B";
  if (n >= 1e6) return fmt(n / 1e6, 1) + "M";
  return Math.round(n).toLocaleString("en-US");
}
function sci(x) {
  const e = Math.floor(Math.log10(x));
  return fmt(x / Math.pow(10, e), 2) + " × 10^" + e;
}

export default function ScaleLesson({ initialScreen }) {
  const [screen, setScreen] = useState(initialScreen || 0);
  const [L, setL] = useState(12);
  const [d, setD] = useState(768);
  const [hit, setHit] = useState({});
  const [pre, setPre] = useState("gpt3");
  const [Nb, setNb] = useState(175);
  const [Db, setDb] = useState(300);
  const [price, setPrice] = useState(1.5);
  const [cL, setCL] = useState(24);
  const [cd, setCd] = useState(2048);
  const [cD, setCD] = useState(20);

  const record = useCallback((l, dd) => {
    const N = paramsOf(l, dd, VOCAB, 2048).total;
    const near = Math.abs(N - 175e9) / 175e9 < 0.02;
    setHit((H) => Object.assign({}, H, near && l === 96 ? { gpt3: true } : {}, near && l === 48 ? { half: true } : {}));
  }, []);
  const onL = useCallback(
    (v) => {
      setL(v);
      record(v, d);
    },
    [record, d]
  );
  const onD = useCallback(
    (v) => {
      setD(v);
      record(L, v);
    },
    [record, L]
  );

  let question, stage, side;
  if (screen === 0) {
    const P = paramsOf(L, d, VOCAB, 2048);
    question = (
      <>
        <b>Where we are:</b> from GPT-1 to GPT-3 the objective never changed: next-token prediction with a causal Transformer (slides 42, 64). What changed is size. <b>The puzzle:</b> GPT-3 has 96 layers and hidden size 12,288, and 175B parameters. Where does that number come from? Build it with the two sliders. Then build a model with half the layers and the same size: what must d become?
      </>
    );
    stage = (
      <>
        <Bars
          rows={[
            { label: "attention 4·L·d²", value: P.attn, color: THEME.accent },
            { label: "MLP 8·L·d²", value: P.mlp, color: THEME.third },
            { label: "token embeddings V·d", value: P.emb, color: THEME.highlight },
            { label: "positions 2048·d", value: P.pos, color: THEME.grid },
          ]}
          range={[0, Math.max(P.mlp, P.emb, 1)]}
          valueFmt={big}
          labelW={170}
          ariaLabel="parameters by part"
        />
        <Eq size={14}>
          N ≈ 12 · {L} · {d.toLocaleString("en-US")}² + 50,257 · {d.toLocaleString("en-US")} + 2,048 · {d.toLocaleString("en-US")} = <b style={{ color: THEME.highlight }}>{big(P.total)}</b>
        </Eq>
      </>
    );
    side = (
      <>
        <Panel>
          <Slider id="g7-L" label="layers L" value={L} min={1} max={128} stepSize={1} onChange={onL} display={"L = " + L} />
          <Slider id="g7-d" label="hidden size d" value={d} min={64} max={20480} stepSize={64} onChange={onD} display={"d = " + d.toLocaleString("en-US")} />
          <Readout rows={[["parameters", big(P.total), THEME.highlight]]} />
          <Goal met={!!hit.gpt3}>reach 175B (±2%) with 96 layers</Goal>
          <Goal met={!!hit.half}>reach 175B (±2%) with only 48 layers</Goal>
        </Panel>
        <Note>Per block: Q, K, V and output projections are 4 matrices of d × d; the MLP maps d → 4d → d, 8d² more. Biases and layer norms are a rounding error. Doubling L doubles N; doubling d quadruples it.</Note>
      </>
    );
  } else if (screen === 1) {
    const c = PRESETS[pre];
    const P = paramsOf(c.L, c.d, c.V, c.ctx);
    question = (
      <>
        <b>Claim to test:</b> the same formula should recover all three GPTs from their published shapes (slides 42, 64; GPT-1 used a 40,478-token BPE vocabulary). Pick each model. Where do its parameters sit? Why does the embedding table matter for GPT-1 and almost not at all for GPT-3?
      </>
    );
    stage = (
      <>
        <Pills options={Object.keys(PRESETS).map((k) => [k, PRESETS[k].name])} value={pre} onPick={setPre} />
        <Bars
          rows={[
            { label: "attention", value: P.attn / P.total, color: THEME.accent },
            { label: "MLP", value: P.mlp / P.total, color: THEME.third },
            { label: "token embeddings", value: P.emb / P.total, color: THEME.highlight },
            { label: "positions", value: P.pos / P.total, color: THEME.grid },
          ]}
          range={[0, 1]}
          valueFmt={(v) => fmt(v * 100, 1) + "%"}
          labelW={140}
          ariaLabel="share of parameters by part"
        />
      </>
    );
    side = (
      <>
        <Panel>
          <Readout
            rows={[
              ["layers · d", c.L + " · " + c.d.toLocaleString("en-US")],
              ["vocabulary · context", c.V.toLocaleString("en-US") + " · " + c.ctx.toLocaleString("en-US")],
              ["formula", big(P.total), THEME.highlight],
              ["reported", c.reported],
            ]}
          />
        </Panel>
        <Note>Blocks grow with d², the embedding table only with d, so at GPT-3's size the table is about 0.4% of the model. GPT-2 also changed the block itself (pre-LN, byte-level BPE with 50,257 tokens, scaled residual initialization, slide 54), which does not change these counts.</Note>
      </>
    );
  } else if (screen === 2) {
    const N = Nb * 1e9;
    const D = Db * 1e9;
    const r = trainCost(N, D, V100_FLOPS, price);
    question = (
      <>
        <b>What you found:</b> N ≈ 12 L d² + V d recovers 117M, 1.5B and 175B. Training cost follows from N and the number of training tokens D: each token costs about 6 N floating-point operations (2 N forward, 4 N backward). GPT-3 read 300B tokens, in global batches of about 3.2M tokens (slide 72). What did that cost (slide 73)?
      </>
    );
    stage = (
      <>
        <Eq size={14}>
          C ≈ 6 · N · D = 6 · {big(N)} · {Db}B = {sci(r.C)} FLOPs
          <br />
          GPU time = C / (28 × 10¹² FLOP/s per V100) = {fmt(r.gpuYears, 0)} GPU-years
          <br />
          cost = GPU-hours × ${fmt(price, 2)} = ${fmt(r.dollars / 1e6, 2)}M
        </Eq>
        <Readout
          rows={[
            ["tokens per input", "2,048"],
            ["tokens per weight update", "≈3.2M ≈ 1,560 full-length sequences"],
            ["weight updates for 300B tokens", "≈ " + Math.round(300e9 / 3.2e6).toLocaleString("en-US")],
          ]}
        />
      </>
    );
    side = (
      <>
        <Panel>
          <Slider id="g7-N" label="parameters N (billions)" value={Nb} min={0.1} max={500} stepSize={0.1} onChange={setNb} display={fmt(Nb, 1) + "B"} />
          <Slider id="g7-D" label="training tokens D (billions)" value={Db} min={1} max={2000} stepSize={1} onChange={setDb} display={Db + "B"} />
          <Slider id="g7-p" label="price per GPU-hour ($)" value={price} min={0.5} max={4} stepSize={0.05} onChange={setPrice} display={"$" + fmt(price, 2)} />
          <Readout
            rows={[
              ["GPU-years", fmt(r.gpuYears, 0)],
              ["cost", "$" + fmt(r.dollars / 1e6, 2) + "M", THEME.highlight],
            ]}
          />
        </Panel>
        <Note>Slide 73: ≈355 V100 GPU-years and ≈$4.6M for one GPT-3 175B run, assuming $1.50 per GPU-hour (Lambda, 2020); 28 TFLOP/s is the effective V100 speed that estimate assumes. GPU-years measure total work, not calendar time, and people, data and failed experiments are extra.</Note>
      </>
    );
  } else {
    const N = paramsOf(cL, cd, VOCAB, 2048).total;
    const D = cD * 1e9;
    const r = trainCost(N, D, V100_FLOPS, 1.5);
    const g1 = r.dollars <= 50000;
    const g2 = D >= (300 / 175) * N;
    const g3 = N >= 10e9;
    question = (
      <>
        Slide 55 asks where to spend a fixed budget: data, model size or training. Yours is $50,000 of V100 time at $1.50 per hour. Design a model of at least 10B parameters that sees at least as many tokens per parameter as GPT-3 did (300B / 175B ≈ 1.71), and stays within budget.
      </>
    );
    stage = (
      <>
        <Bars
          rows={[
            { label: "your cost", value: r.dollars, color: g1 ? THEME.ok : THEME.warn },
            { label: "budget", value: 50000, color: THEME.grid },
          ]}
          range={[0, 150000]}
          valueFmt={(v) => "$" + Math.round(v).toLocaleString("en-US")}
          labelW={90}
          ariaLabel="cost against budget"
        />
        <Eq size={13.5}>
          N = {big(N)} · D = {cD}B tokens · D / N = {fmt(D / N, 2)} · C = {sci(r.C)} FLOPs
        </Eq>
      </>
    );
    side = (
      <>
        <Panel edge={g1 && g2 && g3 ? THEME.ok : THEME.grid}>
          <Slider id="g7-cL" label="layers L" value={cL} min={1} max={128} stepSize={1} onChange={setCL} display={"L = " + cL} />
          <Slider id="g7-cd" label="hidden size d" value={cd} min={256} max={16384} stepSize={128} onChange={setCd} display={"d = " + cd.toLocaleString("en-US")} />
          <Slider id="g7-cD" label="training tokens D (billions)" value={cD} min={1} max={500} stepSize={1} onChange={setCD} display={cD + "B"} />
          <Goal met={g3}>N ≥ 10B parameters</Goal>
          <Goal met={g2}>D / N ≥ 1.71 tokens per parameter</Goal>
          <Goal met={g1}>cost ≤ $50,000</Goal>
          <Hint hints={["Fix the ratio first: D ≈ 1.71 N. Then the cost grows with N². Find the largest N the budget allows.", "$50,000 buys about 3.4 × 10²¹ FLOPs. With D = 1.71 N, 6 · N · 1.71 N must stay below that, so N can be at most about 18B. Pick L and d for a model between 10B and 18B, then set D."]} />
        </Panel>
        <Note>The GPT-3 ratio is a reference point from this deck, not an optimal rule: slide 55 leaves "what balance is best for the cost" open, for the data-and-scaling lecture.</Note>
      </>
    );
  }

  return <LessonFrame screen={screen} onScreen={setScreen} title="What grew: parameters, tokens, compute" question={question} stage={stage} side={side} />;
}
