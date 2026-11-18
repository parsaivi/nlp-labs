import { useState, useCallback, useMemo } from "react";
import { THEME, MONO } from "./common";

// Lesson 5 - Generate without recomputing: the KV cache (Pretraining deck, slides 103-107).
// Aha: during generation, a token's keys and values never change once computed, because the causal mask
// means earlier positions never read later ones. So each layer stores its K and V, and every new token
// computes only its own q, k, v. Work per step drops from the whole prefix to one token; memory grows
// with the context: 2 x layers x d x tokens x bytes. Without the causal mask the cache would be wrong.

// @sim-start
const WORDS = ["the", "cat", "sat", "on", "the", "mat"];
const EMB = { the: [0.9, 0.1], cat: [0.2, 1.0], sat: [-0.7, 0.5], on: [-0.4, -0.8], mat: [0.6, -0.9] };
const L1 = { q: [[1.0, 0.3], [-0.2, 0.8]], k: [[0.7, -0.4], [0.5, 0.9]], v: [[0.6, 0.2], [-0.3, 0.7]] };
const L2 = { q: [[0.4, 0.9], [0.8, -0.5]], k: [[-0.6, 0.8], [0.9, 0.3]], v: [[0.5, -0.4], [0.2, 0.9]] };
function embed(seq) {
  return seq.map((w, i) => [EMB[w][0] + 0.15 * Math.sin(i + 1), EMB[w][1] + 0.15 * Math.cos(i + 1)]);
}
// one self-attention layer with a residual path; causal = each position reads itself and the past only
function layer(X, W, causal) {
  const Q = X.map((x) => matVec(W.q, x));
  const K = X.map((x) => matVec(W.k, x));
  const V = X.map((x) => matVec(W.v, x));
  const out = X.map((x, i) => {
    const sc = K.map((k, j) => (causal && j > i ? -Infinity : dot(Q[i], k) / Math.SQRT2));
    const a = softmax(sc);
    const mix = vsum(V.map((v, j) => vscale(v, a[j])), 2);
    return vadd(x, mix);
  });
  return { out, K, V };
}
// keys of layer 2 for every position, when the prefix has n tokens
function layer2Keys(n, causal) {
  const h1 = layer(embed(WORDS.slice(0, n)), L1, causal).out;
  return layer(h1, L2, causal).K;
}
// positions pushed through the stack, and attention scores computed, to generate G tokens after a prompt of P
function costs(P, G) {
  let noCachePos = 0;
  let noCacheScores = 0;
  for (let t = P; t < P + G; t++) {
    noCachePos += t;
    noCacheScores += t * t;
  }
  let cacheScores = P * P;
  for (let t = P + 1; t < P + G; t++) cacheScores += t;
  return { noCachePos, cachePos: P + G - 1, noCacheScores, cacheScores };
}
// @sim-end

const MODELS = {
  small: { name: "GPT-2 small", L: 12, d: 768, params: 0.124e9 },
  xl: { name: "GPT-2 XL", L: 48, d: 1600, params: 1.558e9 },
  gpt3: { name: "GPT-3 175B", L: 96, d: 12288, params: 175e9 },
};
function kvBytes(m, tokens) {
  return 2 * m.L * m.d * tokens * 2;
}
function gb(b) {
  return b / 1e9;
}
const GPU_GB = 24;

export default function CacheLesson({ initialScreen }) {
  const [screen, setScreen] = useState(initialScreen || 0);
  const [causal, setCausal] = useState(true);
  const [seenModes, setSeenModes] = useState({ c: true });
  const [P, setP] = useState(100);
  const [G, setG] = useState(100);
  const [model, setModel] = useState("xl");
  const [stepN, setStepN] = useState(2);
  const [batch, setBatch] = useState(8);
  const [ctx, setCtx] = useState(1024);

  const onCausal = useCallback((v) => {
    setCausal(v);
    setSeenModes((S) => Object.assign({}, S, { [v ? "c" : "f"]: true }));
  }, []);
  const nextTok = useCallback(() => setStepN((s) => Math.min(6, s + 1)), []);
  const restart = useCallback(() => setStepN(2), []);
  const keys = useMemo(() => [3, 4, 5, 6].map((n) => layer2Keys(n, causal)), [causal]);

  let question, stage, side;
  if (screen === 0) {
    const vals = keys.map((K) => WORDS.map((w, j) => (j < K.length ? K[j][0] : NaN)));
    const first = WORDS.map((w, j) => {
      for (const K of keys) if (j < K.length) return K[j][0];
      return NaN;
    });
    let changed = 0;
    vals.forEach((r) =>
      r.forEach((v, j) => {
        if (isFinite(v) && Math.abs(v - first[j]) > 1e-9) changed++;
      })
    );
    question = (
      <>
        <b>Where we are:</b> GPT generates one token at a time (lesson 4). The plain way re-runs the whole prefix through every layer at every step (slide 103). <b>The puzzle:</b> here is a 2-layer toy model generating <span style={{ fontFamily: MONO }}>the cat sat on the mat</span>. Each row is one generation step. Each cell is the layer-2 key of one position, recomputed at that step. Do old positions' keys ever change? Now remove the causal mask.
      </>
    );
    stage = (
      <>
        <Matrix
          rowLabels={["step: 3 tokens", "step: 4 tokens", "step: 5 tokens", "step: 6 tokens"]}
          colLabels={WORDS.map((w, i) => i + 1 + " " + w)}
          values={vals}
          shade={(v, i, j) => (isFinite(v) ? (Math.abs(v - first[j]) > 1e-9 ? 0.7 : 0.25) : 0)}
          colorFor={(v, i, j) => (isFinite(v) && Math.abs(v - first[j]) > 1e-9 ? THEME.warn : THEME.accent)}
          text={(v) => (isFinite(v) ? fmt(v, 3) : "")}
          mark={(v, i, j) => (isFinite(v) && Math.abs(v - first[j]) > 1e-9 ? "Δ" : null)}
          cell={62}
          rowW={120}
          colH={30}
          ariaLabel="layer-2 key (first coordinate) of each position at each generation step"
        />
        <Note>Δ (red) = this key differs from the value computed when the position first appeared. Toy: 2-D vectors, fixed weights, first coordinate shown.</Note>
      </>
    );
    side = (
      <>
        <Panel>
          <Toggle label="causal mask (GPT)" value={causal} onChange={onCausal} />
          <Readout
            rows={[
              ["old keys that changed", String(changed), changed ? THEME.warn : THEME.ok],
              ["can old K/V be stored and reused?", changed ? "no" : "yes"],
            ]}
          />
          <Goal met={!!seenModes.c && !!seenModes.f}>compare causal and full attention</Goal>
        </Panel>
        <Note>Layer-1 keys depend only on a position's own token, so they never change. Layer 2's keys are computed from layer-1 outputs. Without the mask those outputs read the future, so every new token rewrites them.</Note>
      </>
    );
  } else if (screen === 1) {
    const c = costs(P, G);
    const m = MODELS[model];
    question = (
      <>
        <b>Claim to test:</b> with a causal model, keep each layer's old keys and values and process only the new token (slides 103–104). Prefill runs the prompt once and builds the cache; each decode step adds one token (slide 105). How much work does that save as the prompt grows? And what does it cost in memory?
      </>
    );
    stage = (
      <>
        <Label>positions pushed through all layers</Label>
        <Bars
          rows={[
            { label: "no cache", value: c.noCachePos, color: THEME.warn },
            { label: "KV cache", value: c.cachePos, color: THEME.accent },
          ]}
          range={[0, Math.max(c.noCachePos, 1)]}
          valueFmt={(v) => Math.round(v).toLocaleString("en-US")}
          labelW={90}
          ariaLabel="positions processed"
        />
        <Label>attention scores computed (per layer, per head)</Label>
        <Bars
          rows={[
            { label: "no cache", value: c.noCacheScores, color: THEME.warn },
            { label: "KV cache", value: c.cacheScores, color: THEME.accent },
          ]}
          range={[0, Math.max(c.noCacheScores, 1)]}
          valueFmt={(v) => Math.round(v).toLocaleString("en-US")}
          labelW={90}
          ariaLabel="attention scores computed"
        />
      </>
    );
    side = (
      <>
        <Panel>
          <Slider id="c5-P" label="prompt length P" value={P} min={1} max={2000} stepSize={1} onChange={setP} />
          <Slider id="c5-G" label="tokens to generate G" value={G} min={1} max={1000} stepSize={1} onChange={setG} />
          <Pills options={Object.keys(MODELS).map((k) => [k, MODELS[k].name])} value={model} onPick={setModel} />
          <Readout
            rows={[
              ["work saved (positions)", fmt(c.noCachePos / c.cachePos, 1) + "×", THEME.highlight],
              ["cache for P + G tokens, fp16", fmt(gb(kvBytes(m, P + G)), 3) + " GB"],
              ["layers × hidden size", m.L + " × " + m.d.toLocaleString("en-US")],
            ]}
          />
        </Panel>
        <Note>Cache size = 2 (K and V) × layers × d × tokens × 2 bytes per fp16 number, for one sequence. Configurations from the GPT deck, slide 64. Memory grows with the context (slide 105).</Note>
      </>
    );
  } else if (screen === 2) {
    const toks = WORDS.slice(0, stepN);
    question = (
      <>
        <b>What you found:</b> a causal model never revises a past position, so its keys and values can be stored once and reused. Generation splits into prefill (the full prompt in one pass, with the causal mask inside it) and decode (one new query attends to all cached keys; there are no future keys to block). Step through it.
      </>
    );
    stage = (
      <>
        <Matrix
          rowLabels={["layer 1", "layer 2", "…", "layer L"]}
          colLabels={toks.map((w, i) => i + 1 + " " + w)}
          values={[0, 1, 2, 3].map(() => toks.map((w, j) => (stepN > 2 && j === stepN - 1 ? 2 : 1)))}
          shade={(v) => (v === 2 ? 0.75 : 0.3)}
          colorFor={(v) => (v === 2 ? THEME.highlight : THEME.accent)}
          text={(v, i) => (i === 2 ? "…" : v === 2 ? "new" : "K,V")}
          cell={58}
          rowW={70}
          colH={30}
          ariaLabel="KV cache: one stored key and value per position and layer"
        />
        <Eq size={13}>
          {stepN === 2 ? "PREFILL: the prompt (2 tokens) in one pass, causal mask inside;" : "DECODE: one new token → its q, k, v at each layer;"}
          <br />
          {stepN === 2 ? "store K, V at every layer" : "q attends to " + stepN + " keys (" + (stepN - 1) + " cached + its own); append k, v"}
          <br />
          memory per sequence = 2 · L · d · tokens · bytes per number
        </Eq>
      </>
    );
    side = (
      <>
        <Panel>
          <div style={{ display: "flex", gap: 6 }}>
            <Btn onClick={nextTok} strong disabled={stepN >= 6}>
              decode next token
            </Btn>
            <Btn onClick={restart}>back to prefill</Btn>
          </div>
          <Readout
            rows={[
              ["tokens in the cache", String(stepN)],
              ["new positions computed this step", stepN === 2 ? "2 (prefill)" : "1"],
            ]}
          />
        </Panel>
        <Note>Slide 104: process "robot", reuse K/V from "a", append the new K/V. Slide 107: a full prompt or a new chunk still needs the causal mask inside it; one new token has no future keys. Padding and unused cache slots must always be masked. A bidirectional encoder like BERT cannot cache this way (screen 1).</Note>
      </>
    );
  } else {
    const m = MODELS.xl;
    const weights = gb(m.params * 2);
    const per = gb(kvBytes(m, ctx));
    const total = weights + batch * per;
    const fits = total <= GPU_GB + 1e-9;
    const nextFits = weights + (batch + 1) * per <= GPU_GB + 1e-9;
    const g1 = ctx === 1024;
    const g2 = fits;
    const g3 = fits && !nextFits;
    question = (
      <>
        You serve GPT-2 XL (1.558B parameters, 48 layers, d = 1,600) on one 24 GB GPU, in fp16. The weights take 2 bytes per parameter. Every sequence being generated needs its own KV cache. At the full 1,024-token context, what is the largest batch of sequences you can decode at once?
      </>
    );
    stage = (
      <>
        <Bars
          rows={[
            { label: "weights", value: weights, color: THEME.third },
            { label: "KV caches", value: batch * per, color: THEME.accent },
            { label: "total", value: total, color: fits ? THEME.ok : THEME.warn },
          ]}
          range={[0, 40]}
          valueFmt={(v) => fmt(v, 2) + " GB"}
          labelW={90}
          ariaLabel="GPU memory use"
        />
        <Note>GPU memory: {GPU_GB} GB. Activations and other overheads are ignored here.</Note>
      </>
    );
    side = (
      <>
        <Panel edge={g1 && g3 ? THEME.ok : THEME.grid}>
          <Slider id="c5-b" label="sequences in the batch" value={batch} min={1} max={128} stepSize={1} onChange={setBatch} display={String(batch)} />
          <Slider id="c5-c" label="context length (tokens)" value={ctx} min={128} max={1024} stepSize={64} onChange={setCtx} display={String(ctx)} />
          <Readout
            rows={[
              ["weights", fmt(weights, 3) + " GB"],
              ["cache per sequence", fmt(per, 4) + " GB"],
              ["total", fmt(total, 2) + " GB", fits ? THEME.ok : THEME.warn],
            ]}
          />
          <Goal met={g1}>full 1,024-token context</Goal>
          <Goal met={g2}>everything fits in 24 GB</Goal>
          <Goal met={g3}>one more sequence would not fit</Goal>
          <Hint hints={["First compute one sequence's cache: 2 × 48 × 1,600 × 1,024 × 2 bytes.", "Subtract the weights (1.558B × 2 bytes) from 24 GB and divide by one sequence's cache. Round down."]} />
        </Panel>
      </>
    );
  }

  return <LessonFrame screen={screen} onScreen={setScreen} title="Generate without recomputing: the KV cache" question={question} stage={stage} side={side} />;
}
