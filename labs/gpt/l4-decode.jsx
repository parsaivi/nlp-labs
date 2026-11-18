import { useState, useCallback, useMemo } from "react";
import { THEME, MONO } from "./common";

// Lesson 4 - Which token? Greedy, temperature, top-k, top-p (GPT, slides 44, 45 and 53).
// Aha: the model gives a distribution; choosing from it is a separate decision. Greedy takes the best next
// token, which need not give the best sequence. Temperature rescales the logits (same weights, a sharper or
// flatter distribution). Top-k keeps a fixed number of candidates; top-p keeps the smallest set holding
// probability p, so its size adapts to how confident the model is.

// @sim-start
function withTemp(ps, T) {
  const z = ps.map((p) => Math.log(p) / T);
  return softmax(z);
}
// keep the top k, or the smallest set whose mass reaches p; renormalize. Returns probs (0 = filtered out)
function filterDist(ps, mode, k, p) {
  const order = ps.map((v, i) => i).sort((a, b) => ps[b] - ps[a]);
  const keep = {};
  if (mode === "k") {
    for (let i = 0; i < Math.min(k, ps.length); i++) keep[order[i]] = true;
  } else if (mode === "p") {
    let acc = 0;
    for (const i of order) {
      keep[i] = true;
      acc += ps[i];
      if (acc >= p - 1e-9) break;
    }
  } else {
    for (const i of order) keep[i] = true;
  }
  let s = 0;
  for (const i in keep) s += ps[i];
  return ps.map((v, i) => (keep[i] ? v / s : 0));
}
function keptCount(q) {
  return q.filter((v) => v > 0).length;
}
const CONFIDENT = [0.92, 0.03, 0.02, 0.01, 0.01, 0.01];
const FLAT = [0.18, 0.16, 0.15, 0.14, 0.13, 0.12, 0.12];
// @sim-end

const TOK6 = ["mat", "floor", "chair", "rug", "bed", "sofa"];
const P6 = [0.4, 0.25, 0.15, 0.1, 0.06, 0.04]; // slide 53
const TOK3 = ["mat", "floor", "chair"];
const P3 = [0.55, 0.25, 0.2]; // slides 44-45
// a toy two-step tree after "The cat sat on the"
const TREE = {
  mat: [[".", 0.4], ["and", 0.3], ["near", 0.3]],
  floor: [[".", 0.96], ["and", 0.04]],
  chair: [[".", 0.9], ["and", 0.1]],
};
const CONF_TOK = ["mat", "floor", "rug", "bed", "sofa", "chair"];
const FLAT_TOK = ["blue", "red", "green", "old", "big", "small", "new"];

function entropy(q) {
  let h = 0;
  for (const v of q) if (v > 0) h -= v * Math.log(v);
  return h;
}
function DistPanel({ toks, base, final, counts, n }) {
  return (
    <Bars
      rows={toks.map((t, i) => ({ label: t, value: final[i], color: final[i] > 0 ? THEME.accent : THEME.grid, mark: final[i] > 0 ? (counts ? "drawn " + fmt((100 * counts[i]) / n, 1) + "%" : "was " + fmt(base[i] * 100, 0) + "%") : "filtered" }))}
      range={[0, 1]}
      valueFmt={(v) => fmt(v * 100, 1) + "%"}
      labelW={70}
      ariaLabel="sampling distribution after temperature and filtering"
    />
  );
}
function PathBtn({ a, b, onPick, active }) {
  const click = useCallback(() => onPick(a + " " + b), [onPick, a, b]);
  return (
    <button onClick={click} aria-pressed={active} style={{ fontFamily: MONO, fontSize: 12, padding: "3px 8px", margin: 2, borderRadius: 4, cursor: "pointer", border: "1px solid " + (active ? THEME.highlight : THEME.grid), background: active ? rgba(THEME.highlight, 0.16) : THEME.bg, color: THEME.text }}>
      {a + " " + b}
    </button>
  );
}

export default function DecodeLesson({ initialScreen }) {
  const [screen, setScreen] = useState(initialScreen || 0);
  const [path, setPath] = useState("mat .");
  const [seen, setSeen] = useState({ "mat .": true });
  const [T, setT] = useState(1);
  const [mode, setMode] = useState("none");
  const [k, setK] = useState(3);
  const [p, setP] = useState(0.9);
  const [draws, setDraws] = useState(null);
  const [seed, setSeed] = useState(1);
  const [preset, setPreset] = useState("T1");
  const [cm, setCm] = useState("k");
  const [ck, setCk] = useState(3);
  const [cp, setCp] = useState(0.5);

  const pickPath = useCallback((s) => {
    setPath(s);
    setSeen((S) => Object.assign({}, S, { [s]: true }));
  }, []);
  const greedy = useCallback(() => pickPath("mat ."), [pickPath]);
  const final6 = useMemo(() => filterDist(withTemp(P6, T), mode, k, p), [T, mode, k, p]);
  const sample = useCallback(() => {
    const rng = mulberry32(seed);
    const c = final6.map(() => 0);
    for (let i = 0; i < 1000; i++) {
      let u = rng();
      let j = 0;
      while (j < final6.length - 1 && u >= final6[j]) {
        u -= final6[j];
        j++;
      }
      c[j]++;
    }
    setDraws({ c, key: [T, mode, k, p].join("|") });
    setSeed((s) => s + 1);
  }, [final6, seed, T, mode, k, p]);
  const draw = draws && draws.key === [T, mode, k, p].join("|") ? draws.c : null;

  let question, stage, side;
  if (screen === 0) {
    const parts = path.split(" ");
    const p1 = P3[TOK3.indexOf(parts[0])];
    const p2 = TREE[parts[0]].find((x) => x[0] === parts[1])[1];
    const allPaths = [];
    for (const a of TOK3) for (const x of TREE[a]) allPaths.push([a, x[0], P3[TOK3.indexOf(a)] * x[1]]);
    allPaths.sort((u, v) => v[2] - u[2]);
    const foundBest = !!seen[allPaths[0][0] + " " + allPaths[0][1]];
    question = (
      <>
        <b>Where we are:</b> GPT-2 generates by predicting a distribution, choosing a token and appending it (lesson 3). But "choose" hides a decision (slide 44). <b>The puzzle:</b> greedy decoding always takes the most likely next token. After <span style={{ fontFamily: MONO }}>The cat sat on the</span>, that is "mat", then "." Is "mat ." the most likely two-token continuation? Try other paths.
      </>
    );
    stage = (
      <>
        <svg viewBox="0 0 600 250" style={{ width: "100%", display: "block", background: THEME.bg, border: "1px solid " + THEME.grid, borderRadius: 6 }} role="img" aria-label="two-step tree of next-token probabilities">
          <SvgLabel x={14} y={128} size={12}>
            The cat sat on the
          </SvgLabel>
          {TOK3.map((a, i) => {
            const y = 45 + i * 80;
            const on = parts[0] === a;
            return (
              <g key={a}>
                <line x1={150} y1={124} x2={250} y2={y} stroke={on ? THEME.highlight : THEME.grid} strokeWidth={on ? 2.5 : 1.2} />
                <SvgLabel x={196} y={(124 + y) / 2 - 4} size={11} color={THEME.muted} anchor="middle">
                  {fmt(P3[i], 2)}
                </SvgLabel>
                <SvgLabel x={256} y={y + 4} size={13} color={on ? THEME.highlight : THEME.text}>
                  {a}
                </SvgLabel>
                {TREE[a].map((x, j) => {
                  const y2 = y - 24 + j * 22;
                  const on2 = on && parts[1] === x[0];
                  return (
                    <g key={x[0]}>
                      <line x1={300} y1={y} x2={420} y2={y2} stroke={on2 ? THEME.highlight : THEME.grid} strokeWidth={on2 ? 2.5 : 1} />
                      <SvgLabel x={426} y={y2 + 4} size={12} color={on2 ? THEME.highlight : THEME.text}>
                        {x[0] + "  " + fmt(x[1], 2)}
                      </SvgLabel>
                      <SvgLabel x={590} y={y2 + 4} size={11} anchor="end" color={THEME.muted}>
                        {fmt(P3[i] * x[1], 4)}
                      </SvgLabel>
                    </g>
                  );
                })}
              </g>
            );
          })}
        </svg>
        <div style={{ marginTop: 6 }}>
          {allPaths.map((q) => (
            <PathBtn key={q[0] + q[1]} a={q[0]} b={q[1]} onPick={pickPath} active={path === q[0] + " " + q[1]} />
          ))}
        </div>
      </>
    );
    side = (
      <>
        <Panel>
          <Btn onClick={greedy}>greedy path</Btn>
          <Readout
            rows={[
              ["chosen path", path],
              ["P = P(first) × P(second | first)", fmt(p1, 2) + " × " + fmt(p2, 2) + " = " + fmt(p1 * p2, 4), THEME.highlight],
              ["greedy path probability", fmt(0.55 * 0.4, 4)],
            ]}
          />
          <Goal met={foundBest}>find the most likely two-token continuation</Goal>
        </Panel>
        <Note>First step from slide 44 (mat 55%, floor 25%, chair 20%). The second step is a toy. Greedy chooses the best next token, not necessarily the best sequence (slide 44).</Note>
      </>
    );
  } else if (screen === 1) {
    const kept = keptCount(final6);
    const mass = P6.reduce((a, v, i) => a + (final6[i] > 0 ? withTemp(P6, T)[i] : 0), 0);
    question = (
      <>
        <b>Claim to test:</b> instead of always taking the top token, sample one (slide 44's beads). Temperature T divides the logits before the softmax; top-k and top-p cut the list before sampling (slides 45, 53). Draw 1,000 samples to check that sampling follows the bars. What does T → 0 turn sampling into? When does top-k cut a token that top-p keeps?
      </>
    );
    stage = (
      <>
        <DistPanel toks={TOK6} base={P6} final={final6} counts={draw} n={1000} />
        <div style={{ display: "flex", gap: 6, marginTop: 8 }}>
          <Btn onClick={sample} strong>
            draw 1,000 samples
          </Btn>
        </div>
      </>
    );
    side = (
      <>
        <Panel>
          <Slider id="d4-T" label="temperature T" value={T} min={0.05} max={3} stepSize={0.01} onChange={setT} display={"T = " + fmt(T, 2)} />
          <Label>filter</Label>
          <Pills
            options={[
              ["none", "none"],
              ["k", "top-k"],
              ["p", "top-p"],
            ]}
            value={mode}
            onPick={setMode}
          />
          {mode === "k" ? <Slider id="d4-k" label="k" value={k} min={1} max={6} stepSize={1} onChange={setK} display={"k = " + k} /> : null}
          {mode === "p" ? <Slider id="d4-p" label="p" value={p} min={0.05} max={1} stepSize={0.01} onChange={setP} display={"p = " + fmt(p, 2)} /> : null}
          <Readout
            rows={[
              ["candidates kept", kept + " / 6"],
              ["probability mass kept (before renormalizing)", fmt(mass * 100, 1) + "%"],
              ["entropy of the sampling distribution", fmt(entropy(final6), 3) + " nats"],
            ]}
          />
        </Panel>
        <Note>Base distribution from slide 53: 40, 25, 15, 10, 6, 4%. Samples come from a seeded random generator, so everyone sees the same draws.</Note>
      </>
    );
  } else if (screen === 2) {
    const cfg = { T05: [0.5, P3], T1: [1, P3], T2: [2, P3] };
    const q3 = preset.startsWith("T") ? withTemp(P3, cfg[preset][0]) : null;
    const q6 = preset === "k3" ? filterDist(P6, "k", 3, 0) : preset === "p90" ? filterDist(P6, "p", 0, 0.9) : null;
    question = (
      <>
        <b>What you found:</b> greedy is one fixed choice; sampling draws from the distribution. Temperature reshapes it without changing any weight: T &lt; 1 sharpens, T &gt; 1 flattens, T → 0 becomes greedy. Top-k keeps a fixed count; top-p keeps the smallest set reaching mass p, so the count follows the model's confidence. Load the slides' own settings.
      </>
    );
    stage = (
      <>
        <Eq size={13.5}>
          pᵢ(T) = exp(zᵢ / T) / Σⱼ exp(zⱼ / T) (zᵢ = logits; here zᵢ = ln pᵢ)
          <br />
          top-k: keep the k largest · top-p: keep the smallest set with Σ ≥ p
          <br />
          then renormalize and sample
        </Eq>
        {q3 ? <DistPanel toks={TOK3} base={P3} final={q3} /> : <DistPanel toks={TOK6} base={P6} final={q6} />}
      </>
    );
    side = (
      <>
        <Panel>
          <Pills
            options={[
              ["T05", "T = 0.5"],
              ["T1", "T = 1"],
              ["T2", "T = 2"],
              ["k3", "top-k = 3"],
              ["p90", "top-p = 0.90"],
            ]}
            value={preset}
            onPick={setPreset}
          />
        </Panel>
        <Note>Slide 45: T = 0.5 gives 74.7 / 15.4 / 9.9%, T = 2 gives 43.9 / 29.6 / 26.5%. Slide 53: top-k = 3 keeps 80% of the mass (mat, floor, chair); top-p = 0.90 keeps 4 tokens, 90%. GPT-2's unicorn story was sampled with top-k = 40 (slide 50). Top-p is a later method (Holtzman et al.).</Note>
      </>
    );
  } else {
    const qc = filterDist(CONFIDENT, cm, ck, cp);
    const qf = filterDist(FLAT, cm, ck, cp);
    const nc = keptCount(qc);
    const nf = keptCount(qf);
    const g1 = nc === 1;
    const g2 = nf >= 5;
    question = (
      <>
        One decoding setting must serve two moments in the same text. When the model is sure ("The cat sat on the …": mat 92%), keep only that token, so no typo-like word sneaks in. When many words fit ("She painted the door …"), keep at least 5 candidates, so the text stays varied. Choose top-k or top-p and one value that does both.
      </>
    );
    stage = (
      <>
        <Label>confident context</Label>
        <DistPanel toks={CONF_TOK} base={CONFIDENT} final={qc} />
        <Label>open context</Label>
        <DistPanel toks={FLAT_TOK} base={FLAT} final={qf} />
      </>
    );
    side = (
      <>
        <Panel edge={g1 && g2 ? THEME.ok : THEME.grid}>
          <Pills
            options={[
              ["k", "top-k"],
              ["p", "top-p"],
            ]}
            value={cm}
            onPick={setCm}
          />
          {cm === "k" ? <Slider id="d4-ck" label="k" value={ck} min={1} max={7} stepSize={1} onChange={setCk} display={"k = " + ck} /> : <Slider id="d4-cp" label="p" value={cp} min={0.05} max={1} stepSize={0.01} onChange={setCp} display={"p = " + fmt(cp, 2)} />}
          <Readout
            rows={[
              ["kept in the confident context", String(nc), g1 ? THEME.ok : THEME.text],
              ["kept in the open context", String(nf), g2 ? THEME.ok : THEME.text],
            ]}
          />
          <Goal met={g1}>confident context: exactly 1 candidate</Goal>
          <Goal met={g2}>open context: at least 5 candidates</Goal>
          <Hint hints={["With top-k, the count is the same in both contexts. Can one number be 1 and at least 5?", "With top-p, the confident context reaches p with its first token if p ≤ 0.92. How large must p be before the open context needs 5 tokens?"]} />
        </Panel>
        <Note>Toy distributions, temperature 1.</Note>
      </>
    );
  }

  return <LessonFrame screen={screen} onScreen={setScreen} title="Which token? Greedy, temperature, top-k, top-p" question={question} stage={stage} side={side} />;
}
