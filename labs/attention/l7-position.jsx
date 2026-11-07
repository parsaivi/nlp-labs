import { useState, useMemo, useCallback, useRef, useEffect } from "react";
import { THEME, MONO } from "./common";

// Lesson 7 - Where is word order? (Attention, slides 56-61).
// Aha: self-attention only sees a set of vectors. Shuffle the tokens and every output is the same, just
// shuffled (permutation-equivariant), so "the dog chased the cat" and "the cat chased the dog" look the
// same to it. Adding a position vector to each token breaks the tie. Sinusoids at many frequencies act like
// clock hands: fast ones separate neighbours, slow ones separate distant positions.

// @sim-start
const DM = 512;
const MAXP = 50;
const MIN_SEP = 0.5;
function peVal(pos, dim) {
  const i = Math.floor(dim / 2);
  const w = 1 / Math.pow(10000, (2 * i) / DM);
  return dim % 2 === 0 ? Math.sin(pos * w) : Math.cos(pos * w);
}
function period(i) {
  return 2 * Math.PI * Math.pow(10000, (2 * i) / DM);
}
// closest pair of positions 0..MAXP using only the sin/cos pairs listed in `pairs`
function minSeparation(pairs) {
  let best = Infinity;
  let arg = [0, 1];
  for (let a = 0; a <= MAXP; a++)
    for (let b = a + 1; b <= MAXP; b++) {
      let s = 0;
      for (const i of pairs) {
        const da = peVal(a, 2 * i) - peVal(b, 2 * i);
        const db = peVal(a, 2 * i + 1) - peVal(b, 2 * i + 1);
        s += da * da + db * db;
      }
      const d = Math.sqrt(s);
      if (d < best) {
        best = d;
        arg = [a, b];
      }
    }
  return { best, arg };
}
// @sim-end

// toy 2-D token embeddings and a toy position code for the puzzle
const EMB = { the: [0.2, 0.1], dog: [1.5, 0.6], chased: [-0.3, 1.4], cat: [1.2, -0.9] };
const SA = ["the", "dog", "chased", "the", "cat"];
const SB = ["the", "cat", "chased", "the", "dog"];
function toyPE(i) {
  return [0.9 * Math.sin(0.9 * i), 0.9 * Math.cos(0.9 * i) - 0.9];
}
function selfAttend(tokens, withPE) {
  const x = tokens.map((t, i) => (withPE ? vadd(EMB[t], toyPE(i)) : EMB[t]));
  return x.map((q) => {
    const a = softmax(x.map((k) => dot(q, k)));
    return [0, 1].map((d) => x.reduce((s, v, j) => s + a[j] * v[d], 0));
  });
}

function PEHeat({ dimSel, posSel }) {
  const ref = useRef(null);
  const D = 128;
  useEffect(() => {
    const c = ref.current;
    if (!c) return;
    const ctx = c.getContext("2d");
    const img = ctx.createImageData(MAXP + 1, D);
    const lo = THEME.highlight;
    const hi = THEME.accent;
    const p = (h) => [parseInt(h.slice(1, 3), 16), parseInt(h.slice(3, 5), 16), parseInt(h.slice(5, 7), 16)];
    const A = p(lo);
    const B = p(hi);
    const Z = p(THEME.bg);
    for (let d = 0; d < D; d++)
      for (let x = 0; x <= MAXP; x++) {
        const v = peVal(x, d);
        const col = v >= 0 ? B : A;
        const t = Math.abs(v);
        const k = (d * (MAXP + 1) + x) * 4;
        img.data[k] = Z[0] + (col[0] - Z[0]) * t;
        img.data[k + 1] = Z[1] + (col[1] - Z[1]) * t;
        img.data[k + 2] = Z[2] + (col[2] - Z[2]) * t;
        img.data[k + 3] = 255;
      }
    ctx.putImageData(img, 0, 0);
  }, []);
  return (
    <div style={{ position: "relative", width: "100%" }}>
      <canvas ref={ref} width={MAXP + 1} height={D} style={{ width: "100%", height: 220, imageRendering: "pixelated", display: "block", border: "1px solid " + THEME.grid, borderRadius: 4 }} />
      <svg viewBox={"0 0 " + (MAXP + 1) + " " + D} preserveAspectRatio="none" style={{ position: "absolute", inset: 0, width: "100%", height: 220, pointerEvents: "none" }}>
        <rect x={0} y={dimSel} width={MAXP + 1} height={1} fill="none" stroke={THEME.text} strokeWidth={0.4} />
        <rect x={posSel} y={0} width={1} height={D} fill="none" stroke={THEME.text} strokeWidth={0.4} />
      </svg>
      <div style={{ display: "flex", justifyContent: "space-between", fontFamily: MONO, fontSize: 11, color: THEME.muted, marginTop: 2 }}>
        <span>position 0</span>
        <span>dims 0–127 top to bottom · violet + · apricot −</span>
        <span>position {MAXP}</span>
      </div>
    </div>
  );
}
function Line({ fns, xr, yr, label, marks }) {
  const W = 600;
  const H = 150;
  const X = (x) => 34 + ((x - xr[0]) / (xr[1] - xr[0])) * (W - 50);
  const Y = (y) => H - 22 - ((y - yr[0]) / (yr[1] - yr[0])) * (H - 40);
  return (
    <svg viewBox={"0 0 " + W + " " + H} style={{ width: "100%", display: "block" }} role="img" aria-label={label}>
      <line x1={34} x2={W - 16} y1={Y(0)} y2={Y(0)} stroke={THEME.grid} />
      {fns.map((f, k) => {
        let d = "";
        for (let x = xr[0]; x <= xr[1] + 1e-9; x += (xr[1] - xr[0]) / 200) d += (d ? "L" : "M") + X(x).toFixed(1) + "," + Y(clamp(f.fn(x), yr[0], yr[1])).toFixed(1);
        return <path key={k} d={d} fill="none" stroke={f.color} strokeWidth={2} strokeDasharray={f.dash} />;
      })}
      {(marks || []).map((m, k) => (
        <circle key={k} cx={X(m[0])} cy={Y(m[1])} r={3} fill={THEME.text} />
      ))}
      <SvgLabel x={34} y={13} color={THEME.muted} size={11}>
        {label}
      </SvgLabel>
      <SvgLabel x={W - 16} y={H - 6} anchor="end" color={THEME.muted} size={10.5}>
        {xr[1]}
      </SvgLabel>
      <SvgLabel x={34} y={H - 6} color={THEME.muted} size={10.5}>
        {xr[0]}
      </SvgLabel>
    </svg>
  );
}
function Clocks({ i1, i2, arg }) {
  const W = 600;
  const H = 230;
  const cs = [
    [150, 118],
    [450, 118],
  ];
  const R = 88;
  return (
    <svg viewBox={"0 0 " + W + " " + H} style={{ width: "100%", display: "block" }} role="img" aria-label="Positions 0 to 50 on two clock faces">
      {[i1, i2].map((i, c) => {
        const w = 1 / Math.pow(10000, (2 * i) / DM);
        return (
          <g key={c}>
            <circle cx={cs[c][0]} cy={cs[c][1]} r={R} fill="none" stroke={THEME.grid} />
            {Array.from({ length: MAXP + 1 }, (_, p) => {
              const x = cs[c][0] + R * Math.sin(p * w);
              const y = cs[c][1] - R * Math.cos(p * w);
              const hot = p === arg[0] || p === arg[1];
              return <circle key={p} cx={x} cy={y} r={hot ? 6 : 3.2} fill={hot ? THEME.highlight : rgba(THEME.accent, 0.35 + (0.65 * p) / MAXP)} stroke={hot ? THEME.text : "none"} />;
            })}
            <SvgLabel x={cs[c][0]} y={H - 6} anchor="middle" color={THEME.muted} size={11.5}>
              {"pair i = " + i + " · period " + fmt(period(i), 1)}
            </SvgLabel>
          </g>
        );
      })}
    </svg>
  );
}

export default function PositionLesson({ initialScreen }) {
  const [screen, setScreen] = useState(initialScreen || 0);
  const [order, setOrder] = useState("A");
  const [pe, setPe] = useState(false);
  const [pairI, setPairI] = useState(0);
  const [pos, setPos] = useState(10);
  const [p0, setP0] = useState(12);
  const [i1, setI1] = useState(0);
  const [i2, setI2] = useState(1);

  const zA = useMemo(() => selfAttend(SA, pe), [pe]);
  const zB = useMemo(() => selfAttend(SB, pe), [pe]);
  const sep = useMemo(() => minSeparation([i1, i2]), [i1, i2]);

  let question, stage, side;
  if (screen === 0) {
    const toks = order === "A" ? SA : SB;
    const z = order === "A" ? zA : zB;
    const dDog = Math.hypot(zA[1][0] - zB[4][0], zA[1][1] - zB[4][1]);
    const dCat = Math.hypot(zA[4][0] - zB[1][0], zA[4][1] - zB[1][1]);
    const pts = [];
    SA.forEach((t, i) => pts.push({ id: "a" + i, label: i === 3 ? "" : t + " (A)", x: zA[i][0], y: zA[i][1], color: THEME.accent, shape: "circle", r: 7 }));
    SB.forEach((t, i) => pts.push({ id: "b" + i, label: i === 3 ? "" : t + " (B)", x: zB[i][0], y: zB[i][1], color: THEME.highlight, shape: "square", r: 6, hollow: true, ly: 16 }));
    question = (
      <>
        <b>Where we are:</b> lessons 4–6 built the encoder and decoder from attention alone. Nothing in them looks at word order. <b>The puzzle:</b> A = "the dog chased the cat", B = "the cat chased the dog" (slide 56). The plane shows every token's self-attention output for both sentences. Do any outputs differ between A and B? With only word embeddings, can the model tell who chased whom? Then add position vectors.
      </>
    );
    stage = (
      <>
        <div style={{ fontFamily: MONO, fontSize: 15, margin: "4px 0 8px" }}>
          {toks.map((t, i) => (
            <span key={i} style={{ padding: "2px 7px", marginRight: 4, border: "1px solid " + THEME.grid, borderRadius: 4 }}>
              <span style={{ color: THEME.muted, fontSize: 11 }}>{i} </span>
              {t}
            </span>
          ))}
        </div>
        <WordPlane xr={[-0.8, 2.2]} yr={[-1.6, 1.8]} ticks={0.5} points={pts} ariaLabel="Self-attention outputs for sentence A (filled) and B (hollow)" />
      </>
    );
    side = (
      <>
        <Panel>
          <Pills
            options={[
              ["A", "A: the dog chased the cat"],
              ["B", "B: the cat chased the dog"],
            ]}
            value={order}
            onPick={setOrder}
          />
          <Toggle label="add a position vector to each token" value={pe} onChange={setPe} />
          <Readout
            rows={[
              ["‖z_dog(A) − z_dog(B)‖", fmt(dDog, 3), dDog > 1e-6 ? THEME.ok : THEME.warn],
              ["‖z_cat(A) − z_cat(B)‖", fmt(dCat, 3), dCat > 1e-6 ? THEME.ok : THEME.warn],
            ]}
          />
          <Goal met={dDog > 0.05}>dog's output depends on where dog stands</Goal>
        </Panel>
        <Note>● sentence A · □ sentence B. Without positions, each word has the same output in both sentences: swapping the words only swaps the outputs. Self-attention is permutation-equivariant (slide 56).</Note>
      </>
    );
  } else if (screen === 1) {
    const i = pairI;
    const w = 1 / Math.pow(10000, (2 * i) / DM);
    question = (
      <>
        <b>Claim to test:</b> a position vector can be learned (one trainable vector per position), or computed by a rule (slide 58). The original Transformer computes it: every pair of dimensions is a sine and cosine of the position at its own frequency (slide 59). Pick a dimension pair. Which pairs change quickly with position, and which barely move over 50 tokens? If you kept only pair 0, could you tell position 3 from position 9?
      </>
    );
    stage = (
      <>
        <PEHeat dimSel={2 * i} posSel={pos} />
        <Line
          fns={[
            { fn: (x) => Math.sin(x * w), color: THEME.accent },
            { fn: (x) => Math.cos(x * w), color: THEME.highlight, dash: "5 4" },
          ]}
          xr={[0, MAXP]}
          yr={[-1.1, 1.1]}
          label={"pair i = " + i + ": sin (solid) and cos (dashed) over positions"}
          marks={[
            [pos, Math.sin(pos * w)],
            [pos, Math.cos(pos * w)],
          ]}
        />
      </>
    );
    side = (
      <>
        <Panel>
          <Slider id="a7-i" label="dimension pair i (dims 2i, 2i+1)" value={pairI} min={0} max={63} stepSize={1} onChange={setPairI} display={"i = " + i} />
          <Slider id="a7-p" label="position" value={pos} min={0} max={MAXP} stepSize={1} onChange={setPos} display={String(pos)} />
          <Readout
            rows={[
              ["frequency 1/10000^(2i/512)", fmt(w, 4)],
              ["period in positions", fmt(period(i), 1)],
              ["PE(pos, 2i), PE(pos, 2i+1)", fmt(Math.sin(pos * w), 3) + ", " + fmt(Math.cos(pos * w), 3)],
              ["pair 0 at positions 3 and 9", "(" + fmt(Math.sin(3), 2) + ", " + fmt(Math.cos(3), 2) + ") vs (" + fmt(Math.sin(9), 2) + ", " + fmt(Math.cos(9), 2) + ")"],
            ]}
          />
        </Panel>
        <Note>Pair 0 repeats every 2π ≈ 6.3 positions, so on its own it confuses positions about 6 apart. High pairs change so slowly that neighbours look identical. Each pair alone is ambiguous. Together they are not.</Note>
      </>
    );
  } else if (screen === 2) {
    const sim = (p, k) => {
      let s = 0;
      for (let d = 0; d < DM; d++) s += peVal(p, d) * peVal(p + k, d);
      return s / (DM / 2);
    };
    question = (
      <>
        <b>What you found:</b> attention alone cannot see order, so each token gets its position added: the same token at a different position becomes a different input vector (slide 57). The sinusoidal code uses many frequencies, like the hands of a clock. Fast pairs tell neighbours apart, slow pairs tell distant positions apart. Move the start position p: the similarity of PE(p) and PE(p + k) depends only on the offset k.
      </>
    );
    stage = (
      <>
        <Line
          fns={[
            { fn: (k) => sim(0, k), color: THEME.muted, dash: "5 4" },
            { fn: (k) => sim(p0, k), color: THEME.accent },
          ]}
          xr={[0, 40]}
          yr={[-0.2, 1.05]}
          label={"PE(p)·PE(p + k) / 256 against offset k · dashed p = 0, solid p = " + p0}
        />
        <Eq size={12.5}>
          PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
          <br />
          PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
          <br />
          input to the first layer = token embedding × √d_model + PE(pos)
        </Eq>
      </>
    );
    side = (
      <>
        <Panel>
          <Slider id="a7-p0" label="start position p" value={p0} min={0} max={40} stepSize={1} onChange={setP0} display={"p = " + p0} />
          <Readout
            rows={[
              ["similarity at k = 1", fmt(sim(p0, 1), 3)],
              ["similarity at k = 10", fmt(sim(p0, 10), 3)],
            ]}
          />
        </Panel>
        <Note>
          Slides 56–61. The two curves lie on top of each other: each sin/cos pair turns by a fixed angle per step, so the dot product depends only on k. The paper also tried learned position embeddings with similar results. Now the whole Transformer is assembled: encoder self-attention, masked decoder self-attention, cross-attention, and positions (slide 61).
        </Note>
      </>
    );
  } else {
    const ok = sep.best >= MIN_SEP;
    question = (
      <>
        Suppose a position code may use only two sin/cos pairs (4 numbers) out of the 256 in d_model = 512. Choose the two pairs so that positions 0 to {MAXP} all stay distinguishable: every two different positions must be at least {MIN_SEP} apart.
      </>
    );
    stage = (
      <>
        <Clocks i1={i1} i2={i2} arg={sep.arg} />
        <Note>Each clock face shows one pair: position p sits at angle p × frequency. Darker dots are later positions. The two large dots are the closest pair of positions.</Note>
      </>
    );
    side = (
      <>
        <Panel edge={ok ? THEME.ok : THEME.grid}>
          <Slider id="a7-i1" label="first pair i₁" value={i1} min={0} max={255} stepSize={1} onChange={setI1} display={"i₁ = " + i1 + " (period " + fmt(period(i1), 1) + ")"} />
          <Slider id="a7-i2" label="second pair i₂" value={i2} min={0} max={255} stepSize={1} onChange={setI2} display={"i₂ = " + i2 + " (period " + fmt(period(i2), 1) + ")"} />
          <Readout
            rows={[
              ["closest positions", sep.arg[0] + " and " + sep.arg[1]],
              ["their distance", fmt(sep.best, 3), ok ? THEME.ok : THEME.text],
            ]}
          />
          <Goal met={ok}>every pair of positions at least {MIN_SEP} apart</Goal>
          <Hint
            hints={[
              "Look at which two positions collide. Are they neighbours (the clocks turn too slowly) or far apart (a clock wrapped around)?",
              "Use one fast clock to separate neighbours, and one slow clock whose period is about the length of the range, roughly 50 positions, so nothing wraps around. Pair 0 has period 6.3. Pairs near 60 have periods near 54.",
            ]}
          />
        </Panel>
        <Note>The real code keeps all 256 pairs, so it has clocks at every speed, from period 6.3 up to about 60,000 positions.</Note>
      </>
    );
  }

  return <LessonFrame screen={screen} onScreen={setScreen} title="Where is word order?" question={question} stage={stage} side={side} />;
}
