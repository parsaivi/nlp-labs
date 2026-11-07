import { useState, useMemo, useCallback } from "react";
import { THEME, MONO } from "./common";

// Lesson 4 - Every token asks, every token answers (Attention, slides 31, 37-39, 46-47, 69-71).
// Aha: in self-attention every token is a query over the same sentence. What a token looks for (its query)
// is separate from what it hands over (its value), so an ambiguous word like «شیر» can ask "which content
// words are around me?" and take its meaning from their values. Same word, different context, different output.

// @sim-start
// toy embeddings x = [meaning 1, meaning 2, content]: content is high for nouns and low for function words
const X = {
  صبحانه: [-1.4, -0.8, 1.0],
  خوردم: [-0.9, -0.7, 0.7],
  جنگل: [1.4, -0.8, 1.0],
  دیدم: [0.2, -0.1, 0.5],
  در: [0, -0.1, 0.1],
  آب: [0, 1.6, 1.0],
  را: [0, 0, 0.05],
  باز: [0.2, 0.9, 0.6],
  کردم: [0, 0.1, 0.3],
  شیر: [0, 0, 1.0],
};
const SENTS = [
  { s: "صبحانه شیر خوردم", sense: "milk", en: "I had milk for breakfast" },
  { s: "در جنگل شیر دیدم", sense: "lion", en: "I saw a lion in the forest" },
  { s: "شیر آب را باز کردم", sense: "tap", en: "I turned on the tap" },
];
const SENSE = { milk: 210, lion: 330, tap: 90 };
const SENSE_NAME = { milk: "milk", lion: "lion", tap: "tap" };
const WIDTH = 35;
const MIN_R = 0.6;
// learned projections (hand-set): W_Q and W_K both keep only the content channel, scaled by g;
// W_V keeps the two meaning channels
function attnRow(tokens, i, q) {
  const sc = tokens.map((t) => dot(q, X[t]));
  const a = softmax(sc);
  const z = [0, 0];
  tokens.forEach((t, j) => {
    z[0] += a[j] * X[t][0];
    z[1] += a[j] * X[t][1];
  });
  return { sc, a, z };
}
function defaultQuery(tok, g) {
  return [0, 0, g * g * X[tok][2]];
}
function senseOf(z) {
  const r = Math.hypot(z[0], z[1]);
  if (r < MIN_R) return "weak";
  let ang = (Math.atan2(z[1], z[0]) * 180) / Math.PI;
  if (ang < 0) ang += 360;
  for (const k in SENSE) {
    let d = Math.abs(ang - SENSE[k]);
    d = Math.min(d, 360 - d);
    if (d <= WIDTH) return k;
  }
  return "between";
}
// @sim-end

const MR = [-2, 2];
const SENSE_COL = { milk: THEME.highlight, lion: THEME.accent, tap: THEME.third };
const SENSE_SHAPE = { milk: "square", lion: "circle", tap: "triangle" };
function sectors(P) {
  const o = P(0, 0);
  const R = 1.9;
  return (
    <g>
      {Object.keys(SENSE).map((k) => {
        const a0 = ((SENSE[k] - WIDTH) * Math.PI) / 180;
        const a1 = ((SENSE[k] + WIDTH) * Math.PI) / 180;
        const pts = [o];
        for (let t = 0; t <= 8; t++) {
          const a = a0 + ((a1 - a0) * t) / 8;
          pts.push(P(R * Math.cos(a), R * Math.sin(a)));
        }
        const lab = P(1.75 * Math.cos((SENSE[k] * Math.PI) / 180), 1.75 * Math.sin((SENSE[k] * Math.PI) / 180));
        return (
          <g key={k}>
            <polygon points={pts.map((p) => p[0] + "," + p[1]).join(" ")} fill={rgba(SENSE_COL[k], 0.1)} stroke={SENSE_COL[k]} strokeDasharray="3 4" strokeWidth={0.8} />
            <SvgLabel x={lab[0]} y={lab[1]} anchor="middle" color={SENSE_COL[k]} size={11.5}>
              {SENSE_NAME[k] + " region"}
            </SvgLabel>
          </g>
        );
      })}
      <circle cx={o[0]} cy={o[1]} r={Math.abs(P(MIN_R, 0)[0] - o[0])} fill={THEME.bg} stroke={THEME.grid} strokeDasharray="2 3" />
    </g>
  );
}
function tokenPoints(tokens, focus) {
  return tokens.map((t, j) => ({ id: "t" + j, label: t, fa: true, x: X[t][0], y: X[t][1], color: j === focus ? THEME.text : THEME.muted, shape: "diamond", r: j === focus ? 7 : 5, dim: X[t][2] < 0.4 }));
}

export default function SelfAttnLesson({ initialScreen }) {
  const [screen, setScreen] = useState(initialScreen || 0);
  const [si, setSi] = useState(0);
  const [qi, setQi] = useState(1);
  const [g, setG] = useState(2.2);
  const [n, setN] = useState(16);
  const [qa, setQa] = useState(-2);
  const [qb, setQb] = useState(-1.5);
  const [qg, setQg] = useState(0.5);

  const S = SENTS[si];
  const tokens = S.s.split(" ");
  const shir = tokens.indexOf("شیر");
  const pickSent = useCallback((i) => {
    setSi(i);
    setQi(SENTS[i].s.split(" ").indexOf("شیر"));
  }, []);
  const rows = useMemo(() => tokens.map((t, i) => attnRow(tokens, i, defaultQuery(t, screen === 1 ? g : 2.2))), [S, g, screen]);
  const qIdx = Math.min(qi, tokens.length - 1);
  const R = rows[qIdx];
  const outPt = (z, label, col, shape) => ({ id: "z" + label, label, x: z[0], y: z[1], color: col, shape: shape || "circle", r: 9 });

  const sentPills = <Pills options={SENTS.map((x, i) => [i, x.s])} value={si} onPick={pickSent} />;
  const tokenChips = (
    <div dir="rtl" style={{ display: "flex", gap: 5, flexWrap: "wrap" }}>
      {tokens.map((t, j) => (
        <Chip key={j} value={j} on={j === qIdx} onPick={setQi}>
          {t}
        </Chip>
      ))}
    </div>
  );

  let question, stage, side;
  if (screen === 0) {
    const zs = rows[shir].z;
    question = (
      <>
        <b>Where we are:</b> Bahdanau attention still runs through an RNN, so each state waits for the previous one (slide 31). The Transformer asks whether attention can replace recurrence entirely. Its encoder lets every token attend to every token of the same sentence. <b>The puzzle:</b> the Word Embedding Lab ended with <Fa>شیر</Fa>, one vector for milk, lion and tap. Here it is the same input vector in three sentences. Switch sentences and watch where <Fa>شیر</Fa>'s output lands. Which token pulls it, and why that one and not the others?
      </>
    );
    stage = <WordPlane xr={MR} yr={MR} ticks={0.5} extra={sectors} points={tokenPoints(tokens, shir).concat([outPt(zs, "output of شیر", SENSE_COL[S.sense], SENSE_SHAPE[S.sense])])} ariaLabel="Meaning plane: token values and the output for شیر" axisLabels={["meaning 1", "meaning 2"]} />;
    side = (
      <>
        <Panel>
          {sentPills}
          <Note>{S.en}</Note>
          <Label>attention of شیر over the sentence</Label>
          <Bars rows={tokens.map((t, j) => ({ label: t, fa: true, value: rows[shir].a[j], color: j === shir ? THEME.muted : THEME.accent }))} range={[0, 1]} labelW={70} rowH={22} ariaLabel="weights from شیر" />
          <Readout rows={[["output lands in", senseOf(zs) === "weak" ? "too close to 0" : senseOf(zs)]]} />
        </Panel>
        <Note>Diamonds are each token's value (its meaning). Faded diamonds are function words. The input vector of شیر sits at the origin, because on its own it means none of the three.</Note>
      </>
    );
  } else if (screen === 1) {
    const all = rows.map((r) => r.a);
    question = (
      <>
        <b>Claim to test:</b> each token makes a query q, a key k and a value v with shared matrices (slide 38). A query matches all keys, and the weights combine the values, giving one output per token (slide 39). Here the query and key keep only the content channel, so content words get found. Click any token to see its row. Then slide the query–key strength down to 0. What is left of the context?
      </>
    );
    stage = (
      <>
        <Matrix rowLabels={tokens} colLabels={tokens} values={all} shade={(v) => v} text={(v) => fmt(v, 2)} hiRow={qIdx} faRows faCols cell={tokens.length > 4 ? 50 : 58} rowW={80} colH={32} ariaLabel="attention weights, rows are queries, columns are keys" />
        <WordPlane xr={MR} yr={MR} ticks={0.5} extra={sectors} H={340} points={tokenPoints(tokens, qIdx).concat([outPt(R.z, "z of " + tokens[qIdx], THEME.highlight, "circle")])} ariaLabel="Output of the selected token" axisLabels={["meaning 1", "meaning 2"]} />
      </>
    );
    side = (
      <>
        <Panel>
          {sentPills}
          <Label>query token (row)</Label>
          {tokenChips}
          <Slider id="a4-g" label="query–key strength (scales W_Q and W_K)" value={g} min={0} max={3} stepSize={0.02} onChange={setG} display={fmt(g, 2)} />
          <Readout
            rows={[
              ["weights of «" + tokens[qIdx] + "»", R.a.map((v) => fmt(v, 2)).join("  ")],
              ["its output z", "(" + fmt(R.z[0], 2) + ", " + fmt(R.z[1], 2) + ")"],
            ]}
          />
        </Panel>
        <Note>At strength 0 every score is equal, so every token gets the plain average of the sentence: lesson 1's blurred average again. Function words then dilute the meaning.</Note>
      </>
    );
  } else if (screen === 2) {
    const nn = n * n;
    question = (
      <>
        <b>What you found:</b> one layer of self-attention gives every token its own weighted mix of the whole sentence. What a token looks for (Q, K) is separate from what it hands over (V), so <Fa>شیر</Fa> can look for content words and take their meaning. All positions are computed at once, and any two tokens are one step apart. The price is n² scores. Change the sentence length n.
      </>
    );
    stage = (
      <>
        <Eq size={12.5}>
          Q = XW_Q, K = XW_K, V = XW_V (one row per token, same matrices for every position)
          <br />
          Z = softmax(QKᵀ / √d_k) V, softmax over keys, row by row
          <br />
          stack 6 such layers (Transformer Base, slide 46): each output can use the whole source
        </Eq>
        <Bars
          rows={[
            { label: "attention scores per layer (n²)", value: nn, color: THEME.accent },
            { label: "RNN sequential steps (n)", value: n, color: THEME.highlight },
          ]}
          range={[0, 4096]}
          digits={0}
          labelW={220}
          ariaLabel="cost comparison"
        />
      </>
    );
    side = (
      <>
        <Panel>
          <Slider id="a4-n" label="sentence length n" value={n} min={2} max={64} stepSize={1} onChange={setN} display={"n = " + n} />
          <Readout
            rows={[
              ["", "self-attention · RNN"],
              ["core cost per layer", "O(n²d) · O(nd²)"],
              ["sequential steps", "O(1) · O(n) = " + n],
              ["longest path between tokens", "1 · " + (n - 1)],
              ["scores at this n", nn.toLocaleString("en-US") + " · —"],
            ]}
          />
        </Panel>
        <Note>Slides 38–39 and 69–71 (Vaswani et al. 2017, Table 1). Doubling n makes 4× as many scores. O(1) sequential steps does not mean constant time. Next lesson: why the scores are divided by √d_k.</Note>
      </>
    );
  } else {
    const q = [qa, qb, qg];
    const res = SENTS.map((x) => {
      const tk = x.s.split(" ");
      const r = attnRow(tk, tk.indexOf("شیر"), q);
      return { z: r.z, got: senseOf(r.z), want: x.sense, a: r.a, tk };
    });
    const okAll = res.every((r) => r.got === r.want);
    question = (
      <>
        Design <Fa>شیر</Fa>'s query yourself: q = [a, b, c] is matched against every token's [meaning 1, meaning 2, content]. One query has to work in all three sentences. Each output must land in its sense region (milk, lion, tap) and at least {MIN_R} from the centre.
      </>
    );
    stage = <WordPlane xr={MR} yr={MR} ticks={0.5} extra={sectors} points={res.map((r, i) => outPt(r.z, "شیر in s" + (i + 1) + " (" + r.want + ")", SENSE_COL[r.want], SENSE_SHAPE[r.want]))} ariaLabel="Outputs of شیر in the three sentences" axisLabels={["meaning 1", "meaning 2"]} />;
    side = (
      <>
        <Panel edge={okAll ? THEME.ok : THEME.grid}>
          <Slider id="a4-qa" label="a: weight on meaning 1" value={qa} min={-3} max={3} stepSize={0.05} onChange={setQa} />
          <Slider id="a4-qb" label="b: weight on meaning 2" value={qb} min={-3} max={3} stepSize={0.05} onChange={setQb} />
          <Slider id="a4-qg" label="c: weight on content" value={qg} min={0} max={6} stepSize={0.05} onChange={setQg} />
          {res.map((r, i) => (
            <Goal key={i} met={r.got === r.want}>
              s{i + 1} «{SENTS[i].s}» reads as {r.want} (now: {r.got})
            </Goal>
          ))}
          <Hint
            hints={[
              "The starting query looks for food (a and b negative). That finds صبحانه in s1, but what is there to find in s2 and s3?",
              "Don't ask for a meaning. Ask for content: set a and b near 0 and raise c. The values then hand over whatever meaning the content words carry.",
            ]}
          />
        </Panel>
        <Note>This is why Q, K and V are separate projections (slide 37): learn what to ask, how to match, and what information to combine.</Note>
      </>
    );
  }

  return <LessonFrame screen={screen} onScreen={setScreen} title="Every token asks, every token answers" question={question} stage={stage} side={side} />;
}
