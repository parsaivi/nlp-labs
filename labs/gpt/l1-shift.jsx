import { useState, useCallback } from "react";
import { THEME, MONO } from "./common";

// Lesson 1 - One pass, a loss at every position (GPT, slides 15-25; Pretraining deck, slides 96-102).
// Aha: a sentence of N tokens holds N prefix -> next-token examples. Feeding them one cut at a time
// recomputes the same prefixes again and again. A causal Transformer reads the input once, compares each
// position's output with the input shifted one step left, and the causal mask keeps every position from
// seeing its own answer. So every token is a training signal, against BERT's 15%.

// @sim-start
const SENT = ["The", "cat", "sat", "on", "the", "mat"];
const SLIDE_P = [0.5, 0.25, 0.5, 0.5, 0.25, 0.5]; // Pretraining deck, slide 101
function meanCE(ps) {
  let s = 0;
  for (const p of ps) s -= Math.log(p);
  return s / ps.length;
}
// token-positions processed: one forward pass per cut vs one pass for all cuts
function workCuts(n) {
  return (n * (n - 1)) / 2;
}
function workOnePass(n) {
  return n - 1;
}
// @sim-end

const IN6 = ["the", "cat", "sat", "on", "the", "mat"];
const TG6 = ["cat", "sat", "on", "the", "mat", "[EOS]"];
const FA_TOKENS = ["<BOS>", "من", "هر", "روز", "چای", "می‌نوشم", "<EOS>"];
const GOLD_IN = ["<BOS>", "من", "هر", "روز", "چای", "می‌نوشم"];
const GOLD_TG = ["من", "هر", "روز", "چای", "می‌نوشم", "<EOS>"];

function CutRows({ n, onePass }) {
  const w = 600;
  const cw = Math.min(56, (w - 90) / n);
  const rows = onePass ? 1 : n - 1;
  const rh = 26;
  const H = rows * rh + 30;
  const words = [];
  for (let i = 0; i < n; i++) words.push(i < SENT.length ? SENT[i] : "w" + (i + 1));
  return (
    <svg viewBox={"0 0 " + w + " " + H} style={{ width: "100%", display: "block", background: THEME.bg, border: "1px solid " + THEME.grid, borderRadius: 6 }} role="img" aria-label="token positions processed per forward pass">
      {Array.from({ length: rows }).map((_, r) => {
        const len = onePass ? n - 1 : r + 1;
        return (
          <g key={r}>
            <SvgLabel x={8} y={20 + r * rh} size={10.5} color={THEME.muted}>
              {onePass ? "1 pass" : "pass " + (r + 1)}
            </SvgLabel>
            {Array.from({ length: len }).map((__, i) => {
              const redo = !onePass && i < r;
              return <rect key={i} x={64 + i * cw} y={8 + r * rh} width={cw - 3} height={rh - 8} rx={2} fill={redo ? rgba(THEME.warn, 0.35) : rgba(THEME.accent, 0.45)} stroke={redo ? THEME.warn : THEME.accent} strokeWidth={0.8} strokeDasharray={redo ? "3 2" : undefined} />;
            })}
            {onePass ? (
              Array.from({ length: len }).map((__, i) => (
                <SvgLabel key={"t" + i} x={64 + i * cw + cw / 2 - 1.5} y={rh + 20} anchor="middle" size={9.5} color={THEME.highlight}>
                  {"→" + words[i + 1]}
                </SvgLabel>
              ))
            ) : (
              <SvgLabel x={64 + len * cw + 4} y={20 + r * rh} size={10} color={THEME.highlight}>
                {"→ " + words[r + 1]}
              </SvgLabel>
            )}
          </g>
        );
      })}
    </svg>
  );
}
function ShiftView({ ps, leak }) {
  return (
    <div style={{ overflowX: "auto" }}>
      <div style={{ display: "grid", gridTemplateColumns: "80px repeat(6, minmax(52px, 1fr))", gap: 4, fontFamily: MONO, fontSize: 13, alignItems: "center", minWidth: 420 }}>
        <span style={{ color: THEME.muted, fontSize: 11 }}>P(target)</span>
        {ps.map((p, i) => (
          <span key={"p" + i} style={{ textAlign: "center", color: leak ? THEME.warn : THEME.text }}>
            {fmt(p, 2)}
          </span>
        ))}
        <span style={{ color: THEME.muted, fontSize: 11 }}>TARGET</span>
        {TG6.map((w, i) => (
          <span key={"t" + i} style={{ textAlign: "center", color: THEME.highlight, border: "1px solid " + THEME.grid, borderRadius: 3, padding: "2px 0" }}>
            {w}
          </span>
        ))}
        <span />
        {TG6.map((w, i) => (
          <span key={"a" + i} style={{ textAlign: "center", color: THEME.muted }}>
            ↑
          </span>
        ))}
        <span style={{ color: THEME.muted, fontSize: 11 }}>INPUT</span>
        {IN6.map((w, i) => (
          <span key={"i" + i} style={{ textAlign: "center", border: "1px solid " + THEME.grid, borderRadius: 3, padding: "2px 0" }}>
            {w}
          </span>
        ))}
      </div>
    </div>
  );
}
function Slot({ w }) {
  return <div style={{ minWidth: 62, minHeight: 30, border: "1px dashed " + THEME.grid, borderRadius: 4, display: "flex", alignItems: "center", justifyContent: "center", fontFamily: FA, fontSize: 15, background: THEME.bg }}>{w || ""}</div>;
}

export default function ShiftLesson({ initialScreen }) {
  const [screen, setScreen] = useState(initialScreen || 0);
  const [n, setN] = useState(6);
  const [one, setOne] = useState(false);
  const [maxN, setMaxN] = useState(6);
  const [ps, setPs] = useState(SLIDE_P.slice());
  const [leak, setLeak] = useState(false);
  const [T, setT] = useState(1024);
  const [row, setRow] = useState("in");
  const [inRow, setInRow] = useState([]);
  const [tgRow, setTgRow] = useState([]);

  const onN = useCallback((v) => {
    setN(v);
    setMaxN((m) => Math.max(m, v));
  }, []);
  const setI = useCallback((k, v) => setPs((P) => P.map((x, i) => (i === k ? v : x))), []);
  const setters = [useCallback((v) => setI(0, v), [setI]), useCallback((v) => setI(1, v), [setI]), useCallback((v) => setI(2, v), [setI]), useCallback((v) => setI(3, v), [setI]), useCallback((v) => setI(4, v), [setI]), useCallback((v) => setI(5, v), [setI])];
  const resetP = useCallback(() => setPs(SLIDE_P.slice()), []);
  const addTok = useCallback(
    (w) => {
      if (row === "in") setInRow((R) => (R.length < 6 ? R.concat([w]) : R));
      else setTgRow((R) => (R.length < 6 ? R.concat([w]) : R));
    },
    [row]
  );
  const undo = useCallback(() => {
    if (row === "in") setInRow((R) => R.slice(0, -1));
    else setTgRow((R) => R.slice(0, -1));
  }, [row]);
  const clearRows = useCallback(() => {
    setInRow([]);
    setTgRow([]);
  }, []);

  let question, stage, side;
  if (screen === 0) {
    const a = workCuts(n);
    const b = workOnePass(n);
    question = (
      <>
        <b>Where we are:</b> the Pretraining Lab ended with BERT, an encoder that cannot generate. GPT is a Transformer trained on the plain next-token game (slides 14–18). One sentence gives many examples: <span style={{ fontFamily: MONO }}>The → cat</span>, <span style={{ fontFamily: MONO }}>The cat → sat</span>, … (slide 19). <b>The puzzle:</b> if each cut is its own forward pass, how much of the work is repeated? Grow the sentence. Can one pass give every prediction (slide 20)?
      </>
    );
    stage = (
      <>
        <CutRows n={Math.min(n, 12)} onePass={one} />
        <Note>{one ? "One pass over the input. Each position's output predicts the token after it (→)." : "Each row is a separate forward pass over one prefix. Dashed red cells are positions already computed in an earlier pass."} {n > 12 ? "Drawn for the first 12 tokens; the counts use all " + n + "." : ""}</Note>
      </>
    );
    side = (
      <>
        <Panel>
          <Slider id="s1-n" label="sentence length N (tokens)" value={n} min={2} max={1024} stepSize={1} onChange={onN} display={"N = " + n} />
          <Toggle label="one pass, predict at every position" value={one} onChange={setOne} />
          <Readout
            rows={[
              ["training examples", String(n - 1)],
              ["positions processed, one pass per cut", a.toLocaleString("en-US")],
              ["positions processed, one pass for all", b.toLocaleString("en-US")],
              ["ratio", fmt(a / b, 1) + "×", THEME.highlight],
            ]}
          />
          <Goal met={one}>try one pass that predicts at every position</Goal>
          <Goal met={maxN >= 201}>find a length where cutting does at least 100× the work</Goal>
        </Panel>
        <Note>Cut-by-cut work is 1 + 2 + … + (N − 1) = N(N − 1)/2 positions, so it grows with N². GPT-1 trained on 512-token inputs (slide 42).</Note>
      </>
    );
  } else if (screen === 1) {
    const use = leak ? ps.map(() => 0.99) : ps;
    const H = meanCE(use);
    question = (
      <>
        <b>Claim to test:</b> one pass works if the targets are the input shifted one step left: position t reads tokens 1…t and is scored on token t + 1 (slide 101). But plain self-attention lets every position look at every other (slide 22). Change the probabilities, then remove the causal mask. Why does the loss collapse, and why is that bad news?
      </>
    );
    stage = (
      <>
        <ShiftView ps={use} leak={leak} />
        <Eq size={13.5}>
          mean cross-entropy = −(1/6) Σ ln P(target) = {fmt(H, 3)} {leak ? "(the model is copying)" : ""}
        </Eq>
        <Matrix
          rowLabels={IN6.map((w, i) => i + 1 + " " + w)}
          colLabels={IN6.map((w, i) => String(i + 1))}
          values={IN6.map((_, i) => IN6.map((__, j) => (leak || j <= i ? 0 : -Infinity)))}
          shade={(v, i, j) => (v === 0 ? (j > i ? 0.7 : 0.35) : 0)}
          colorFor={(v, i, j) => (j > i ? THEME.warn : THEME.accent)}
          text={(v) => (v === 0 ? "0" : "−∞")}
          mark={(v, i, j) => (j === i + 1 && v === 0 ? "★" : null)}
          cell={40}
          rowW={70}
          colH={24}
          ariaLabel="mask added to attention scores: rows are queries, columns are keys"
        />
        <Note>Mask M added to the attention scores (rows: queries, columns: keys). ★ marks the key that holds that row's own target.</Note>
      </>
    );
    side = (
      <>
        <Panel>
          <Toggle label="remove the causal mask" value={leak} onChange={setLeak} />
          {IN6.map((w, i) => (
            <Slider key={i} id={"s1-p" + i} label={"P(" + TG6[i] + " | " + IN6.slice(0, i + 1).join(" ") + ")"} value={ps[i]} min={0.01} max={1} stepSize={0.01} onChange={setters[i]} />
          ))}
          <Btn onClick={resetP}>slide 101 values</Btn>
        </Panel>
        <Note>Slide 101: P(target) = .50 .25 .50 .50 .25 .50 gives mean cross-entropy 0.924. Without the mask each position can read its target one key to the right, so a copying model scores near 0 in training, then fails at generation, when the next token does not exist yet (slides 22–23).</Note>
      </>
    );
  } else if (screen === 2) {
    question = (
      <>
        <b>What you found:</b> shifting the targets turns one forward pass into a prediction at every position, with the LM head's weights shared across positions (slide 21). The causal mask adds −∞ to every later key, in every layer and head, so each position sees itself and its past only (slides 23–24). Compare how many targets each sequence gives GPT and BERT.
      </>
    );
    stage = (
      <>
        <Eq size={13.5}>
          L = −(1/T) Σₜ ln P(x_(t+1) | x₁ … xₜ) · Attention = softmax(QKᵀ/√d_k + M) V
          <br />
          M_ij = 0 if j ≤ i, −∞ if j &gt; i
        </Eq>
        <Bars
          rows={[
            { label: "GPT: every position", value: T, color: THEME.accent },
            { label: "BERT: 15% selected", value: 0.15 * T, color: THEME.third },
          ]}
          range={[0, T]}
          digits={0}
          labelW={150}
          ariaLabel="training targets per sequence"
        />
      </>
    );
    side = (
      <>
        <Panel>
          <Slider id="s1-T" label="tokens per sequence T" value={T} min={16} max={2048} stepSize={16} onChange={setT} display={"T = " + T} />
          <Readout
            rows={[
              ["GPT targets per sequence", String(T)],
              ["BERT MLM targets per sequence", fmt(0.15 * T, 0)],
            ]}
          />
        </Panel>
        <Note>Is GPT an encoder or a decoder? A GPT block is a decoder block without cross-attention, or equally an encoder block whose self-attention gets the causal mask (slide 15). BERT sees both sides; GPT sees itself and the past (slide 25). For a downstream task, GPT-1 reads the final token's vector, the way BERT reads [CLS] (slides 16, 26).</Note>
      </>
    );
  } else {
    const okIn = inRow.join("|") === GOLD_IN.join("|");
    const okTg = tgRow.join("|") === GOLD_TG.join("|");
    const shiftOk = inRow.length === tgRow.length && inRow.length > 1 && inRow.slice(1).every((w, i) => tgRow[i] === w);
    question = (
      <>
        Build one training example for <Fa>من هر روز چای می‌نوشم</Fa> (I drink tea every day), with &lt;BOS&gt; and &lt;EOS&gt;. Fill the input row and the target row so that a single causal forward pass yields a prediction at every position, and every token of the sentence, including the end, is predicted once.
      </>
    );
    stage = (
      <>
        <Panel>
          <Label>input row (position 1 → 6, left to right)</Label>
          <div style={{ display: "flex", gap: 5, flexWrap: "wrap", direction: "ltr" }}>
            {[0, 1, 2, 3, 4, 5].map((i) => (
              <Slot key={i} w={inRow[i]} />
            ))}
          </div>
          <Label>target row</Label>
          <div style={{ display: "flex", gap: 5, flexWrap: "wrap", direction: "ltr" }}>
            {[0, 1, 2, 3, 4, 5].map((i) => (
              <Slot key={i} w={tgRow[i]} />
            ))}
          </div>
        </Panel>
        <Label>{"click a token to add it to the " + (row === "in" ? "input" : "target") + " row"}</Label>
        <div style={{ display: "flex", gap: 6, flexWrap: "wrap", direction: "ltr" }}>
          {FA_TOKENS.map((w) => (
            <Chip key={w} value={w} on={false} onPick={addTok}>
              {w}
            </Chip>
          ))}
        </div>
      </>
    );
    side = (
      <>
        <Panel edge={okIn && okTg ? THEME.ok : THEME.grid}>
          <Pills
            options={[
              ["in", "fill input row"],
              ["tg", "fill target row"],
            ]}
            value={row}
            onPick={setRow}
          />
          <div style={{ display: "flex", gap: 6, marginBottom: 6 }}>
            <Btn onClick={undo}>undo</Btn>
            <Btn onClick={clearRows}>clear both</Btn>
          </div>
          <Goal met={inRow.length === 6 && tgRow.length === 6}>both rows have 6 tokens</Goal>
          <Goal met={shiftOk}>each target is the next input (the shift)</Goal>
          <Goal met={okIn && okTg}>the sentence starts after &lt;BOS&gt; and every word and &lt;EOS&gt; is a target</Goal>
          <Hint hints={["Position 1 has read only the start symbol. What should it predict?", "The input row is <BOS> plus the sentence; the target row is the sentence plus <EOS>. Neither row contains both <BOS> and <EOS>."]} />
        </Panel>
        <Note>Positions run left to right here even though the words are Persian: position 1 is the first token read.</Note>
      </>
    );
  }

  return <LessonFrame screen={screen} onScreen={setScreen} title="One pass, a loss at every position" question={question} stage={stage} side={side} />;
}
