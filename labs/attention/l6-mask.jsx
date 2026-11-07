import { useState, useCallback } from "react";
import { THEME, MONO } from "./common";

// Lesson 6 - Don't peek at the answer (Attention, slides 48-55).
// Aha: with teacher forcing, the decoder's inputs are the gold target shifted right, and every position is
// trained at once. If a position can attend to a later input, it can see the very token it must predict and
// copy it, so the training loss looks perfect while generation fails. Adding −∞ to the scores of every
// later key (the causal mask) makes their weight exactly 0.

const IN4 = ["<BOS>", "من", "چای", "می‌نوشم"];
const OUT4 = ["من", "چای", "می‌نوشم", "<EOS>"];
const IN5 = ["<BOS>", "من", "هر", "روز", "چای"];
const OUT5 = ["من", "هر", "روز", "چای", "می‌نوشم"];
const HONEST = 0.6;
const COPY = 0.99;

// a row is "leaky" if it can see a key after its own position: key i + 1 holds the token it must predict
function analyse(vis) {
  const n = vis.length;
  return vis.map((row, i) => {
    const any = row.some(Boolean);
    const leak = row.some((v, j) => v && j > i);
    const sawPast = row.every((v, j) => (j <= i ? v : true));
    const train = !any ? NaN : leak ? -Math.log(COPY) : -Math.log(HONEST);
    return { any, leak, sawPast, train, inferOk: any && !leak };
  });
}
function grid(n, fn) {
  const out = [];
  for (let i = 0; i < n; i++) {
    const r = [];
    for (let j = 0; j < n; j++) r.push(fn(i, j));
    out.push(r);
  }
  return out;
}
function MaskGrid({ rows, cols, vis, onCell, rowNote }) {
  const res = analyse(vis);
  return (
    <Matrix
      rowLabels={rows.map((r, i) => (rowNote ? rowNote(r, i) : r))}
      colLabels={cols}
      values={vis.map((r) => r.map((v) => (v ? 1 : 0)))}
      shade={(v, i, j) => (v ? (j > i ? 0.75 : 0.45) : 0)}
      colorFor={(v, i, j) => (j > i ? THEME.warn : THEME.accent)}
      text={(v, i, j) => (v ? (j > i ? "✓!" : "✓") : "×")}
      mark={(v, i, j) => (j === i + 1 && v ? "★" : null)}
      onCell={onCell}
      faRows
      faCols
      cell={56}
      rowW={rowNote ? 128 : 84}
      colH={32}
      ariaLabel="mask grid: rows are queries, columns are keys; click to show or hide"
    />
  );
}

export default function MaskLesson({ initialScreen }) {
  const [screen, setScreen] = useState(initialScreen || 0);
  const [vis, setVis] = useState(grid(4, () => true));
  const [sc, setSc] = useState([1, 2, 3, 4]);
  const [mval, setMval] = useState("inf");
  const [n, setN] = useState(6);
  const [vis5, setVis5] = useState(grid(5, (i, j) => j < i));

  const flip = useCallback((i, j) => setVis((V) => V.map((r, a) => (a === i ? r.map((v, b) => (b === j ? !v : v)) : r))), []);
  const flip5 = useCallback((i, j) => setVis5((V) => V.map((r, a) => (a === i ? r.map((v, b) => (b === j ? !v : v)) : r))), []);
  const setS = useCallback((k, v) => setSc((S) => S.map((x, i) => (i === k ? v : x))), []);
  const setS0 = useCallback((v) => setS(0, v), [setS]);
  const setS1 = useCallback((v) => setS(1, v), [setS]);
  const setS2 = useCallback((v) => setS(2, v), [setS]);
  const setS3 = useCallback((v) => setS(3, v), [setS]);
  const resetAll = useCallback(() => setVis(grid(4, () => true)), []);

  let question, stage, side;
  if (screen === 0) {
    const res = analyse(vis);
    const trainLoss = res.every((r) => r.any) ? mean(res.map((r) => r.train)) : NaN;
    const inferOk = res.filter((r) => r.inferOk).length;
    question = (
      <>
        <b>Where we are:</b> lessons 4–5 built the encoder. The decoder generates one token at a time, but in training it gets the gold target shifted right (<Fa>{"<BOS> من چای می‌نوشم"}</Fa>), and all positions are processed in one pass (teacher forcing, slide 51). <b>The puzzle:</b> right now every position may attend to every input. Training loss looks great. Why does generation then fail? Click cells to hide query–key pairs until training and generation agree.
      </>
    );
    stage = (
      <>
        <MaskGrid rows={IN4} cols={IN4} vis={vis} onCell={flip} />
        <Note>Rows: decoder inputs as queries. Columns: the same inputs as keys. Each row must predict the next token, so ★ marks the key that holds that row's own answer. The last row predicts &lt;EOS&gt;, which is not an input.</Note>
      </>
    );
    side = (
      <>
        <Panel>
          <Readout
            rows={[
              ["training loss (average)", isFinite(trainLoss) ? fmt(trainLoss, 3) : "NaN: a row sees nothing", res.some((r) => r.leak) ? THEME.warn : THEME.text],
              ["positions that can copy their answer", String(res.filter((r) => r.leak).length) + " / 4"],
              ["positions that work at generation", inferOk + " / 4", inferOk === 4 ? THEME.ok : THEME.warn],
            ]}
          />
          <Goal met={res.every((r) => !r.leak)}>no position can see its own answer</Goal>
          <Goal met={res.every((r) => r.any && r.sawPast)}>every position still sees all earlier inputs</Goal>
          <Btn onClick={resetAll}>show everything again</Btn>
        </Panel>
        <Note>At generation time the future tokens do not exist yet (slides 49–50). A model trained to copy them has nothing to copy. A copying position gets loss ≈ {fmt(-Math.log(COPY), 2)}, an honest one ≈ {fmt(-Math.log(HONEST), 2)} (made-up numbers).</Note>
      </>
    );
  } else if (screen === 1) {
    const M = [0, 0, mval === "inf" ? -Infinity : -parseFloat(mval), mval === "inf" ? -Infinity : -parseFloat(mval)];
    const masked = sc.map((s, j) => s + M[j]);
    const w = softmax(masked);
    const leak = w[2] + w[3];
    question = (
      <>
        <b>Claim to test:</b> the mask is not a separate mechanism. It is a number added to the scores before the softmax: 0 where a key is allowed, −∞ where it is hidden (slides 54–55). Here query <Fa>من</Fa> predicts <Fa>چای</Fa>, so the keys <Fa>چای</Fa> and <Fa>می‌نوشم</Fa> must get zero weight. Change the scores and the mask value. Why −∞, and not just a large negative number?
      </>
    );
    stage = (
      <>
        <Matrix
          rowLabels={["scaled scores", "+ causal mask", "masked scores", "softmax weights"]}
          colLabels={IN4}
          values={[sc, M, masked, w]}
          shade={(v, i) => (i === 3 ? v : 0)}
          text={(v, i) => (i === 3 ? fmt(v, 3) : fmt(v, 1))}
          colorFor={(v, i, j) => (j >= 2 ? THEME.warn : THEME.accent)}
          faCols
          cell={64}
          rowW={120}
          colH={32}
          ariaLabel="slide 54: scores, mask, masked scores and weights"
        />
      </>
    );
    side = (
      <>
        <Panel>
          <Slider id="a6-s0" label="score for <BOS>" value={sc[0]} min={-4} max={10} stepSize={0.1} onChange={setS0} />
          <Slider id="a6-s1" label="score for من" value={sc[1]} min={-4} max={10} stepSize={0.1} onChange={setS1} />
          <Slider id="a6-s2" label="score for چای (future)" value={sc[2]} min={-4} max={10} stepSize={0.1} onChange={setS2} />
          <Slider id="a6-s3" label="score for می‌نوشم (future)" value={sc[3]} min={-4} max={10} stepSize={0.1} onChange={setS3} />
          <Label>value added to hidden keys</Label>
          <Pills
            options={[
              ["0", "0 (no mask)"],
              ["1", "−1"],
              ["5", "−5"],
              ["20", "−20"],
              ["inf", "−∞"],
            ]}
            value={mval}
            onPick={setMval}
          />
          <Readout rows={[["weight that leaks to the future", fmt(leak, 6), leak > 1e-3 ? THEME.warn : THEME.ok]]} />
        </Panel>
        <Note>Slide 54's row is scores [1, 2, 3, 4], giving weights [0.269, 0.731, 0, 0]. With a finite mask, a large enough future score still leaks. Raise چای to 10 with −5 and see.</Note>
      </>
    );
  } else if (screen === 2) {
    const M = grid(Math.min(n, 8), (i, j) => (j <= i ? 0 : -Infinity));
    question = (
      <>
        <b>What you found:</b> teacher forcing trains every position in one pass. The causal mask keeps that honest: each query sees itself and earlier keys, and every later key gets −∞, so its weight is exactly 0. Training is then parallel across positions, while generation stays one token at a time (slides 51–55, 60). Change the target length n.
      </>
    );
    stage = (
      <>
        <Matrix rowLabels={M.map((_, i) => "q" + (i + 1))} colLabels={M.map((_, j) => "k" + (j + 1))} values={M} shade={(v) => (v === 0 ? 0.35 : 0)} text={(v) => (v === 0 ? "0" : "−∞")} cell={44} rowW={44} colH={26} ariaLabel="causal mask matrix" />
        <Eq size={12.5}>
          Attention(Q, K, V) = softmax(QKᵀ / √d_k + M) V
          <br />
          M_ij = 0 if j ≤ i, −∞ if j &gt; i (i: query position, j: key position)
        </Eq>
      </>
    );
    side = (
      <>
        <Panel>
          <Slider id="a6-n" label="target length n" value={n} min={2} max={64} stepSize={1} onChange={setN} display={"n = " + n} />
          <Readout
            rows={[
              ["visible query–key pairs", String((n * (n + 1)) / 2) + " of " + n * n],
              ["training: decoder passes", "1 (all positions at once)"],
              ["generation: decoder passes", n + " (one per token)"],
            ]}
          />
        </Panel>
        <Note>The decoder also has cross-attention: its queries come from the decoder, keys and values from the encoder outputs, which are computed once and reused at every step (slides 49–50). The mask applies only to the decoder's self-attention. Grid shown up to 8×8.</Note>
      </>
    );
  } else {
    const res = analyse(vis5);
    const okLeak = res.every((r) => !r.leak);
    const okEmpty = res.every((r) => r.any);
    const okPast = res.every((r) => r.sawPast);
    question = (
      <>
        A new sentence: <Fa>من هر روز چای می‌نوشم</Fa> (I drink tea every day). Each row says which input a position reads and which token it must predict. Someone has set up the mask below. Fix it, so that no position sees its own answer, no row is left with nothing to attend to, and every position sees all the inputs it will have at generation time.
      </>
    );
    stage = (
      <>
        <MaskGrid rows={IN5} cols={IN5} vis={vis5} onCell={flip5} />
        <Note>✓ visible · × hidden · ✓! visible but in the future · ★ the key holding the token this row must predict. The last row predicts می‌نوشم, which is not an input.</Note>
      </>
    );
    side = (
      <>
        <Panel edge={okLeak && okEmpty && okPast ? THEME.ok : THEME.grid}>
          <Readout
            rows={[
              ["rows that would be NaN", res.filter((r) => !r.any).map((r, i) => i).length ? IN5.filter((_, i) => !res[i].any).join(", ") : "none", okEmpty ? THEME.text : THEME.warn],
              ["rows that can copy their answer", res.filter((r) => r.leak).length ? IN5.filter((_, i) => res[i].leak).join(", ") : "none", okLeak ? THEME.text : THEME.warn],
            ]}
          />
          <Goal met={okLeak}>no position sees its own answer</Goal>
          <Goal met={okEmpty}>no row is empty (softmax over only −∞ is NaN)</Goal>
          <Goal met={okPast}>every position sees every input up to and including its own</Goal>
          <Hint
            hints={[
              "Look at the first row: <BOS> must predict من. Which keys exist when it is generated?",
              "A position may see its own input, because that input is the previous token, not its answer. Its answer sits one column to the right (★). So the diagonal stays visible.",
            ]}
          />
        </Panel>
      </>
    );
  }

  return <LessonFrame screen={screen} onScreen={setScreen} title="Don't peek at the answer" question={question} stage={stage} side={side} />;
}
