import { useState, useCallback } from "react";
import { THEME, MONO } from "./common";

// Lesson 7 - Reuse BERT: pick the vectors, add a head (Pretraining, slides 76-89).
// Aha: one pretrained encoder serves many tasks. What changes per task is the input format (one text or a
// pair), which output vectors feed the new head ([CLS] for a label of the whole input, every token vector
// for per-token labels or spans), and the head's output size. The head is tiny; the pretrained start is what
// buys the accuracy (68.1 -> 84.9 on MNLI-m with the same network).

const D = 768;
function NOOP() {}
const TASKS = {
  sst: { name: "SST-2 sentiment", input: ["[CLS]", "a", "warm", ",", "funny", "film", "[SEP]"], labels: ["positive", "negative"], reads: "cls", C: 2, gold: "positive", slide: "77" },
  cola: { name: "CoLA acceptability", input: ["[CLS]", "the", "cats", "sleeps", "[SEP]"], labels: ["acceptable", "unacceptable"], reads: "cls", C: 2, gold: "unacceptable", slide: "77" },
  mnli: { name: "MNLI inference", input: ["[CLS]", "a", "woman", "is", "running", "[SEP]", "someone", "is", "moving", "[SEP]"], labels: ["entailment", "neutral", "contradiction"], reads: "cls", C: 3, gold: "entailment", slide: "78" },
  sts: { name: "STS-B similarity", input: ["[CLS]", "a", "cat", "is", "sleeping", "[SEP]", "a", "dog", "is", "sleeping", "[SEP]"], labels: ["score 0–5"], reads: "cls", C: 1, gold: "3.2 (regression)", slide: "78" },
  ner: { name: "Person names (NER)", input: ["[CLS]", "Ali", "met", "Sara", "in", "Tehran", "[SEP]"], labels: ["PER", "LOC", "O"], reads: "all", C: 3, gold: "Ali, Sara = PER", slide: "79" },
};
const NER_GOLD = { Ali: "PER", met: "O", Sara: "PER", in: "O", Tehran: "LOC" };

function VecRow({ input, routed, onPick, outputs }) {
  return (
    <div style={{ overflowX: "auto" }}>
      <div style={{ display: "flex", gap: 5, minWidth: "min-content" }}>
        {input.map((w, i) => (
          <VecCell key={i} i={i} w={w} on={!!routed[i]} onPick={onPick} out={outputs ? outputs[i] : null} />
        ))}
      </div>
    </div>
  );
}
function VecCell({ i, w, on, onPick, out }) {
  const click = useCallback(() => onPick(i), [onPick, i]);
  const special = w[0] === "[";
  return (
    <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 3, minWidth: 54 }}>
      <div style={{ fontFamily: MONO, fontSize: 11.5, height: 16, color: out ? THEME.highlight : THEME.muted }}>{out || ""}</div>
      <button
        onClick={click}
        aria-pressed={on}
        title="route this output vector to the head"
        style={{ width: 46, height: 30, borderRadius: 4, cursor: "pointer", border: "1px " + (on ? "solid " + THEME.highlight : "dashed " + THEME.grid), background: on ? rgba(THEME.highlight, 0.18) : THEME.bg, color: THEME.text, fontFamily: MONO, fontSize: 11 }}
      >
        {on ? "◆ h" : "◇ h"}
      </button>
      <div style={{ width: 2, height: 10, background: THEME.grid }} />
      <div style={{ fontFamily: MONO, fontSize: 12.5, padding: "2px 6px", border: "1px solid " + THEME.grid, borderRadius: 3, color: special ? THEME.muted : THEME.text }}>{w}</div>
    </div>
  );
}
function EncoderBar() {
  return <div style={{ margin: "4px 0", padding: "6px 0", textAlign: "center", fontFamily: MONO, fontSize: 12, color: THEME.muted, border: "1px solid " + THEME.accent, borderRadius: 5, background: THEME.surface }}>pretrained BERT encoder (bidirectional): one output vector h ∈ R⁷⁶⁸ per input token</div>;
}

// challenge: extractive QA, toy head scores for each configuration
const QA_PAIR = ["[CLS]", "who", "did", "sara", "meet", "?", "[SEP]", "sara", "met", "ali", "rezaei", "in", "shiraz", ".", "[SEP]"];
const QA_SINGLE = ["[CLS]", "sara", "met", "ali", "rezaei", "in", "shiraz", ".", "[SEP]"];
function qaOutcome(fmtSel, reads, C, norm) {
  const toks = fmtSel === "pair" ? QA_PAIR : QA_SINGLE;
  const pStart = fmtSel === "pair" ? "ali" : "sara";
  const pEnd = fmtSel === "pair" ? "rezaei" : "sara";
  const span = (a, b) => {
    const i = toks.lastIndexOf(a);
    const j = toks.lastIndexOf(b);
    return toks.slice(i, j + 1).join(" ");
  };
  if (reads === "cls") {
    if (norm === "pos") return { ok: false, text: "One vector gives nothing to choose among positions: a softmax over positions needs one score per token." };
    if (C === 1) return { ok: false, text: "A softmax over one class is always 1. The head says nothing." };
    return { ok: false, text: "The head outputs one label for the whole input (like QNLI's \"contains the answer\", slide 78). It cannot point at the words of the answer." };
  }
  if (norm === "pos") {
    if (C === 1) return { ok: false, answer: pStart, text: "One distribution over positions picks a single token. The answer can be longer than one token." };
    if (C === 2) return { ok: true, answer: span(pStart, pEnd), text: "Two scores per token: a start distribution and an end distribution over positions. The answer is the span from the best start to the best end." };
    return { ok: false, text: "Three distributions over positions: start, end and a third one that nothing defines." };
  }
  if (C === 1) return { ok: false, text: "A softmax over one class is always 1. Every token gets the same tag." };
  const tagged = fmtSel === "pair" ? "ali rezaei" : "sara";
  return { ok: true, answer: tagged, text: C === 2 ? "Two classes per token (in the answer / not): the tagged tokens form the answer." : "Three classes per token (begin / inside / outside): the B and I tokens form the answer." };
}

export default function FinetuneLesson({ initialScreen }) {
  const [screen, setScreen] = useState(initialScreen || 0);
  const [routeP, setRouteP] = useState({ 0: true });
  const [nerC, setNerC] = useState(2);
  const [task, setTask] = useState("sst");
  const [reads, setReads] = useState("cls");
  const [C, setC] = useState(2);
  const [done, setDone] = useState({});
  const [init, setInit] = useState("pre");
  const [qf, setQf] = useState("single");
  const [qr, setQr] = useState("cls");
  const [qc, setQc] = useState(2);
  const [qn, setQn] = useState("cls");

  const flipP = useCallback((i) => setRouteP((R) => Object.assign({}, R, { [i]: !R[i] })), []);
  const pickTask = useCallback((t) => setTask(t), []);
  const checkTask = useCallback(
    (r, c) => {
      const T = TASKS[task];
      if (r === T.reads && c === T.C) setDone((D0) => Object.assign({}, D0, { [task]: true }));
    },
    [task]
  );
  const pickReads = useCallback(
    (r) => {
      setReads(r);
      checkTask(r, C);
    },
    [checkTask, C]
  );
  const pickC = useCallback(
    (c) => {
      setC(c);
      checkTask(reads, c);
    },
    [checkTask, reads]
  );

  let question, stage, side;
  if (screen === 0) {
    const T = TASKS.ner;
    const outs = T.input.map((w, i) => {
      if (!routeP[i]) return null;
      if (i === 0) return nerC === 3 ? "1 label?" : "has PER: yes";
      if (w[0] === "[") return "—";
      if (nerC < 3) return NER_GOLD[w] === "PER" ? "PER" : "not PER";
      return NER_GOLD[w];
    });
    const tokIdx = [1, 2, 3, 4, 5];
    const allTok = tokIdx.every((i) => routeP[i]);
    const g1 = allTok;
    const g2 = allTok && nerC === 3;
    const g3 = !routeP[0] && !routeP[6];
    question = (
      <>
        <b>Where we are:</b> lesson 6 pretrained an encoder with MLM and NSP. Now we keep the encoder, drop the pretraining heads and add a new head for our task (slides 76, 66). <b>The puzzle (slide 79):</b> mark the person names in <span style={{ fontFamily: MONO }}>Ali met Sara in Tehran</span>. Which output vectors should feed the head? Click vectors to route them, and choose how many labels the head outputs.
      </>
    );
    stage = (
      <>
        <VecRow input={T.input} routed={routeP} onPick={flipP} outputs={outs} />
        <EncoderBar />
        <Note>Above each routed vector: what the head predicts from it (a toy, already-trained head).</Note>
      </>
    );
    side = (
      <>
        <Panel>
          <Label>head outputs per vector</Label>
          <Pills
            options={[
              [2, "2: PER / not"],
              [3, "3: PER / LOC / O"],
            ]}
            value={nerC}
            onPick={setNerC}
          />
          <Goal met={g1}>every word gets its own prediction</Goal>
          <Goal met={g2}>Tehran is labeled as a place, not lumped with "met"</Goal>
          <Goal met={g3}>no vector routed that has no word to label</Goal>
        </Panel>
        <Note>The [CLS] vector summarizes the whole input; one head on it gives one answer per sentence. Labels align with WordPiece tokens: if Tehran were split into two pieces, a common choice is to label the first piece.</Note>
      </>
    );
  } else if (screen === 1) {
    const T = TASKS[task];
    const shapeOk = reads === T.reads && C === T.C;
    const nVec = reads === "cls" ? 1 : T.input.filter((w) => w[0] !== "[").length;
    const params = D * C + C;
    question = (
      <>
        <b>Claim to test:</b> one encoder can serve sentiment, acceptability, inference, similarity and tagging, and only the input format and the head change (slides 77–79). For each task, choose which vectors the head reads and how many outputs it has. When the head's output matches the task's label, the task is wired. How big is the head compared with BERT-base's 110M parameters?
      </>
    );
    stage = (
      <>
        <Pills options={Object.keys(TASKS).map((k) => [k, (done[k] ? "◆ " : "◇ ") + TASKS[k].name])} value={task} onPick={pickTask} />
        <VecRow input={T.input} routed={reads === "cls" ? { 0: true } : T.input.reduce((o, w, i) => Object.assign(o, w[0] === "[" ? {} : { [i]: true }), {})} onPick={NOOP} outputs={null} />
        <EncoderBar />
        <Panel edge={shapeOk ? THEME.ok : THEME.grid}>
          <Readout
            rows={[
              ["task labels", T.labels.join(" / ")],
              ["example gold label", T.gold],
              ["head output", nVec + " vector" + (nVec > 1 ? "s" : "") + " × " + C + (C === 1 ? " number" : " scores")],
              ["matches the task", shapeOk ? "yes" : "no", shapeOk ? THEME.ok : THEME.warn],
            ]}
          />
        </Panel>
      </>
    );
    side = (
      <>
        <Panel>
          <Label>head reads</Label>
          <Pills
            options={[
              ["cls", "[CLS] vector only"],
              ["all", "every token vector"],
            ]}
            value={reads}
            onPick={pickReads}
          />
          <Label>outputs per vector</Label>
          <Pills
            options={[
              [1, "1 (a number)"],
              [2, "2"],
              [3, "3"],
            ]}
            value={C}
            onPick={pickC}
          />
          <Readout
            rows={[
              ["head parameters 768·C + C", params.toLocaleString("en-US")],
              ["share of BERT-base (110M)", fmt((100 * params) / 110e6, 4) + "%"],
              ["tasks wired", Object.keys(done).length + " / 5"],
            ]}
          />
        </Panel>
        <Note>Pairs share one input: [CLS] A [SEP] B [SEP], with segment embeddings telling A from B (slides 65, 78). Fine-tuning usually updates the encoder and the head; training the head only is cheaper, but then the [CLS] vector itself cannot adapt (slides 66, 84).</Note>
      </>
    );
  } else if (screen === 2) {
    const rows = [
      ["MNLI-m", 68.1, 84.9],
      ["SST-2", 81.4, 93.2],
      ["RTE", 54.9, 72.2],
    ];
    question = (
      <>
        <b>What you found:</b> a whole-input label reads [CLS], a per-token label reads every token vector, and the head is a single linear layer of a few thousand numbers. So the head cannot be where the skill comes from. Did pretraining help? Compare the same encoder and head, started from random weights or from pretrained ones (slides 80–81).
      </>
    );
    stage = (
      <>
        <Eq size={13}>
          whole input: p = softmax(W h_[CLS] + b), W ∈ R^(C×768)
          <br />
          per token i: pᵢ = softmax(W hᵢ + b)
          <br />
          loss: cross-entropy, averaged over labeled tokens
        </Eq>
        <Label>GLUE dev accuracy %, full-precision BERT (Ardakani 2022, Table 1)</Label>
        <Bars rows={rows.map((r) => ({ label: r[0], value: init === "pre" ? r[2] : r[1], color: init === "pre" ? THEME.accent : THEME.warn, mark: init === "pre" ? "(+" + fmt(r[2] - r[1], 1) + ")" : "" }))} range={[0, 100]} digits={1} labelW={80} ariaLabel="random vs pretrained initialization" />
        <Label>Does BERT need NSP? MNLI-m dev accuracy % (slide 83)</Label>
        <Bars
          rows={[
            { label: "BERT: MLM + NSP", value: 84.4, color: THEME.accent },
            { label: "BERT: MLM only", value: 83.9, color: THEME.accent },
            { label: "RoBERTa: pairs + NSP", value: 84.0, color: THEME.third },
            { label: "RoBERTa: doc text, no NSP", value: 84.7, color: THEME.third },
          ]}
          range={[80, 86]}
          digits={1}
          labelW={190}
          ariaLabel="NSP ablations"
        />
      </>
    );
    side = (
      <>
        <Panel>
          <Label>starting weights</Label>
          <Pills
            options={[
              ["rand", "random"],
              ["pre", "pretrained"],
            ]}
            value={init}
            onPick={setInit}
          />
        </Panel>
        <Note>Same architecture, tokenizer and splits; only the starting point differs, and pretraining compute is a separate cost (slide 80). NSP: in BERT's own ablation, dropping it cost 0.5; RoBERTa dropped NSP while also changing how inputs are packed, and gained 0.7. Compare rows only within each study. RoBERTa also changed data, batch size, dynamic masking and the tokenizer (slide 85).</Note>
        <Note>Can this encoder write a summary? It returns N vectors for N input tokens; a summary needs M new tokens and a rule for the first token, the next one and when to stop (slides 87–89). That generation process is the GPT Lab's subject.</Note>
      </>
    );
  } else {
    const out = qaOutcome(qf, qr, qc, qn);
    const win = out.ok && out.answer === "ali rezaei";
    const toks = qf === "pair" ? QA_PAIR : QA_SINGLE;
    const routed = qr === "cls" ? { 0: true } : toks.reduce((o, w, i) => Object.assign(o, w[0] === "[" ? {} : { [i]: true }), {});
    const ans = out.answer ? out.answer.split(" ") : [];
    const outs = toks.map((w, i) => (ans.indexOf(w) >= 0 && (qf === "single" || i > 6) ? "◀ answer" : null));
    question = (
      <>
        A new task: extractive question answering. Given <span style={{ fontFamily: MONO }}>who did sara meet?</span> and the passage <span style={{ fontFamily: MONO }}>sara met ali rezaei in shiraz.</span>, the answer must be copied out of the passage: <span style={{ fontFamily: MONO }}>ali rezaei</span>. Design the input and the head so that the fine-tuned model returns exactly that span.
      </>
    );
    stage = (
      <>
        <VecRow input={toks} routed={routed} onPick={NOOP} outputs={outs} />
        <EncoderBar />
        <Panel edge={win ? THEME.ok : THEME.grid}>
          <div style={{ fontSize: 14, lineHeight: 1.5 }}>{out.text}</div>
          <Readout rows={[["predicted answer", out.answer || "—", win ? THEME.ok : THEME.text]]} />
        </Panel>
      </>
    );
    side = (
      <>
        <Panel>
          <Label>input</Label>
          <Pills
            options={[
              ["single", "passage only"],
              ["pair", "question [SEP] passage"],
            ]}
            value={qf}
            onPick={setQf}
          />
          <Label>head reads</Label>
          <Pills
            options={[
              ["cls", "[CLS] only"],
              ["all", "every token vector"],
            ]}
            value={qr}
            onPick={setQr}
          />
          <Label>outputs per vector</Label>
          <Pills
            options={[
              [1, "1"],
              [2, "2"],
              [3, "3"],
            ]}
            value={qc}
            onPick={setQc}
          />
          <Label>softmax across</Label>
          <Pills
            options={[
              ["cls", "the outputs of each vector"],
              ["pos", "positions (for each output)"],
            ]}
            value={qn}
            onPick={setQn}
          />
          <Goal met={win}>the model returns exactly "ali rezaei"</Goal>
          <Hint hints={["Without the question in the input, what would tell the model which name to copy?", "The answer is a span of passage tokens, so the head must score each token. One score per token can point at only one position. What do you need to mark where the span starts and where it ends?"]} />
        </Panel>
        <Note>Toy head outputs. BERT's SQuAD head is the start/end design: two vectors S and E, a softmax over passage positions for each (Devlin et al. 2019, §4.2).</Note>
      </>
    );
  }

  return <LessonFrame screen={screen} onScreen={setScreen} title="Reuse BERT: pick the vectors, add a head" question={question} stage={stage} side={side} />;
}
