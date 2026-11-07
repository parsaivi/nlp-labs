import { useState, useMemo, useCallback } from "react";
import { THEME, MONO } from "./common";

// Lesson 5 - Relations are directions (Word Embedding, slides 20-22, 29-30).
// Aha: in a good embedding, one relation (male → female, capital → country, base → comparative) is roughly
// the same arrow for every pair. So b − a + c lands near the missing word d, and "solving an analogy"
// becomes vector arithmetic plus a nearest-neighbour search that skips the three input words.

// @sim-start
const SETS = {
  gender: {
    name: "man → woman",
    pairs: [
      ["man", "woman"],
      ["king", "queen"],
      ["boy", "girl"],
      ["uncle", "aunt"],
    ],
    v: { man: [-2.0, -1.1], woman: [-1.3, -0.7], king: [0.3, 1.3], queen: [1.6, 1.1], boy: [-2.6, 1.0], girl: [-1.8, 1.5], uncle: [1.5, -1.5], aunt: [2.3, -1.0] },
  },
  capital: {
    name: "capital → country",
    pairs: [
      ["Athens", "Greece"],
      ["Paris", "France"],
      ["Tehran", "Iran"],
      ["Oslo", "Norway"],
    ],
    v: { Athens: [-2.4, -0.6], Greece: [-1.7, 1.1], Paris: [-0.6, -1.6], France: [0.2, 0.0], Tehran: [1.0, -0.4], Iran: [1.6, 1.4], Oslo: [2.6, -1.5], Norway: [3.3, 0.2] },
  },
  comparative: {
    name: "adjective → comparative",
    pairs: [
      ["big", "bigger"],
      ["small", "smaller"],
      ["fast", "faster"],
      ["cold", "colder"],
    ],
    v: { big: [-2.0, 1.4], bigger: [-0.8, 1.9], small: [-2.2, -1.2], smaller: [-1.0, -0.6], fast: [0.8, 1.0], faster: [2.0, 1.4], cold: [1.0, -1.6], colder: [2.1, -1.0] },
  },
};
const XR = [-3.5, 4.5];
const YR = [-2.5, 2.5];
function questions(pairs) {
  const qs = [];
  for (let i = 0; i < pairs.length; i++)
    for (let j = 0; j < pairs.length; j++) {
      if (i === j) continue;
      qs.push([pairs[i][0], pairs[i][1], pairs[j][0], pairs[j][1]]);
      qs.push([pairs[i][1], pairs[i][0], pairs[j][1], pairs[j][0]]);
    }
  return qs;
}
function query(v, a, b, c) {
  return [v[b][0] - v[a][0] + v[c][0], v[b][1] - v[a][1] + v[c][1]];
}
// ranked candidates for the query point q
function rank(v, q, skip, metric) {
  return Object.keys(v)
    .filter((w) => skip.indexOf(w) < 0)
    .map((w) => [w, metric === "cos" ? cosine(v[w], q) : -Math.hypot(v[w][0] - q[0], v[w][1] - q[1])])
    .sort((x, y) => y[1] - x[1]);
}
function solve(v, a, b, c, excl, metric) {
  return rank(v, query(v, a, b, c), excl ? [a, b, c] : [], metric)[0][0];
}
function accuracy(pairs, v, excl, metric) {
  const qs = questions(pairs);
  const wrong = qs.filter((q) => solve(v, q[0], q[1], q[2], excl, metric) !== q[3]);
  return { right: qs.length - wrong.length, total: qs.length, wrong };
}
// @sim-end

// Mikolov et al. 2013, Table 3 (slide 20): accuracy % on the analogy test set
const TABLE = [
  ["RNNLM", 9, 36],
  ["NNLM", 23, 53],
  ["CBOW", 24, 64],
  ["Skip-gram", 55, 59],
];
const START_C = { Norway: [-3.0, 2.0], Iran: [4.0, -2.0] };

function setPoints(set, v, roles, drag) {
  const male = set.pairs.map((p) => p[0]);
  return Object.keys(v).map((w) => {
    const first = male.indexOf(w) >= 0;
    const r = roles[w];
    return {
      id: w,
      label: w,
      x: v[w][0],
      y: v[w][1],
      color: r === "ans" ? THEME.ok : r ? THEME.text : first ? THEME.accent : THEME.highlight,
      shape: first ? "circle" : "square",
      r: r ? 8 : 6,
      drag: drag && drag.indexOf(w) >= 0,
    };
  });
}

export default function AnalogyLesson({ initialScreen }) {
  const [screen, setScreen] = useState(initialScreen || 0);
  const [setId, setSetId] = useState("gender");
  const [a, setA] = useState("man");
  const [b, setB] = useState("woman");
  const [c, setC] = useState("king");
  const [excl, setExcl] = useState(false);
  const [metric, setMetric] = useState("dist");
  const [edits, setEdits] = useState({});
  const [dragW, setDragW] = useState("queen");
  const [kind, setKind] = useState(1);
  const [cv, setCv] = useState(START_C);

  const set = SETS[setId];
  const v = useMemo(() => Object.assign({}, set.v, edits[setId] || {}), [set, edits, setId]);
  const words = Object.keys(set.v);
  const pickSet = useCallback((id) => {
    setSetId(id);
    const p = SETS[id].pairs;
    setA(p[0][0]);
    setB(p[0][1]);
    setC(p[1][0]);
    setDragW(p[1][1]);
  }, []);
  const onDrag = useCallback(
    (id, x, y) => {
      setEdits((E) => {
        const cur = Object.assign({}, E[setId] || {});
        cur[id] = [x, y];
        return Object.assign({}, E, { [setId]: cur });
      });
    },
    [setId]
  );
  const resetSet = useCallback(() => setEdits((E) => Object.assign({}, E, { [setId]: {} })), [setId]);
  const onDragC = useCallback((id, x, y) => {
    if (id === "Norway" || id === "Iran") setCv((C) => Object.assign({}, C, { [id]: [x, y] }));
  }, []);

  const q = query(v, a, b, c);
  const ranked = rank(v, q, excl ? [a, b, c] : [], metric);
  const ans = ranked[0][0];
  const roles = { [a]: "in", [b]: "in", [c]: "in", [ans]: "ans" };
  const analogyArrows = [
    { x1: v[a][0], y1: v[a][1], x2: v[b][0], y2: v[b][1], color: THEME.text, width: 2 },
    { x1: v[c][0], y1: v[c][1], x2: q[0], y2: q[1], color: THEME.third, width: 2, dash: "6 4" },
  ];
  const qMark = useCallback(
    (P) => {
      const p = P(q[0], q[1]);
      return (
        <g>
          <circle cx={p[0]} cy={p[1]} r={11} fill="none" stroke={THEME.third} strokeWidth={2} />
          <SvgLabel x={p[0] + 13} y={p[1] + 16} color={THEME.third}>
            q
          </SvgLabel>
        </g>
      );
    },
    [q]
  );
  const slotPills = (val, onPick) => <Pills options={words.map((w) => [w, w])} value={val} onPick={onPick} />;

  let question, stage, side;
  if (screen === 0) {
    question = (
      <>
        <b>Where we are:</b> lessons 3 and 4 trained vectors with Word2Vec. How do we know they are any good? Mikolov et al. tested them with analogies (slides 20–21): <i>man is to woman as king is to ?</i> Build the point q = king − man + woman (the teal dashed arrow copies the white arrow man → woman onto king) and take the closest word to it. <b>The puzzle:</b> the closest word comes out as <b>king</b>, one of the inputs. Why? And what is the closest word if the three inputs are not allowed as answers?
      </>
    );
    stage = <WordPlane xr={XR} yr={YR} points={setPoints(set, v, roles)} arrows={analogyArrows} top={qMark} ariaLabel="Analogy as a vector offset" axisLabels={["dim 1", "dim 2"]} />;
    side = (
      <>
        <Panel>
          <Label>a is to b</Label>
          {slotPills(a, setA)}
          {slotPills(b, setB)}
          <Label>as c is to ?</Label>
          {slotPills(c, setC)}
          <Toggle label="don't allow a, b, c as the answer" value={excl} onChange={setExcl} />
          <Readout
            rows={[
              ["q = " + b + " − " + a + " + " + c, "(" + fmt(q[0], 2) + ", " + fmt(q[1], 2) + ")"],
              ["closest word to q", ans, THEME.ok],
              ["2nd, 3rd", ranked[1][0] + ", " + ranked[2][0]],
            ]}
          />
        </Panel>
        <Note>● first word of each pair · ■ second word. The vectors are 2-D and hand-placed. Real offsets are only approximate (slide 21), and that is exactly what makes king win here.</Note>
      </>
    );
  } else if (screen === 1) {
    const acc = accuracy(set.pairs, v, excl, metric);
    question = (
      <>
        <b>Claim to test:</b> an analogy works when the relation is the same arrow for every pair (slide 21). The evaluation asks every question a set allows, here 24, and counts exact matches. Drag a word away from its parallelogram and watch the accuracy. How far can it drift before answers break? Then switch the ranking from distance to cosine. Does the score change?
      </>
    );
    stage = <WordPlane xr={XR} yr={YR} points={setPoints(set, v, roles, [dragW])} arrows={analogyArrows} top={qMark} onDrag={onDrag} ariaLabel="Drag a word and watch the analogy accuracy" axisLabels={["dim 1", "dim 2"]} />;
    side = (
      <>
        <Panel>
          <Label>relation set</Label>
          <Pills
            options={[
              ["gender", "man → woman"],
              ["capital", "capital → country"],
              ["comparative", "big → bigger"],
            ]}
            value={setId}
            onPick={pickSet}
          />
          <Label>word to drag (dashed ring)</Label>
          {slotPills(dragW, setDragW)}
          <Pills
            options={[
              ["dist", "rank by distance"],
              ["cos", "rank by cosine"],
            ]}
            value={metric}
            onPick={setMetric}
          />
          <Toggle label="don't allow a, b, c as the answer" value={excl} onChange={setExcl} />
          <Readout
            rows={[
              ["questions right", acc.right + " / " + acc.total, acc.right === acc.total ? THEME.ok : THEME.text],
              ["current question", a + " : " + b + " :: " + c + " : " + ans],
              ["first wrong one", acc.wrong.length ? acc.wrong[0].slice(0, 3).join(" : ") + " → " + solve(v, acc.wrong[0][0], acc.wrong[0][1], acc.wrong[0][2], excl, metric) : "—"],
            ]}
          />
          <Btn onClick={resetSet}>reset this set</Btn>
        </Panel>
        <Note>Cosine compares directions from the origin and ignores length. In 2-D, words at similar angles collide, so the two rankings can disagree.</Note>
      </>
    );
  } else if (screen === 2) {
    question = (
      <>
        <b>What you found:</b> a relation is a direction. Adding b − a to c moves along it, and a nearest-neighbour search that skips a, b and c reads off the answer. Exact match is strict, so small errors in the offsets cost points. Mikolov et al. used 19,544 such questions, 8,869 semantic and 10,675 syntactic. Switch between the two kinds to compare models.
      </>
    );
    stage = (
      <>
        <WordPlane xr={XR} yr={YR} points={setPoints(set, v, roles)} arrows={analogyArrows} top={qMark} ariaLabel="Analogy as a vector offset" axisLabels={["dim 1", "dim 2"]} />
        <Label>accuracy (%), Mikolov et al. 2013, Table 3: 320M words, 640 dimensions</Label>
        <Bars rows={TABLE.map((r) => ({ label: r[0], value: r[kind], color: r[0] === "Skip-gram" || r[0] === "CBOW" ? THEME.accent : THEME.muted }))} range={[0, 100]} digits={0} labelW={90} ariaLabel="analogy accuracy by model" />
      </>
    );
    side = (
      <>
        <Panel>
          <Pills
            options={[
              [1, "semantic (Athens : Greece)"],
              [2, "syntactic (big : bigger)"],
            ]}
            value={kind}
            onPick={setKind}
          />
          <Eq size={12.5}>
            q = v_b − v_a + v_c
            <br />
            answer = argmax over w ∉ {"{a, b, c}"} of cos(v_w, q)
            <br />
            cos(u, v) = u·v / (‖u‖ ‖v‖)
          </Eq>
        </Panel>
        <Note>
          Slides 20–22 and 29. Analogies and similarity ratings are <i>intrinsic</i> tests: they inspect the vectors directly. Similarity tests compare cosines with human ratings using Spearman correlation. High intrinsic scores do not guarantee gains on a real task. <i>Extrinsic</i> tests put the vectors into a downstream classifier and measure accuracy or F1 under controlled conditions (slide 30).
        </Note>
      </>
    );
  } else {
    const S = SETS.capital;
    const vc = Object.assign({}, S.v, cv);
    const acc = accuracy(S.pairs, vc, true, "dist");
    const roleC = { Norway: "in", Iran: "in" };
    const pts = setPoints(S, vc, {}, ["Norway", "Iran"]).map((p) => (roleC[p.id] ? Object.assign({}, p, { color: THEME.third, label: p.id + " ?" }) : p));
    const okAll = acc.right === acc.total;
    question = (
      <>
        Two country vectors were lost: Norway and Iran sit in the wrong corners. Drag them back so that every capital → country question in the set is answered correctly (24 questions, answers ranked by distance, inputs excluded). This includes the reversed questions, such as Greece : Athens :: Norway : ?
      </>
    );
    stage = <WordPlane xr={XR} yr={YR} points={pts} onDrag={onDragC} ariaLabel="Place Norway and Iran" axisLabels={["dim 1", "dim 2"]} />;
    side = (
      <>
        <Panel edge={okAll ? THEME.ok : THEME.grid}>
          <Readout
            rows={[
              ["questions right", acc.right + " / " + acc.total, okAll ? THEME.ok : THEME.text],
              ["still wrong", acc.wrong.length ? acc.wrong.slice(0, 3).map((w) => w[0] + ":" + w[1] + "::" + w[2] + ":?").join("  ") : "—"],
            ]}
          />
          <Goal met={acc.wrong.every((w) => w.indexOf("Norway") < 0)}>every question involving Norway is right</Goal>
          <Goal met={acc.wrong.every((w) => w.indexOf("Iran") < 0)}>every question involving Iran is right</Goal>
          <Goal met={okAll}>all 24 right</Goal>
          <Hint
            hints={[
              "Look at the three pairs that are intact. Is the arrow from capital to country roughly the same each time?",
              "Put Norway at Oslo + (Greece − Athens), and Iran at Tehran + the same arrow.",
            ]}
          />
        </Panel>
        <Note>This is the parallelogram from slide 21, used in reverse: if the relation is one arrow, a missing word's position is fixed by its partner.</Note>
      </>
    );
  }

  return <LessonFrame screen={screen} onScreen={setScreen} title="Relations are directions" question={question} stage={stage} side={side} />;
}
