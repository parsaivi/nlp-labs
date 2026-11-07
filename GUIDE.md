# Guide: turning the next deck into a lab

You generate a lab with the **interactive-educator** skill (`/aha`), exactly as published.
This repo doesn't change the skill. It only gives the skill's output a place to live, a build, a check and a public page.
The pipeline is the same as in the Deep Learning Labs repo.

```
deck PDF ──/aha──▶ lesson .jsx files ──tools/build.js──▶ docs/<lab>/index.html ──git push──▶ GitHub Pages
```

## 0. One-time setup

- Start Claude Code **inside this folder** (`nlp-labs/`). Claude Code picks up the skill from
  `.claude/skills/interactive-educator/` (a byte-identical copy of the upstream skill, CC BY 4.0), so `/aha` works
  without installing anything.
- The slide PDFs stay one folder up (`../03-Attention.pdf` etc.) and are never committed.
- Needs `node` (for the build) and `google-chrome` (for screenshots and the interaction check).

## 1. Make the lab folder

```bash
node tools/new-lab.js transformers "Transformer Lab" "04-Transformers.pdf"
```

This creates `labs/<id>/lab.json` and copies `common.jsx` (the ink-notebook UI kit, vector math, `WordPlane`,
`Bars`, `Matrix`, `Chip`, `Fa`) from the newest lab. Add the deck to `course.json` first so the course map shows it.

## 2. Generate the lessons with the skill

A prompt that worked for the two existing labs (replace the deck and folder):

> /aha the deck ../04-Transformers.pdf — one 4-screen lesson per big idea in the deck.
> Learner profile: reuse the one in GUIDE.md (don't re-ask). Follow the teaching rules in CLAUDE.md.
> Save each lesson as labs/transformers/l1-<name>.jsx, l2-…, and add it to labs/transformers/lab.json `lessons` with the slide range it covers.
> You may use the helpers in labs/transformers/common.jsx. Tune every challenge numerically first.
> Build with `node tools/build.js transformers`, check every screen with tools/shoot.sh and run `node tools/fuzz.js transformers`.

Notes:
- The skill's default output path (`/mnt/user-data/outputs/…`) only exists on claude.ai. In Claude Code, telling it where to save
  (as in the prompt above) is enough.
- The build supports the environment the skill targets: `react` hooks, `lucide-react`, `recharts` and Tailwind core classes.
- If you run `/aha` on claude.ai instead, download the `.jsx`, drop it into `labs/<lab>/`, and add it to `lab.json` by hand.

### Learner profile (answers to the skill's Phase 1 questions)

| Question | Answer used so far |
|---|---|
| Background | Knows linear algebra, softmax and basic neural networks (taking Deep Learning in parallel); new to NLP |
| Theme | Ink notebook: dark ruled page, violet ink (accent), apricot marks (highlight), teal as a third colour, IBM Plex Serif headings, Vazirmatn for Persian |
| Running story (anchor) | **None** (see CLAUDE.md §2): use the concept and the deck's own examples, including its Persian sentences |
| Depth | One lesson per big idea, about 7 per deck, each Puzzle → Explore → Name → Challenge |

## 3. Build and look

```bash
node tools/build.js attention                    # one lab (+ the gallery). No argument = all labs.
xdg-open docs/attention/index.html               # open it locally
tools/shoot.sh attention mask-3 /tmp/mask-3.png  # screenshot lesson "mask", screen 3, and fail on any crash
node tools/fuzz.js attention                     # click / slide / type / drag through all 28 screens
```

`shoot.sh` and `fuzz.js` run on an offline copy of the page (`tools/offline.js`): React comes from `tools/vendor/`
(downloaded once, gitignored) and the web fonts are dropped, because headless Chrome can hang when it can't reach the CDN.
Persian text then falls back to a system font in screenshots. The live site uses Vazirmatn.

Each lab page links lessons as `#<lessonId>` or `#<lessonId>-<screen 1..4>`, so you can send someone straight to a challenge.
Lesson ids are lowercase letters only (the router's pattern).

## 4. lab.json fields

| Field | Meaning |
|---|---|
| `id` | folder name and URL path (`/attention/`) |
| `title`, `sub`, `stamp`, `footer` | header, sub-header, the boxed legend top-right, and the source/credits line (HTML allowed in `stamp`/`footer`) |
| `decks` | PDF file names this lab covers; marks them ✓ on the gallery's course map |
| `anchor` | what the stage shows, shown on the gallery card |
| `deckDividers` | `{ "4": "Part 2 · …" }` puts a heading before lesson 4 (used for the two halves of the Attention deck) |
| `lessons[]` | `id`, `n`, `title`, `sub` (one-line keywords), `file`, `slides` |
| `artifact` | optional claude.ai Artifact URL, shown as "claude.ai copy" |
| `built` | date, shown on the card |

## 5. Publish

```bash
git add -A && git commit -m "Add Transformer lab" && git push
```

GitHub Pages serves `docs/` from `main`, so the gallery and the new lab are live a minute later at
https://parsaivi.github.io/nlp-labs/. Push over HTTPS (SSH pushes hang on this machine).
For a private claude.ai copy as well, ask Claude to publish `build/<lab>.artifact.html` as an Artifact and put the URL in `lab.json`.

## Quality checklist (lessons from the first two labs)

- **Tune every challenge numerically before building the UI.** Put the lesson's math between `// @sim-start` and `// @sim-end`
  and load it in node with `tools/sim.js` (as `eval(require("./tools/sim.js")("labs/attention", "l1-bottleneck.jsx"))`).
  Check that the intended answer passes, the starting state fails, and the pass region is neither a pinhole nor half the board.
  Example: the Word Embedding challenge "place والیبال with one sentence" passes only with «والیبال بازی کرد» at window ±2.
- **Average out randomness the learner didn't choose.** A single random encoder gave a jagged recall curve in Attention lesson 1;
  averaging 24 seeded encoders made "the smallest d" a well-defined answer. Seed every random draw (`mulberry32`) so every learner sees the same thing.
- **Check the deck's own numbers.** Worked examples (slide 21's [2/3, 5/6], slide 26's α ≈ [0.087, 0.574, 0.340], slide 54's [0.269, 0.731, 0, 0],
  the WEAT p = 0.34 and 0.37 from all 70 splits) are reproduced exactly; keep it that way.
- **No JSX spread props** (`<rect {...attrs} />`). Babel then injects an `_extends` helper, and the build's `return` wrapper turns that helper into
  a self-calling function: the lesson dies with "Maximum call stack size exceeded". Write the props out.
- **Persian inside English:** wrap Persian words in `<Fa>` (a `<bdi lang="fa">`) in prose. In SVG text, never join two Persian words with an
  arrow or other neutral character: bidi reorders them and the arrow points the wrong way. Use separate labels or a ★ marker instead.
- **Say what's made up.** The footer states which vectors, corpora and weights are teaching toys and which numbers come from slides or papers.
- **Check every screen**, not just the first, with `tools/fuzz.js`. Also look at a phone-width view.
- **2-D toys distort cosine.** With only two dimensions most angles bunch together, so the Skip-gram lesson reads plain distances and says why.
