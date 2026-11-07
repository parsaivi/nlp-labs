# NLP Labs

Interactive, hands-on lessons built from the NLP course decks (CS 677, M. Samiei, Sharif University of Technology).
Each lab takes one lecture deck and turns every big idea into a 4-screen lesson: **Puzzle → Explore → Name → Challenge**.
You drag, tune and break things before the idea gets its name.

**Open the gallery: https://parsaivi.github.io/nlp-labs/**

| Lab | Deck | Lessons |
|---|---|---|
| [Word Embedding Lab](https://parsaivi.github.io/nlp-labs/word-embedding/) | Lecture 2: Word Embedding | one-hot vs dense · distributional hypothesis & co-occurrence · Skip-gram trained live in the browser · negative sampling · analogies & intrinsic evaluation · FastText n-grams & polysemy · WEAT & the permutation test |
| [Attention Lab](https://parsaivi.github.io/nlp-labs/attention/) | Lecture 3: Attention | the fixed-vector bottleneck · score, softmax, mix · Bahdanau additive attention · self-attention with Q, K, V · √d_k scaling & multi-head · teacher forcing & the causal mask · sinusoidal positions |

Sister site, same method and build: [Deep Learning Labs](https://parsaivi.github.io/deep-learning-labs/).
This one wears a different theme: an ink notebook (ruled dark page, violet ink, apricot marks, serif headings, Vazirmatn for Persian).

## Layout

```
labs/<lab>/          lab.json (titles, decks, lesson list) + one .jsx per lesson + common.jsx (UI kit, vector math, plots)
course.json          every course deck, in order (drives the course map)
tools/build.js       labs/ → docs/ (GitHub Pages) and build/ (claude.ai Artifact copies)
tools/new-lab.js     start a new lab folder
tools/shoot.sh       headless-Chrome screenshot + crash check of one lesson screen
tools/fuzz.js        clicks, slides, types and drags through every screen of a lab and reports crashes
tools/offline.js     offline copy of a page for the two tools above (React from tools/vendor/, no web fonts)
tools/sim.js         load a lesson's math in node, for tuning challenges
docs/                built site, served by GitHub Pages (don't edit by hand)
.claude/skills/interactive-educator/   the /aha skill, unchanged from upstream
```

**Adding the next deck: see [GUIDE.md](GUIDE.md).** In short: `node tools/new-lab.js`, run `/aha` on the deck,
`node tools/build.js <lab>`, `node tools/fuzz.js <lab>`, then commit and push.

## Credits

- Lessons are generated with Claude using the [interactive-educator](https://github.com/Wamikmk/interactive-educator) skill by Wamik (CC BY 4.0), vendored unchanged in `.claude/skills/`.
- Slide content belongs to its author (M. Samiei; after Mikolov et al. 2013, Bojanowski et al. 2017, Caliskan et al. 2017, Bahdanau et al. 2015, Vaswani et al. 2017). The PDFs aren't included here.
- Word vectors, corpora and model weights are small made-up teaching examples, except where a lesson cites a slide or paper (each lab's footer says which).
