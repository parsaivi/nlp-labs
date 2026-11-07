# Teaching rules for this repo (Parsa's configuration)

These rules apply on top of the interactive-educator skill (`/aha`) whenever a lab is generated or revised here.
They don't change the skill. When the skill and these rules disagree, these rules win.
They come from learner feedback on the Deep Learning labs (2026-11) and are kept identical to that repo's rules,
so the two sites teach the same way.

## 1. Big picture first

Every lesson opens by placing itself in the deck's own argument, in two or three plain sentences:
**what we already showed → what is still broken or missing → what this lesson fixes.**
Use the course's real storyline, not an invented one. The learner should always know what problem is being solved and why it comes next.

- Bad: a lesson on self-attention that opens with a party where guests whisper to each other. The learner has to decode the story before seeing the question.
- Good (Attention Lab, lesson 4): "Bahdanau attention still runs through an RNN, so each state waits for the previous one (slide 31). The Transformer asks whether attention can replace recurrence entirely."

The Name screen closes the loop the same way: what we just learned, in one sentence, and what the next lesson builds on it.

## 2. No stories unless they make it simpler

Default to the concept itself and the slides' own examples: the deck's Persian sentences (تیم ما فوتبال بازی کرد, I drink tea → من چای می‌نوشم), شیر with its three meanings, the worked numbers on the slides.
An analogy or running story is allowed only when it makes the problem **easier** than the plain version. If the learner has to translate the story back into the math, drop it.
So skip the skill's Phase 1 anchor-domain question (Q3): the answer is "none, use the concept and the slides' examples".

## 3. Every sentence must be readable on its own

Each prompt and note says one thing plainly: what you are looking at, what to do, and what to notice.
Name the real quantities (query q, key k, d_k, the weights α, the window size), not story stand-ins.
If a sentence only makes sense after playing with the widget, or needs the story to parse, rewrite it.

## 4. The interaction serves a stated claim

Before each widget, state in course terms the claim or question it tests. After it, state the result in words, then the formula, then the slide number.
Test: could the learner say what they learned **without** the widget in front of them? If they could only say "I moved the slider and the line changed", the lesson failed.
Fiddling with a picture is not the goal. Understanding the idea on the slide is.

## Quick check before shipping a lesson

- [ ] The first lines say where we are in the deck and what problem this lesson solves.
- [ ] No story or analogy, or one that clearly makes things simpler.
- [ ] Every prompt names real quantities and makes sense read alone.
- [ ] The Name screen states the takeaway in plain words, the formula and the slide, and links to the next lesson.
- [ ] The challenge was tuned numerically: its intended answer passes, its starting state fails (see GUIDE.md).
- [ ] `node tools/fuzz.js <lab>` reports no crashes and no console errors.
