---
layout: default
title: WeirdML v3 — Scoring
weirdml_section: scoring
---

{% include weirdml-v3-theme.html %}

<div class="v3-hero" markdown="1">

# WeirdML v3

In order to get more information from each run, we introduce **area-under-the-curve scoring** and **risk-free hints**. These mechanisms make the score more gradual, the skill ceiling higher and the skill floor lower. This means that we get much more information from each task than a single pass/fail grade.
{: .weirdml-intro }

{% include weirdml-v3-nav.html %}

</div>

<div class="weirdml-scoring-page" markdown="1">



## Ten submissions {#ten-submissions}

In WeirdML a model is placed in a sandbox with some kind of data and a task to perform. It then has to submit some kind of solution. This can be a Python script that runs in a separate test sandbox or an array of numbers or other output data. **Every submission gets a score between 0 and 1** (on some normalized scale that differs from task to task). The model typically has 10 such submissions and often, though not always, gets back the score and/or some other form of feedback. **The best score so far is always what counts**, so a model that has already achieved a high score is incentivized to experiment and try different approaches to increase the score, without having to worry about having a "bad" submission.

<figure class="scoring-static">
  <picture><source media="(max-width: 600px)" srcset="{{ '/assets/images/weirdml-v3/scoring/submissions-mobile.svg' | relative_url }}"><img src="{{ '/assets/images/weirdml-v3/scoring/submissions.svg' | relative_url }}" alt="Ten numbered example submissions plotted against total tokens on a logarithmic axis. The solid line retains the best score; lower-scoring attempts are shown in rose. The last attempt scores 0.68, but the best stays at 0.73."></picture>
  <figcaption>Illustrative run. Numbered markers identify the ten submissions; the x-axis shows total tokens on a logarithmic scale. A worse attempt does not lower the best score, which carries forward to the token limit.</figcaption>
</figure>

## Score is the area under the curve {#area-under-the-curve}

**The official score on a task is 0.8 × the area under the "best-so-far" curve on a logarithmic token axis from 500k total tokens to the 50M total token budget, plus 0.2 × the final best submission.** A model thus has an incentive to use submissions early to get credit for a partial solution, and to not wait until the budget limit to submit. This also makes the scoring more gradual, and makes the skill ceiling much higher.

<figure class="scoring-static">
  <div class="scoring-comparison">
    <div><h3>Earlier improvement</h3><img src="{{ '/assets/images/weirdml-v3/scoring/area-early.svg' | relative_url }}" alt="Score rises from 0.20 to 0.80 at 5M tokens. Normalized log-area is 0.50; official score is 0.560."><p>Area <b>0.500</b> · Official score <b>0.560</b></p></div>
    <div><h3>Later improvement</h3><img src="{{ '/assets/images/weirdml-v3/scoring/area-late.svg' | relative_url }}" alt="Score rises from 0.20 to 0.80 at 20M tokens. Normalized log-area is approximately 0.319; official score is approximately 0.416."><p>Area <b>0.319</b> · Official score <b>0.416</b></p></div>
  </div>
  <figcaption>Illustrative runs with the same final best score of 0.80. Earlier progress earns more shaded area. Official score = 0.8 × normalized area + 0.2 × final best.</figcaption>
</figure>

## Risk-free hints {#browse-hints}

Many tasks offer a set of hints that help the model get to a scoring solution. **Buying a hint multiplies the score of every later submission by a fixed factor.** A cheap hint may cost ×0.9, while a hint with a complete solution method may cost ×0.25. A model gets access to a "browse_hints" tool. Calling this tool makes a call to the same model, with the same context, in a side-conversation, where the model gets to read the full text of every hint, together with the price, and decides which hints to keep, before that conversation is thrown away. **This way browsing the hints is risk-free**: you only pay if you decide that one or more of the hints are worth the cost.

Even without buying any hints, the model can still extract some information from browsing them, either implicitly (perhaps I'm on the right track, since the hints would not help me) or explicitly, by coordinating with the hint browser (which can be done in various ways). To minimize this, we allow only three hint browsings on each task.

<figure class="scoring-static">
  <picture><source media="(max-width: 600px)" srcset="{{ '/assets/images/weirdml-v3/scoring/hints-mobile.svg' | relative_url }}"><img src="{{ '/assets/images/weirdml-v3/scoring/hints.svg' | relative_url }}" alt="Hint browsing branches off from the main run into a side-conversation, while the main context is retained. Selected hints rejoin the main run; the preview and rejected hints are discarded."></picture>
  <figcaption>Only selected hints rejoin the main run. For example, keeping a ×0.9 hint makes a later score of 0.80 count as 0.72; an earlier best is preserved.</figcaption>
</figure>

</div>
