---
layout: default
title: WeirdML
weirdml_section: results
---

{% include weirdml-v3-theme.html %}

<div class="v3-hero" markdown="1">

# WeirdML v3

WeirdML (v3) is an **agentic benchmark** featuring **11 complex hand-made tasks** made to challenge the model to explore and understand unfamiliar data, develop machine learning and data analysis pipelines and produce appropriate results despite limited data, unspecified goals and/or very limited feedback.
{: .weirdml-intro }

WeirdML v3 was created by me ([Håvard Tveit Ihle]({{ '/' | relative_url }})) at the [Norwegian Defence Research Establishment (NDRE)](https://www.ffi.no/en/about-ffi). API costs were supported primarily by [EpochAI](https://epoch.ai/), secondarily by [METR](https://metr.org/) and [NDRE](https://www.ffi.no/en/about-ffi). Thanks for the support!

Previous versions: [WeirdML v2](weirdml_v2.html) · [WeirdML v1](weirdml_v1.html).

{% include weirdml-v3-nav.html %}

</div>

<div id="results"></div>

<p class="v3-standalone">
  Open standalone: <a href="{{ '/weirdml_v3_interactive.html' | relative_url }}">Interactive plot</a> · <a href="{{ '/weirdml_v3_summary.html' | relative_url }}">Model summary</a> · <a href="{{ '/assets/data/weirdml_v3.json' | relative_url }}">Prepared data (.json)</a> · <a href="{{ '/data/weirdml_v3_results.json' | relative_url }}">Full data (.json)</a>
</p>

<div class="v3-embed">
  <iframe title="WeirdML v3 interactive progress plot" src="{{ '/weirdml_v3_interactive.html' | relative_url }}?embed" scrolling="no"></iframe>
</div>

**In the Tokens view above, each model’s line shows its average best-so-far effective score across all 11 tasks. That model’s official score is 80% normalized area under its line on a logarithmic token axis, plus 20% of its final value.** Only the interval from **500k to 50M tokens** contributes to the area; earlier progress is shown for context, and the best score reached before 500k carries into the scoring window. The best-so-far score is carried forward to the full 50M-token limit, even if a run ends early.
{: .v3-note }

The curve averages runs within each configuration, then weights each of the 11 tasks equally; hinted and hintless twins each receive half their task’s weight. Ship Detect’s cost-weighted tokens are scaled ×25: its native 20k–2M scoring window maps to 500k–50M on the combined plot. Effective scores include normalization and hint penalties; they are not raw accuracy. Select **Per Task** to explore any of the 15 configurations, **Cost** or **Date** to compare official scores, or **Open vs Closed** to compare the score frontiers over time.
{: .v3-note }

<div class="v3-embed">
  <iframe id="v3-summary-iframe" title="WeirdML v3 model summary" src="{{ '/weirdml_v3_summary.html' | relative_url }}?embed" scrolling="no"></iframe>
</div>

Shaded score bands show approximate 95% run-uncertainty intervals, with variance pooled across configurations and models. Black markers show all 15 configuration means. Cost is mean API cost per run, using the same task weighting as scores. Final Best Score is the equally weighted mean final effective score. Harness shows the agent software and version used for the included runs. Only models with at least one valid run in every configuration are included.
{: .v3-note }


<script src="{{ '/assets/js/weirdml-v3-page.js' | relative_url }}" defer></script>
