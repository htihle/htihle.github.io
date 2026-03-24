---
layout: default
title: WeirdML v2
permalink: /weirdml_v2.html
---

<style>
/* Hero banner */
.wml-hero {
  background: linear-gradient(135deg, var(--color-primary) 0%, var(--color-primary-lighter) 100%);
  color: var(--color-text-inverse);
  padding: var(--space-16) var(--space-8);
  border-radius: var(--radius-2xl);
  margin-bottom: var(--space-12);
  text-align: center;
}
.wml-hero h1 {
  color: var(--color-text-inverse);
  font-size: var(--text-5xl);
  margin-bottom: var(--space-4);
  border: none;
}
.wml-hero .subtitle {
  font-size: var(--text-xl);
  color: var(--color-text-light);
  max-width: 700px;
  margin: 0 auto var(--space-8);
  line-height: var(--leading-relaxed);
}
.wml-hero .hero-stats {
  display: flex;
  justify-content: center;
  gap: var(--space-10);
  flex-wrap: wrap;
}
.wml-hero .stat {
  text-align: center;
}
.wml-hero .stat-number {
  display: block;
  font-size: var(--text-4xl);
  font-weight: var(--font-bold);
  color: var(--color-accent-light);
}
.wml-hero .stat-label {
  font-size: var(--text-sm);
  color: var(--color-text-light);
}

/* Tool cards row */
.tool-cards {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: var(--space-6);
  margin: var(--space-8) 0 var(--space-12);
}
@media (max-width: 768px) {
  .tool-cards { grid-template-columns: 1fr; }
}
.tool-card {
  background: var(--color-surface);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-xl);
  overflow: hidden;
  transition: box-shadow var(--transition-base), transform var(--transition-base);
  text-decoration: none;
  color: inherit;
  display: flex;
  flex-direction: column;
}
.tool-card:hover {
  box-shadow: var(--shadow-lg);
  transform: translateY(-2px);
  text-decoration: none;
}
.tool-card img {
  width: 100%;
  height: 180px;
  object-fit: cover;
  object-position: top;
  border-bottom: 1px solid var(--color-border);
}
.tool-card .tool-card-body {
  padding: var(--space-5);
}
.tool-card .tool-card-title {
  font-size: var(--text-lg);
  font-weight: var(--font-semibold);
  color: var(--color-primary);
  margin-bottom: var(--space-2);
}
.tool-card .tool-card-desc {
  font-size: var(--text-sm);
  color: var(--color-text-muted);
  margin: 0;
}

/* Featured analysis section */
.featured-section {
  background: var(--color-surface);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-2xl);
  padding: var(--space-10);
  margin: var(--space-12) 0;
}
.featured-section h2 {
  margin-top: 0;
  border: none;
}
.featured-layout {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: var(--space-10);
  align-items: center;
  margin-top: var(--space-8);
}
@media (max-width: 768px) {
  .featured-layout { grid-template-columns: 1fr; }
}
.featured-layout img {
  width: 100%;
  border-radius: var(--radius-lg);
  box-shadow: var(--shadow-md);
}
.featured-text .key-finding {
  font-size: var(--text-3xl);
  font-weight: var(--font-bold);
  color: var(--color-accent-dark);
  line-height: var(--leading-tight);
  margin-bottom: var(--space-4);
}
.featured-text p {
  color: var(--color-text-muted);
  line-height: var(--leading-relaxed);
}

/* Timeline table */
.timeline-table {
  width: 100%;
  margin-top: var(--space-6);
  font-size: var(--text-sm);
}
.timeline-table th {
  background: var(--color-bg-alt);
  font-weight: var(--font-semibold);
  padding: var(--space-2) var(--space-3);
  text-align: left;
}
.timeline-table td {
  padding: var(--space-2) var(--space-3);
}
.timeline-table tr:hover {
  background: var(--color-bg);
}

/* Analysis grid */
.analysis-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: var(--space-8);
  margin: var(--space-8) 0;
}
@media (max-width: 768px) {
  .analysis-grid { grid-template-columns: 1fr; }
}
.analysis-item {
  background: var(--color-surface);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-xl);
  overflow: hidden;
  transition: box-shadow var(--transition-base);
}
.analysis-item:hover {
  box-shadow: var(--shadow-md);
}
.analysis-item img, .analysis-item video {
  width: 100%;
  display: block;
}
.analysis-item .analysis-body {
  padding: var(--space-5);
}
.analysis-item h3 {
  font-size: var(--text-lg);
  margin: 0 0 var(--space-2);
}
.analysis-item p {
  font-size: var(--text-sm);
  color: var(--color-text-muted);
  margin: 0;
}

/* Iframe embed */
.embed-frame {
  width: 100%;
  border: 1px solid var(--color-border);
  border-radius: var(--radius-xl);
  overflow: hidden;
  margin: var(--space-6) 0;
  box-shadow: var(--shadow-md);
}
.embed-frame iframe {
  width: 100%;
  border: none;
  display: block;
}

/* Section divider */
.section-divider {
  display: flex;
  align-items: center;
  gap: var(--space-4);
  margin: var(--space-16) 0 var(--space-8);
}
.section-divider::before,
.section-divider::after {
  content: '';
  flex: 1;
  height: 1px;
  background: var(--color-border);
}
.section-divider span {
  font-size: var(--text-sm);
  font-weight: var(--font-semibold);
  color: var(--color-text-muted);
  text-transform: uppercase;
  letter-spacing: 0.1em;
  white-space: nowrap;
}
</style>

<!-- Hero -->
<div class="wml-hero">
  <h1>WeirdML</h1>
  <p class="subtitle">
    How good are LLMs at doing machine learning on novel, weird datasets?
    A benchmark that tests genuine understanding, not pattern matching.
  </p>
  <div class="hero-stats">
    <div class="stat">
      <span class="stat-number">87</span>
      <span class="stat-label">Models evaluated</span>
    </div>
    <div class="stat">
      <span class="stat-number">17</span>
      <span class="stat-label">Tasks</span>
    </div>
    <div class="stat">
      <span class="stat-number">5</span>
      <span class="stat-label">Iterations per run</span>
    </div>
  </div>
</div>

<div class="highlight-box">
  WeirdML is included in <a href="https://epoch.ai/data/ai-benchmarking-dashboard">Epoch AI's Benchmarking Hub</a>. We're grateful to <a href="https://metr.org">METR</a> for supporting the API costs.
  Updates on <a href="https://x.com/htihle">X (@htihle)</a>.
</div>

<!-- Data tools -->
<div class="tool-cards">
  <a href="/weirdml_summary.html" class="tool-card">
    <img src="images_v2/weirdml_table_preview.png" alt="Model Summary">
    <div class="tool-card-body">
      <div class="tool-card-title">Model Overview</div>
      <p class="tool-card-desc">Performance summary across all models &mdash; accuracy, cost, code length, and execution time at a glance.</p>
    </div>
  </a>
  <a href="/weirdml_interactive.html" class="tool-card">
    <img src="images_v2/weirdml_interactive_preview.png" alt="Interactive Plot">
    <div class="tool-card-body">
      <div class="tool-card-title">Interactive Explorer</div>
      <p class="tool-card-desc">Cost vs accuracy, tokens, release dates, open/closed frontiers &mdash; explore the data interactively.</p>
    </div>
  </a>
  <a href="/weirdml_table.html" class="tool-card">
    <img src="images_v2/weirdml_model_summary.png" alt="Data Table" style="object-position: center left;">
    <div class="tool-card-body">
      <div class="tool-card-title">Full Data Table</div>
      <p class="tool-card-desc">Browse all 87 models across 17 tasks. Sort, filter, and download the raw CSV data.</p>
    </div>
  </a>
</div>


<div class="section-divider"><span>Analysis</span></div>

<!-- Time Horizons: Featured -->
<div class="featured-section">
  <h2>Time Horizons: How Long Until LLMs Can Do Your Job?</h2>
  <p>
    We estimate the <strong>time horizon</strong> of each frontier model &mdash; the duration of a human task at which the model has a 50% chance of success on WeirdML tasks. By fitting logistic curves per model and tracking the trend over generations, we can measure how quickly LLM capabilities are growing in concrete, human-relatable terms.
  </p>

  <div class="featured-layout">
    <div>
      <img src="images_v2/weirdml_time_horizons_timeline.png" alt="Time Horizons Timeline">
    </div>
    <div class="featured-text">
      <div class="key-finding">Time horizons double roughly every 5 months</div>
      <p>
        From <strong>~24 minutes</strong> (GPT-4, June 2023) to <strong>~38 hours</strong> (Claude Opus 4.6, February 2026). Tasks that would take a human half a day are now within reach of frontier models &mdash; and the trend shows no sign of slowing down.
      </p>
      <p>
        Each model's time horizon is estimated via bootstrap with 5000 resamples, giving robust uncertainty quantification. The exponential trend fit uses 10,000 bootstrap iterations across all 10 frontier models.
      </p>
    </div>
  </div>

  <details style="margin-top: var(--space-8);">
    <summary style="cursor: pointer; font-weight: var(--font-semibold); color: var(--color-primary); margin-bottom: var(--space-4);">Model-by-model time horizons</summary>
    <div class="table-container">
      <table class="timeline-table">
        <thead>
          <tr>
            <th>Model</th>
            <th>Release</th>
            <th>Time Horizon (50%)</th>
            <th>95% CI</th>
          </tr>
        </thead>
        <tbody>
          <tr><td>gpt-4-0613</td><td>Jun 2023</td><td>24 min</td><td>[4 min, 51 min]</td></tr>
          <tr><td>claude-3-opus</td><td>Mar 2024</td><td>1.1 h</td><td>[16 min, 2.3 h]</td></tr>
          <tr><td>claude-3.5-sonnet</td><td>Jun 2024</td><td>1.9 h</td><td>[59 min, 3.5 h]</td></tr>
          <tr><td>o1-preview</td><td>Sep 2024</td><td>6.2 h</td><td>[4.2 h, 10.5 h]</td></tr>
          <tr><td>o4-mini (high)</td><td>Apr 2025</td><td>8.4 h</td><td>[5.8 h, 13.6 h]</td></tr>
          <tr><td>o3-pro (high)</td><td>Jun 2025</td><td>11.8 h</td><td>[7.2 h, 18.9 h]</td></tr>
          <tr><td>gpt-5 (high)</td><td>Aug 2025</td><td>14.5 h</td><td>[8.6 h, 24.1 h]</td></tr>
          <tr><td>gemini-3-pro (high)</td><td>Nov 2025</td><td>22.3 h</td><td>[14.4 h, 36.2 h]</td></tr>
          <tr><td>gpt-5.2 (xhigh)</td><td>Dec 2025</td><td>30.6 h</td><td>[18.3 h, 54.4 h]</td></tr>
          <tr><td>claude-opus-4.6 (adaptive)</td><td>Feb 2026</td><td>37.7 h</td><td>[21.6 h, 62.4 h]</td></tr>
        </tbody>
      </table>
    </div>
  </details>
</div>


<!-- Further Analysis Grid -->
<div class="analysis-grid">
  <div class="analysis-item">
    <img src="images_v2/weirdml_date_vs_accuracy_with_frontiers.png" alt="Open vs Closed Models">
    <div class="analysis-body">
      <h3>Open vs Closed Models</h3>
      <p>How quickly do open-weights models catch up to closed-weights frontier performance? Step-frontier analysis shows the competitive dynamics.</p>
    </div>
  </div>

  <div class="analysis-item">
    <img src="images_v2/weirdml_open_model_delay_timeline.png" alt="Open Model Delay">
    <div class="analysis-body">
      <h3>Open Model Delay Timeline</h3>
      <p>The time gap between when a closed model first reaches an accuracy threshold and when an open model matches it.</p>
    </div>
  </div>

  <div class="analysis-item">
    <img src="images_v2/weirdml_cost_to_accuracy_over_time_overlaid.png" alt="Cost Decrease">
    <div class="analysis-body">
      <h3>Cost Decrease Over Time</h3>
      <p>Inference costs for hard coding tasks halve roughly every two months. See <a href="https://www.lesswrong.com/posts/ifSBamvobbyB9KWjK/inference-costs-for-hard-coding-tasks-halve-roughly-every">the blog post</a> for details.</p>
    </div>
  </div>

  <div class="analysis-item">
    <img src="images_v2/weirdml_cost_to_accuracy_over_time_overlaid_normalized.png" alt="Normalized Cost">
    <div class="analysis-body">
      <h3>Normalized Cost Decline</h3>
      <p>Cost relative to when each accuracy threshold was first achieved, showing the consistent exponential decline across thresholds.</p>
    </div>
  </div>
</div>


<div class="section-divider"><span>Evolution of the Frontier</span></div>

<div class="analysis-grid">
  <div class="analysis-item">
    <video width="100%" autoplay loop muted playsinline>
      <source src="animations/WeirdML_task_evolution.mp4" type="video/mp4">
    </video>
    <div class="analysis-body">
      <h3>Task-by-Task Evolution</h3>
      <p>Watch how state-of-the-art performance on each individual WeirdML task progresses over time, with records falling as new models appear.</p>
    </div>
  </div>

  <div class="analysis-item">
    <video width="100%" autoplay loop muted playsinline>
      <source src="animations/weirdml_cost_frontier.mp4" type="video/mp4">
    </video>
    <div class="analysis-body">
      <h3>Cost-Accuracy Frontier</h3>
      <p>The Pareto frontier of cost vs accuracy shifts outward over time as newer models push the boundary of what's achievable.</p>
    </div>
  </div>
</div>

<div class="figure" style="margin-top: var(--space-8);">
  <img src="images_v2/weirdml_date_vs_accuracy.png" alt="Date vs Accuracy" width="800">
  <p class="figure-caption">Chronological scatter of all models, showing the overall trajectory of accuracy improvement since mid-2023.</p>
</div>

<div class="figure">
  <img src="images_v2/weirdml_accuracy_cost_frontier_evolution.png" alt="Frontier Snapshots" width="800">
  <p class="figure-caption">Snapshots of the cost-accuracy Pareto frontier at six-month intervals from July 2023 to July 2025.</p>
</div>


<div class="section-divider"><span>Topline Results</span></div>

<div class="figure">
  <img src="images_v2/weirdml_model_summary.png" alt="Model Performance Summary" width="1500">
  <p class="figure-caption">Comprehensive summary of model performance. Average accuracy (bold, with 95% CI from bootstrap), individual task means (scatter points), cost per run, code length distribution, and execution time histograms.</p>
</div>

<div class="analysis-grid">
  <div class="analysis-item">
    <img src="images_v2/weirdml_cost_vs_accuracy.png" alt="Cost vs Accuracy">
    <div class="analysis-body">
      <h3>Cost vs Accuracy</h3>
      <p>The trade-off between cost per run and overall accuracy across all models (log scale).</p>
    </div>
  </div>
  <div class="analysis-item">
    <img src="images_v2/weirdml_tokens_vs_accuracy.png" alt="Tokens vs Accuracy">
    <div class="analysis-body">
      <h3>Tokens vs Accuracy</h3>
      <p>Total output tokens (including reasoning) vs accuracy. More thinking doesn't always help.</p>
    </div>
  </div>
</div>

<div class="figure">
  <img src="images_v2/weirdml_accuracy_progression.png" alt="Accuracy Progression" width="800">
  <p class="figure-caption">Accuracy progression over five iterations for selected models. The leftmost point is zero-shot; later points incorporate feedback from execution output and test accuracy.</p>
</div>


<div class="section-divider"><span>Methodology</span></div>

## How It Works

WeirdML presents LLMs with weird and unusual ML tasks, designed to require careful thinking and genuine understanding. Each model must:

1. **Understand** the data properties and problem structure
2. **Design** an appropriate ML architecture and training setup
3. **Generate** working PyTorch code implementing the solution
4. **Debug** and improve over 5 iterations based on terminal output and test accuracy
5. **Optimize** within strict computational constraints (TITAN V GPU, 12GB, 120s timeout)

<div class="figure">
  <img src="images_v2/evaluation_setup.png" alt="Evaluation Setup" width="500">
  <p class="figure-caption">Evaluation pipeline: LLM code generation, isolated Docker execution, metric evaluation, and feedback loop.</p>
</div>

Each model gets at least 5 runs per task (some expensive models get 2) to account for the high variance in performance. The final score is the mean of the maximum test accuracy achieved across the 5 iterations in each run, averaged over all runs and tasks. See the [full system prompt](prompts/system_prompt_v2.md).


<div class="section-divider"><span>Example Tasks</span></div>

Below are six of the 17 tasks (the original WeirdML v1 set). The remaining 11 tasks serve as a hidden test set.

### Shapes (Easy)

Classify one of five shapes from 512 2D coordinates, where only some points form the shape and the rest are noise. Shapes are centered with fixed orientation.

<div class="figure">
  <img src="images/train_examples_easy.png" alt="Shapes Easy" width="600">
</div>

<div class="figure">
  <img src="images_v2/shapes_easy_max_accuracy_comparison.png" alt="Shapes Easy Results" width="800">
  <p class="figure-caption">This task is essentially solved. Most capable models achieve near-perfect accuracy.</p>
</div>

### Shapes (Hard)

Same as above, but shapes are randomly positioned, oriented, and sized. Requires translation, rotation, and scale invariant features.

<div class="figure">
  <img src="images/train_examples_hard.png" alt="Shapes Hard" width="600">
</div>

<div class="figure">
  <img src="images_v2/shapes_hard_max_accuracy_comparison.png" alt="Shapes Hard Results" width="800">
  <p class="figure-caption">Much harder than the easy variant. Most models do little better than chance, but the best achieve ~90%.</p>
</div>

### Image Patch Shuffling (Easy)

Arrange 9 shuffled grayscale image patches (9x9 pixels) to reconstruct a 27x27 image from Fashion-MNIST.

<div class="figure">
  <img src="images/scrambled_vs_unscrambled_easy.png" alt="Shuffle Easy" width="500">
</div>

<div class="figure">
  <img src="images_v2/shuffle_easy_max_accuracy_comparison.png" alt="Shuffle Easy Results" width="800">
</div>

### Image Patch Shuffling (Hard)

RGB patches from random subsets of ImageNette images. Position can't be inferred from individual patches &mdash; requires combining information across patches.

<div class="figure">
  <img src="images/scrambled_vs_unscrambled_hard.png" alt="Shuffle Hard" width="500">
</div>

<div class="figure">
  <img src="images_v2/shuffle_hard_max_accuracy_comparison.png" alt="Shuffle Hard Results" width="800">
</div>

### Chess Game Outcome Prediction

Predict the outcome (white wins, black wins, draw) from move sequences of beginner-level games.

<div class="figure">
  <img src="images/chess-games.png" alt="Chess" width="600">
</div>

<div class="figure">
  <img src="images_v2/chess_winners_max_accuracy_comparison.png" alt="Chess Results" width="800">
  <p class="figure-caption">Random baseline is ~50% (always guess white). Little progress beyond 80%, suggesting this task is genuinely hard.</p>
</div>

### Unsupervised Digit Recognition

Semi-supervised: classify digits with only 26 labeled examples and ~16,000 unlabeled samples with uneven class distribution.

<div class="figure">
  <img src="images/train_test_data.png" alt="Digits" width="600">
</div>

<div class="figure">
  <img src="images_v2/digits_unsup_max_accuracy_comparison.png" alt="Digits Results" width="800">
  <p class="figure-caption">Large improvement since v1. Best models now consistently score well.</p>
</div>

---

*For the original WeirdML v1 results, see the [archived page](weirdml_v1.html).*
