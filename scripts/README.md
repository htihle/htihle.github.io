# WeirdML v3 preparation and local preview

No publishing or Git operations are part of these commands.

## Prepare results

```sh
python scripts/prepare_weirdml_v3.py
```

The default input is `data/weirdml_v3_results.json`. The default mode is `real`, selecting nonsynthetic runs and models only. Every displayed model must have at least one valid run in all 15 suite configurations.

The standard-library-only script validates valid runs, averages configurations and base tasks, calculates 10,000 partially pooled run-uncertainty draws (seed 20260915), and writes `assets/data/weirdml_v3.json`. Curves are exact mean running-best steps over the union of run change points. Generated output is deterministic and includes the input SHA-256. No individual raw runs are loaded by the webpage.

To prepare a different actual-model export:

```sh
python scripts/prepare_weirdml_v3.py data/new-export.json --mode real
```

The September 16 preliminary export has two complete models; incomplete models are omitted. The full-export link points to this actual-data file, and the page labels results as preliminary. Preserve the v2 CSV and interactive tools.

## Validate and build

```sh
python -m unittest discover -s tests -v
bash scripts/build_site.sh
bundle exec jekyll serve --host 0.0.0.0 --port 4000 --skip-initial-build
```

Then open `/weirdml.html` on port 4000. Install Ruby dependencies using `bundle install` if needed. If using a temporary dependency directory, prefix both build and serve with `BUNDLE_PATH=/tmp/weirdml-v3-gems`.

Run `bash scripts/build_site.sh` again after replacing the data export. Ordinary Jekyll builds alone do not rerun Python analysis. The wrapper accepts the same input and mode arguments as the preparation script.

## Future build automation

When publication is authorized, the build job should run preparation followed by the Jekyll build whenever the input export or analysis code changes, then deploy that built artifact. The existing repository has no checked-in deployment workflow. No deployment workflow is added during this unpublished preview phase.

## Scoring details

Each configuration is a mean of valid runs. Hinted/hintless twins are averaged into one base task. The leaderboard averages the 11 base tasks equally. Mean API cost uses identical weighting. Intervals hold tasks fixed and estimate run uncertainty. Pool the within-model/configuration residual sum of squares divided by its total degrees of freedom, using valid runs in the selected real/synthetic mode, including incomplete models. Never pool differences between configuration means. For each configuration with n runs and residual sum of squares S, use an inverse-gamma variance prior with shape 3 and scale 2 × pooled variance (prior mean = pooled variance). The posterior shape is 3 + (n−1)/2 and scale is (4 × pooled variance + S)/2. Its expected variance is (4 × pooled variance + S)/(4+n−1): singletons use the pool; repeated configurations increasingly use their own variation.

For each of 10,000 draws, sample each configuration variance from that distribution and its mean from Normal(observed mean, variance/n), independently across configurations. Combine with the existing equal-task weights. Use the 2.5th/97.5th percentiles, restricted to [0,1]. The displayed score is unchanged. These are approximate empirical-Bayes intervals, not task-bootstrap confidence intervals: they assume independent fresh runs, approximately normal score noise, and transferable run variance. The fitted pool's uncertainty is not propagated. Four prior degrees of freedom is a fixed modeling choice, not a fitted parameter. If no repeated variation is observed, use the explicit conservative prior mean variance 0.25 (the maximum variance for a [0,1] score); this does not guarantee interval coverage. Metadata records the pool and fallback. Synthetic and actual data never share a pool.

Score validation checks exported running maxima, the log-area integral through the full axis limit, final best, and the weighted score, with a 0.000005 tolerance for rounded exports. Exported effective scores remain authoritative; raw normalization and hint penalties are not reconstructed. Wall-clock submission fields are not used.

The main progress plot uses an exact, precomputed average across the 11 base tasks. Each configuration curve is scaled by `common_token_budget / configuration_budget` before combining. Ordinary 50M-token tasks are unchanged; Ship Detect's 2M cost-weighted-token budget maps to 50M, a factor of 25. Hint twins share one task's weight at every point. All scoring windows must begin at the same fraction of their budgets. The chart labels this axis as equivalent tokens and shows effective score, since raw task metrics cannot be averaged as accuracy.

## V2 presentation and current views

`weirdml_v3_interactive.html` and `weirdml_v3_summary.html` copy the v2 view CSS and layout directly. The landing page embeds them using the same frames as v2. D3 7.9.0 is vendored under `assets/js/vendor/` with its license. The summary uses v2's task markers and uncertainty bands, with both Final Best Score and harness name/version columns (the harness replaces the removed token column); there is no token column. The detailed task table and scoring section are currently omitted from the landing page.

Plot choices are Tokens (overall progress), Per Task, Cost, Date, and Open vs Closed. Cost/date scatterplots use the prepared overall score. Date views use model release dates and omit undated models. Open/closed step frontiers and the shaded gap are prepared in Python and extend to the export date, so opening the page never updates an analysis.

Canonical display columns are fixed in `LEGEND_COLUMNS`: Shapes/Splash, Mystery Box twins, Ship Detect/Ship Tune, Reaction Rates twins, Scan Stitch twins, Shattered Prior twins, Night School/TOD Pipeline, then Bonanza. The prepared `legend_columns` field defines eight columns: seven full columns and Bonanza as the final singleton because there are 15 configurations. The task selector follows this same column-by-column order. `TASK_MARKERS` keeps symbols stable: hint twins share a shape/rotation, filled only when hints are available and enabled; all hintless configurations, including tasks with no hints available, use hollow symbols. Statistical task order is independent of this presentation order.

## V3 appearance

The new style is the sole active style. The comparison button, classic query behavior, and config toggle have been removed. Presentation lives in `assets/css/weirdml-v3-design.css`, with the self-hosted fonts declared in `assets/css/weirdml-v3-fonts.css` and shared section layouts in `assets/css/weirdml-v3-sections.css`.

## V3 sections and task artifact

- `weirdml.md`: Results, with the plot and leaderboard.
- `weirdml_tasks.md`: Selected tasks, adapted from the owner-supplied `selected_tasks_page.zip`. The four reviewed explanations and seven figure captions are preserved, with local WebP images on dark backgrounds and a keyboard-accessible native dialog viewer. The supplied canary remains on the page.
- `weirdml_scoring.md`: The owner’s three-part walkthrough (submissions, log-area scoring, and hints), with static SVG figures. Previous scoring prose is backed up in `scripts/backups/weirdml-scoring-before-walkthrough.html`, excluded from site output.
- Shared navigation: `_includes/weirdml-v3-nav.html`; basic section styles: `assets/css/weirdml-v3-sections.css`.
- The original ZIP is unchanged and excluded from Jekyll output. Only its WebP images are extracted to `assets/images/weirdml-v3/tasks/`; no PNG masters, duplicate standalone page, or external fonts are shipped.
- These changes authorize local preview only, not publication of the task material.

Static scoring figures are hand-authored SVG emitted by `scripts/draw_scoring_figures.py` (standard library only, illustrative data only; regenerate with `python3 scripts/draw_scoring_figures.py`, which rewrites the six files in `assets/images/weirdml-v3/scoring/`). Checked-in SVGs require neither browser JavaScript nor a plotting dependency for ordinary site builds. The previous interactive version is backed up under scripts/backups. Hint browsing consumes run tokens and API cost; only accepting hints applies their score penalties.
