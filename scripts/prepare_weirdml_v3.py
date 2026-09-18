#!/usr/bin/env python3
"""Prepare compact WeirdML v3 results. Standard library only; no network access."""
import argparse
from bisect import bisect_right
from collections import defaultdict
from datetime import date
import hashlib
import json
import math
from pathlib import Path
import random
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
TOLERANCE = 5e-6  # Exported scores have six decimal places.
# Canonical alphabetical base-task order. Variant order never determines shape.
TASK_MARKERS = {
    'mystery_box': ('circle', 0),
    'night_school': ('square', 0),
    'reaction_rates': ('diamond', 0),
    'scan_stitch': ('triangle', 0),
    'shapes_generalize': ('triangle', 180),
    'shattered_prior': ('triangle', -90),
    'ship_detect': ('triangle', 90),
    'ship_tune': ('pentagon', 0),
    'splash_generalize': ('star', 0),
    'tod_pipeline': ('cross', 0),
    'weirdml_bonanza': ('wye', 0),
}

# Canonical display columns, top entry followed by bottom entry.
LEGEND_COLUMNS = [
    ['shapes_generalize', 'splash_generalize'],
    ['mystery_box', 'mystery_box--nohints'],
    ['ship_detect', 'ship_tune'],
    ['reaction_rates', 'reaction_rates--nohints'],
    ['scan_stitch', 'scan_stitch--nohints'],
    ['shattered_prior', 'shattered_prior--nohints'],
    ['night_school', 'tod_pipeline--nohints'],
    ['weirdml_bonanza'],
]


def close(actual, expected, context):
    if not math.isfinite(actual) or abs(actual - expected) > TOLERANCE:
        raise ValueError(f'{context}: exported {actual}, calculated {expected}')


def curve(run, *, full_history=False):
    """Running-best steps for scoring, or full plotting history from zero."""
    axis = run['axis']
    start, limit = axis['start'], axis['limit']
    if axis['kind'] not in ('tokens', 'cost_weighted') or not 0 < start < limit:
        raise ValueError(f"Invalid scoring axis: {run['sample_key']}")
    if full_history:
        start = 0
    points = [[start, 0.0]]
    previous_x, best = 0, 0.0
    for entry in run['submissions']:
        x = entry[axis['kind']]
        if not math.isfinite(x) or x < previous_x:
            raise ValueError(f"Nonmonotonic resource axis: {run['sample_key']}")
        previous_x = x
        if x > limit or entry.get('scored') is False:
            continue
        effective = entry.get('effective', 0)
        if not math.isfinite(effective) or not 0 <= effective <= 1:
            raise ValueError(f"Invalid effective score: {run['sample_key']}")
        best = max(best, effective)
        close(entry['running_max'], best, f"{run['sample_key']} running maximum")
        x = max(start, x)
        if x == points[-1][0]:
            points[-1][1] = best
        elif best != points[-1][1]:
            points.append([x, best])
    if points[-1][0] != limit:
        points.append([limit, best])
    return points


def validate_run(run):
    points = curve(run)
    area = sum(y * math.log(x2 / x) for (x, y), (x2, _) in zip(points, points[1:]))
    area /= math.log(run['axis']['limit'] / run['axis']['start'])
    weight = run['axis']['final_best_weight']
    close(run['area_term'], area, f"{run['sample_key']} log area")
    close(run['final_best'], points[-1][1], f"{run['sample_key']} final best")
    close(run['task_score'], (1 - weight) * run['area_term'] + weight * run['final_best'],
          f"{run['sample_key']} weighted score")
    return points


def quantile(values, p):
    """Linear interpolation between adjacent ordered uncertainty draws."""
    position = (len(values) - 1) * p
    lo, hi = math.floor(position), math.ceil(position)
    return values[lo] + (values[hi] - values[lo]) * (position - lo)


def complete_mean(values):
    values = list(values)
    return None if any(v is None for v in values) else mean(values)


def task_means(config_scores, configurations):
    by_task = defaultdict(list)
    for config in configurations:
        by_task[config['task']].append(config_scores[config['id']])
    return {task: complete_mean(scores) for task, scores in by_task.items()}


PRIOR_DF = 4


def variance_pool(grouped):
    """Pool within-model/configuration residuals, never between-task differences."""
    residuals, degrees = [], 0
    for mid in sorted(grouped):
        for sid in sorted(grouped[mid]):
            values = sorted(r['task_score'] for r in grouped[mid][sid])
            center = mean(values)
            residuals.extend((v - center) ** 2 for v in values)
            degrees += len(values) - 1
    pooled = math.fsum(residuals) / degrees if degrees else 0
    # With no observed variation, use an explicit conservative variance prior.
    fallback = pooled == 0
    return {'variance': .25 if fallback else pooled, 'degrees_of_freedom': degrees,
            'fallback': fallback, 'prior_df': PRIOR_DF}


def run_interval(groups, configurations, pool, resamples, seed):
    """Approximate empirical-Bayes interval; fixed tasks and independent fresh runs.

    Inverse-gamma variance prior has mean pooled variance and prior_df=4.
    Flat location prior gives a normal conditional distribution for each mean.
    Means stay unshrunk; only variances borrow information. Normal sampling is
    an approximation for bounded scores. Final endpoints are restricted to [0,1].
    """
    counts = defaultdict(int)
    for config in configurations:
        counts[config['task']] += 1
    rng = random.Random(seed)
    draws = [0.0] * resamples
    for config in sorted(configurations, key=lambda c: c['id']):
        values = sorted(r['task_score'] for r in groups[config['id']])
        n, center = len(values), mean(values)
        ss = math.fsum((v - center) ** 2 for v in values)
        df = pool['prior_df'] + n - 1
        numerator = pool['prior_df'] * pool['variance'] + ss
        weight = 1 / (len(counts) * counts[config['task']])
        for i in range(resamples):
            variance = numerator / (2 * rng.gammavariate(df / 2 + 1, 1))
            draws[i] += weight * rng.gauss(center, math.sqrt(variance / n))
    draws.sort()
    return [max(0, min(1, quantile(draws, .025))),
            max(0, min(1, quantile(draws, .975)))]


def mean_curve(runs, curves):
    axes = {(r['axis']['kind'], r['axis']['start'], r['axis']['limit']) for r in runs}
    if len(axes) != 1:
        raise ValueError('Cannot combine runs with different resource axes')
    # Union of changes yields an exact mean step curve, not sampled interpolation.
    individual = [curves[r['sample_key']] for r in runs]
    xs = sorted({x for points in individual for x, _ in points})
    positions = [[x for x, _ in points] for points in individual]
    output = []
    for x in xs:
        value = mean(points[bisect_right(axis, x) - 1][1]
                     for points, axis in zip(individual, positions))
        if not output or value != output[-1][1] or x == xs[-1]:
            output.append([x, value])
    return output


def overall_curve(configs, configurations, start, limit):
    """Equal-task mean on a common budget axis; hint twins share task weight."""
    counts = defaultdict(int)
    for config in configurations:
        counts[config['task']] += 1
    weighted = []
    for config in configurations:
        result = configs[config['id']]
        factor = limit / result['axis']['limit']
        if not math.isclose(result['axis']['start'] * factor, start):
            raise ValueError('Scoring windows must start at the same budget fraction')
        points = [[x * factor, y] for x, y in result['curve']]
        weight = 1 / (len(counts) * counts[config['task']])
        weighted.append((points, [x for x, _ in points], weight))
    xs = sorted({x for points, _, _ in weighted for x, _ in points})
    output = []
    for x in xs:
        value = sum(points[bisect_right(axis, x) - 1][1] * weight
                    for points, axis, weight in weighted)
        if not output or value != output[-1][1] or x == xs[-1]:
            output.append([x, value])
    return output


def date_frontiers(models, as_of):
    """Prepare v2-style step lines and shaded gap once, when data changes."""
    dated = sorted((m for m in models if m['release_date']),
                   key=lambda m: (m['release_date'], m['id']))
    for model in dated:
        date.fromisoformat(model['release_date'])
    end = max([as_of] + [m['release_date'] for m in dated])
    lines = {'open': [], 'closed': []}
    best = {'open': None, 'closed': None}
    gap, frontier_ids = [], []
    by_date = defaultdict(list)
    for model in dated:
        by_date[model['release_date']].append(model)
    def append_gap(day):
        # Closed shading extends to zero until an open frontier exists.
        # Once open leads, its fill covers the full area; never invert the gap.
        if best['closed'] is not None:
            lower = best['open'] or 0
            gap.append([day, lower, max(lower, best['closed'])])

    for day, released in by_date.items():
        append_gap(day)
        for key in lines:
            candidates = [m for m in released if m['open_weights'] == (key == 'open')]
            if not candidates:
                continue
            winner = max(candidates, key=lambda m: m['score'])
            if best[key] is None or winner['score'] > best[key]:
                lines[key].extend([[day, best[key] or 0], [day, winner['score']]])
                best[key] = winner['score']
                frontier_ids.append(winner['id'])
        append_gap(day)
    for key in lines:
        if lines[key] and lines[key][-1][0] < end:
            lines[key].append([end, best[key]])
    if gap and gap[-1][0] < end:
        append_gap(end)
    return {**lines, 'gap': gap, 'as_of': end, 'frontier_ids': frontier_ids,
            'undated_models': [m['id'] for m in models if not m['release_date']]}


def prepare(data, mode='synthetic', resamples=10000, seed=20260915):
    if data['schema_version'] != 1:
        raise ValueError('Unsupported schema_version')
    if resamples < 2:
        raise ValueError('At least two uncertainty draws are required')
    configurations = []
    for sid in data['suite']['sample_ids']:
        spec = data['suite']['samples'][sid]
        task = data['tasks'][spec['task']]
        mode_label = ('Hints allowed' if spec['hints_enabled'] else 'No hints') if task['hints'] else None
        if spec['task'] not in TASK_MARKERS:
            raise ValueError(f"Assign a canonical marker for new task {spec['task']}")
        shape, rotation = TASK_MARKERS[spec['task']]
        configurations.append({'id': sid, 'task': spec['task'], 'name': task['display_name'],
                               'hint_mode': mode_label, 'show_raw_metric': task['show_raw_metric'],
                               'marker': {'shape': shape, 'rotation': rotation,
                                          'filled': bool(task['hints']) and spec['hints_enabled']}})
    by_task = defaultdict(list)
    for config in configurations:
        by_task[config['task']].append(config['id'])
    present = {c['id'] for c in configurations}
    legend_columns = [[sid for sid in column if sid in present] for column in LEGEND_COLUMNS]
    legend_columns = [column for column in legend_columns if column]
    display_order = {sid: i for i, sid in enumerate(sid for column in legend_columns for sid in column)}
    if present - display_order.keys():
        raise ValueError('Assign a canonical legend column for each new configuration')
    configurations.sort(key=lambda c: display_order[c['id']])
    required = {c['id'] for c in configurations}
    if len(required) != len(configurations):
        raise ValueError('Duplicate suite configuration IDs')
    # Keep statistical task order independent of presentation order.
    tasks = [task for task in TASK_MARKERS if task in by_task]
    common_axis = {'kind': 'equivalent_tokens',
                   'start': data['scoring_rule']['token_axis_start'],
                   'limit': data['scoring_rule']['token_budget']}
    grouped = defaultdict(lambda: defaultdict(list))
    curves, seen = {}, set()
    validated = 0
    for run in data['samples']:
        key = run['sample_key']
        if key in seen:
            raise ValueError(f'Duplicate sample_key: {key}')
        seen.add(key)
        if run['model_id'] not in data['models'] or run['task'] not in data['tasks']:
            raise ValueError(f'Unknown model or task: {key}')
        if run['verdict'] != 'valid':
            continue
        # Validate all valid runs, including incomplete models, before selecting the preview.
        validate_run(run)
        curves[key] = curve(run, full_history=True)
        validated += 1
        if bool(run['synthetic']) != (mode == 'synthetic') or run['sample_id'] not in required:
            continue
        spec = data['suite']['samples'][run['sample_id']]
        if run['task'] != spec['task'] or (data['tasks'][run['task']]['hints'] and
                                         run['hints_enabled'] != spec['hints_enabled']):
            raise ValueError(f'Run disagrees with suite configuration: {key}')
        grouped[run['model_id']][run['sample_id']].append(run)
    pool = variance_pool(grouped)
    models, excluded = [], []
    for mid, meta in data['models'].items():
        if bool(meta['synthetic']) != (mode == 'synthetic'):
            continue
        groups = grouped[mid]
        missing = sorted(required - groups.keys())
        if missing:
            excluded.append({'id': mid, 'missing_configurations': missing})
            continue
        configs = {}
        for config in configurations:
            runs = groups[config['id']]
            configs[config['id']] = {
                'score': mean(r['task_score'] for r in runs),
                'n': len(runs),
                'final_best': mean(r['final_best'] for r in runs),
                'mean_api_cost_usd': complete_mean(r['cost_usd'] for r in runs),
                'mean_tokens': mean(r['tokens_total'] for r in runs),
                'mean_output_tokens': mean(r['usage']['output'] for r in runs),
                'axis': runs[0]['axis'],
                'curve': mean_curve(runs, curves),
            }
        base_scores = task_means({sid: c['score'] for sid, c in configs.items()}, configurations)
        costs = task_means({sid: c['mean_api_cost_usd'] for sid, c in configs.items()}, configurations)
        output_tokens = task_means({sid: c['mean_output_tokens'] for sid, c in configs.items()}, configurations)
        final_scores = task_means({sid: c['final_best'] for sid, c in configs.items()}, configurations)
        models.append({
            'id': mid, 'name': meta['display_name'], 'agent': meta['agent'], 'slug': meta['slug'],
            'harnesses': [{'name': agent, 'version': version} for agent, version in sorted({
                (r.get('agent') or meta['agent'], r.get('agent_version') or '')
                for runs in groups.values() for r in runs})],
            'release_date': meta['release_date'],
            'reasoning_effort': meta['reasoning_effort'], 'open_weights': meta['open_weights'],
            'synthetic': meta['synthetic'], 'score': mean(base_scores.values()),
            'interval': run_interval(groups, configurations, pool, resamples, seed),
            'mean_api_cost_usd': complete_mean(costs.values()),
            'mean_output_tokens': mean(output_tokens.values()),
            'mean_final_best': mean(final_scores.values()),
            'runs': sum(c['n'] for c in configs.values()), 'configurations': configs,
            'overall_curve': overall_curve(configs, configurations,
                                           common_axis['start'], common_axis['limit']),
        })
    models.sort(key=lambda m: (-m['score'], m['id']))
    return {
        'schema_version': 1, 'generated': data['generated'], 'mode': mode,
        'source_commit': data['source_commit'], 'task_count': len(tasks),
        'configuration_count': len(configurations), 'configurations': configurations,
        'legend_columns': legend_columns,
        'overall_axis': common_axis,
        'date_frontiers': date_frontiers(models, data['generated'][:10]),
        'uncertainty': {'unit': 'run', 'draws': resamples, 'seed': seed,
                        'confidence': .95, 'method': 'partially_pooled_variance',
                        'pool': pool, 'fixed_tasks': True,
                        'assumptions': 'Independent fresh runs; approximately normal scores; shared variance prior. '
                                       'Pool fitted within analysis mode, including incomplete models; '
                                       'pool estimation uncertainty is not propagated.'},
        'validation': {'valid_runs_checked': validated, 'tolerance': TOLERANCE},
        'models': models, 'excluded_models': excluded,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', nargs='?', type=Path,
                        default=ROOT / 'data/weirdml_v3_results.json')
    parser.add_argument('--output', type=Path, default=ROOT / 'assets/data/weirdml_v3.json')
    parser.add_argument('--mode', choices=['synthetic', 'real'], default='real')
    args = parser.parse_args()
    raw = args.input.read_bytes()
    output = prepare(json.loads(raw), mode=args.mode)
    output['input_sha256'] = hashlib.sha256(raw).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, separators=(',', ':'), allow_nan=False) + '\n')
    print(f"Prepared {len(output['models'])} complete models; "
          f"checked {output['validation']['valid_runs_checked']} valid runs. "
          f"Output: {args.output} ({args.output.stat().st_size:,} bytes)")


if __name__ == '__main__':
    main()
