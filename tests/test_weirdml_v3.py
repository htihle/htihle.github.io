"""Scoring invariants and eligibility tests, independent of UI implementation."""
import copy
import importlib.util
import json
import math
from pathlib import Path
import random
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('prepare', ROOT / 'scripts/prepare_weirdml_v3.py')
pipeline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pipeline)


def run_fixture(points, area, final, axis='tokens'):
    return {'sample_key': 'test:task', 'axis': {'kind': axis, 'start': 1, 'limit': 100,
            'final_best_weight': .2}, 'submissions': [
                {axis: x, 'effective': y, 'running_max': max(v for xx, v in points if xx <= x)}
                for x, y in points], 'area_term': area, 'final_best': final,
            'task_score': .8 * area + .2 * final}


class ScoringTests(unittest.TestCase):
    def test_frontier_shading_before_open_release_and_after_crossing(self):
        models = [
            {'id': 'closed', 'release_date': '2026-01-01', 'open_weights': False, 'score': .5},
            {'id': 'open', 'release_date': '2026-02-01', 'open_weights': True, 'score': .3},
            {'id': 'open-best', 'release_date': '2026-03-01', 'open_weights': True, 'score': .7},
        ]
        result = pipeline.date_frontiers(models, '2026-04-01')
        self.assertEqual(result['gap'], [
            ['2026-01-01', 0, .5],
            ['2026-02-01', 0, .5], ['2026-02-01', .3, .5],
            ['2026-03-01', .3, .5], ['2026-03-01', .7, .7],
            ['2026-04-01', .7, .7],
        ])

    def test_date_frontiers_use_best_available_model_and_skip_undated(self):
        models = [
            {'id': 'open-low', 'release_date': '2026-01-01', 'open_weights': True, 'score': .2},
            {'id': 'open-best', 'release_date': '2026-01-01', 'open_weights': True, 'score': .4},
            {'id': 'open-later', 'release_date': '2026-02-01', 'open_weights': True, 'score': .3},
            {'id': 'closed', 'release_date': '2026-02-01', 'open_weights': False, 'score': .7},
            {'id': 'unknown', 'release_date': None, 'open_weights': False, 'score': .9},
        ]
        result = pipeline.date_frontiers(models, '2026-03-01')
        self.assertEqual(result['open'], [['2026-01-01', 0], ['2026-01-01', .4], ['2026-03-01', .4]])
        self.assertEqual(result['closed'][-1], ['2026-03-01', .7])
        self.assertEqual(result['gap'], [['2026-02-01', .4, .7], ['2026-03-01', .4, .7]])
        self.assertEqual(result['undated_models'], ['unknown'])
        self.assertEqual(set(result['frontier_ids']), {'open-best', 'closed'})

    def test_log_area_carries_to_limit(self):
        run = run_fixture([(10, .8)], .4, .8)
        self.assertEqual(pipeline.validate_run(run), [[1, 0], [10, .8], [100, .8]])

    def test_pre_window_and_worse_later_submission(self):
        run = run_fixture([(.5, .6), (10, .2)], .6, .6)
        self.assertEqual(pipeline.validate_run(run), [[1, .6], [100, .6]])

    def test_plot_history_preserves_early_scores_without_changing_scoring(self):
        run = run_fixture([(.25, .3), (.5, .6), (10, .2)], .6, .6)
        self.assertEqual(pipeline.curve(run, full_history=True),
                         [[0, 0], [.25, .3], [.5, .6], [100, .6]])
        self.assertEqual(pipeline.validate_run(run), [[1, .6], [100, .6]])

    def test_unscored_checkpoint_keeps_previous_best(self):
        run = run_fixture([(10, .8)], .4, .8)
        run['submissions'].append({'tokens': 50, 'scored': False,
                                   'effective': None, 'running_max': None})
        self.assertEqual(pipeline.validate_run(run), [[1, 0], [10, .8], [100, .8]])

    def test_limit_submission_gets_final_weight_but_no_area(self):
        pipeline.validate_run(run_fixture([(100, 1)], 0, 1))

    def test_cost_weighted_axis(self):
        pipeline.validate_run(run_fixture([(10, .8)], .4, .8, 'cost_weighted'))

    def test_corrupt_area_fails(self):
        with self.assertRaisesRegex(ValueError, 'log area'):
            pipeline.validate_run(run_fixture([(10, .8)], .6, .8))

    def test_twins_receive_one_task_weight(self):
        configs = [{'id': 'a', 'task': 'a'}, {'id': 'a-nohint', 'task': 'a'},
                   {'id': 'b', 'task': 'b'}]
        values = pipeline.task_means({'a': 1, 'a-nohint': 0, 'b': 0}, configs)
        self.assertEqual(values, {'a': .5, 'b': 0})
        self.assertEqual(sum(values.values()) / len(values), .25)

    def test_overall_curve_scales_cost_axis_and_weights_twins(self):
        configurations = [{'id': 'a', 'task': 'a'}, {'id': 'a-nohint', 'task': 'a'},
                          {'id': 'ship', 'task': 'ship'}]
        configs = {
            'a': {'axis': {'start': 1, 'limit': 100}, 'curve': [[1, 1], [100, 1]]},
            'a-nohint': {'axis': {'start': 1, 'limit': 100}, 'curve': [[1, 0], [100, 0]]},
            'ship': {'axis': {'start': .04, 'limit': 4},
                     'curve': [[.04, 0], [.4, 1], [4, 1]]},
        }
        self.assertEqual(pipeline.overall_curve(configs, configurations, 1, 100),
                         [[1, .25], [10, .75], [100, .75]])

    def test_pool_ignores_differences_in_task_means(self):
        grouped = {'m': {'a': [{'task_score': x} for x in [.1, .3]],
                         'b': [{'task_score': x} for x in [.7, .9]]}}
        self.assertAlmostEqual(pipeline.variance_pool(grouped)['variance'], .02)

    def test_singletons_and_more_runs(self):
        configs = [{'id': 'a', 'task': 'a'}]
        pool = {'variance': .01, 'prior_df': 4}
        widths = []
        for n in [1, 4, 20]:
            groups = {'a': [{'task_score': .5}] * n}
            lo, hi = pipeline.run_interval(groups, configs, pool, 10000, 42)
            self.assertLess(lo, .5)
            self.assertGreater(hi, .5)
            widths.append(hi - lo)
        self.assertGreater(widths[0], widths[1])
        self.assertGreater(widths[1], widths[2])

    def test_no_repeats_has_explicit_nonzero_fallback(self):
        pool = pipeline.variance_pool({'m': {'a': [{'task_score': .5}]}})
        self.assertTrue(pool['fallback'])
        self.assertEqual(pool['variance'], .25)



SYNTHETIC_FIXTURE = ROOT / 'data/weirdml_v3_results_with_synthetic.json'


@unittest.skipUnless(SYNTHETIC_FIXTURE.exists(), 'synthetic export fixture not present (kept out of the published repository)')
class ExportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = json.loads(SYNTHETIC_FIXTURE.read_text())

    def test_full_export_and_determinism(self):
        result = pipeline.prepare(self.data, resamples=100)
        self.assertEqual(result, pipeline.prepare(self.data, resamples=100))
        self.assertEqual(len(result['models']), 2)
        self.assertEqual(result['validation']['valid_runs_checked'], 272)
        self.assertTrue(all(m['runs'] == 75 for m in result['models']))
        self.assertEqual((result['task_count'], result['configuration_count']), (11, 15))

    def test_canonical_task_order_and_twin_markers_ignore_input_order(self):
        data = copy.deepcopy(self.data)
        data['suite']['sample_ids'].reverse()
        result = pipeline.prepare(data, resamples=100)
        self.assertEqual(result, pipeline.prepare(self.data, resamples=100))
        configs = {c['id']: c for c in result['configurations']}
        columns = result['legend_columns']
        self.assertEqual([len(column) for column in columns], [2] * 7 + [1])
        self.assertEqual([sid for column in columns for sid in column], list(configs))
        for task in ['mystery_box', 'reaction_rates', 'scan_stitch', 'shattered_prior']:
            self.assertIn([task, task + '--nohints'], columns)
            hinted, hintless = configs[task]['marker'], configs[task + '--nohints']['marker']
            self.assertEqual((hinted['shape'], hinted['rotation']), (hintless['shape'], hintless['rotation']))
            self.assertTrue(hinted['filled'])
            self.assertFalse(hintless['filled'])
        shapes = {(c['marker']['shape'], c['marker']['rotation']) for c in configs.values()}
        self.assertEqual(len(shapes), 11)

    def test_missing_twin_or_only_invalid_runs_excludes_model(self):
        for invalid in (False, True):
            data = copy.deepcopy(self.data)
            matching = lambda r: r['model_id'] == 'synthetic-strong' and r['sample_id'] == 'mystery_box--nohints'
            if invalid:
                for run in data['samples']:
                    if matching(run):
                        run['verdict'] = 'invalid'
            else:
                data['samples'] = [r for r in data['samples'] if not matching(r)]
            result = pipeline.prepare(data, resamples=100)
            self.assertEqual([m['id'] for m in result['models']], ['synthetic-weak'])

    def test_real_mode_does_not_fill_gaps_with_synthetic_runs(self):
        result = pipeline.prepare(self.data, mode='real', resamples=100)
        self.assertEqual(result['models'], [])
        self.assertEqual(len(result['excluded_models']), 7)

    def test_synthetic_singleton_model_borrows_only_synthetic_variance(self):
        data = copy.deepcopy(self.data)
        seen = set()
        def keep(run):
            if run['model_id'] != 'synthetic-strong':
                return True
            sid = run['sample_id']
            if sid in seen:
                return False
            seen.add(sid)
            return True
        data['samples'] = [r for r in data['samples'] if keep(r)]
        result = pipeline.prepare(data, resamples=1000)
        model = next(m for m in result['models'] if m['id'] == 'synthetic-strong')
        self.assertEqual(model['runs'], 15)
        self.assertLess(model['interval'][0], model['score'])
        self.assertGreater(model['interval'][1], model['score'])
        data['samples'] = [r for r in data['samples'] if r['synthetic']]
        self.assertEqual(result['uncertainty'], pipeline.prepare(data, resamples=1000)['uncertainty'])
        self.assertEqual(result['models'], pipeline.prepare(data, resamples=1000)['models'])

    def test_harness_versions_include_all_valid_runs(self):
        data = copy.deepcopy(self.data)
        runs = [r for r in data['samples'] if r['model_id'] == 'synthetic-strong']
        for r in runs:
            r['agent'] = 'codex_cli'
            r['agent_version'] = '1.0'
        runs[0]['agent_version'] = '2.0'
        result = pipeline.prepare(data, resamples=100)
        model = next(m for m in result['models'] if m['id'] == 'synthetic-strong')
        self.assertEqual(model['harnesses'], [{'name': 'codex_cli', 'version': '1.0'},
                                             {'name': 'codex_cli', 'version': '2.0'}])

    def test_missing_cost_keeps_scores_and_does_not_average_partial_costs(self):
        original = pipeline.prepare(self.data, resamples=100)
        data = copy.deepcopy(self.data)
        run = next(r for r in data['samples'] if r['model_id'] == 'synthetic-strong')
        run['cost_usd'] = None
        result = pipeline.prepare(data, resamples=100)
        model = next(m for m in result['models'] if m['id'] == 'synthetic-strong')
        before = next(m for m in original['models'] if m['id'] == model['id'])
        self.assertIsNone(model['mean_api_cost_usd'])
        self.assertEqual(model['score'], before['score'])
        self.assertEqual(model['interval'], before['interval'])

    def test_integrated_overall_curve_reproduces_leaderboard_score(self):
        result = pipeline.prepare(self.data, resamples=100)
        for model in result['models']:
            points = model['overall_curve']
            axis = result['overall_axis']
            self.assertEqual(points[0][0], 0)
            self.assertEqual(points[-1][0], axis['limit'])
            area = sum(y * math.log(x2 / max(x, axis['start']))
                       for (x, y), (x2, _) in zip(points, points[1:]) if x2 > axis['start'])
            area /= math.log(axis['limit'] / axis['start'])
            self.assertAlmostEqual(.8 * area + .2 * points[-1][1], model['score'], places=5)


if __name__ == '__main__':
    unittest.main()
