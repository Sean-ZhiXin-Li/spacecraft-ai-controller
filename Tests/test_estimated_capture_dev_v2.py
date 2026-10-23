"""Numerical and information-boundary tests for position-aided capture."""
import hashlib
import inspect
import json
from pathlib import Path
import unittest

import numpy as np

from scripts import estimated_capture_dev_v2 as core


class EstimatorTests(unittest.TestCase):
    def setUp(self):
        self.state=np.array(core.model.S0,dtype=float)

    def test_constant_velocity_bias_identified_from_positions(self):
        case=core.ExperimentCase(51,navigation_velocity_bound=15)
        sensor=core.Sensor(case)
        estimator=core.CoastBiasEstimator(case.position_bound)
        estimator.update(sensor.measure(self.state,0),0)
        state=core.model.propagate(self.state,30*core.DAY).y[:,-1]
        estimator.update(sensor.measure(state,30),30)
        self.assertLess(np.linalg.norm(estimator.bias_estimate-sensor.bias[2:4]),.001)
        self.assertTrue(estimator.records[-1]['accepted'])

    def test_estimator_only_accepts_measurements_and_times(self):
        parameters=list(inspect.signature(core.CoastBiasEstimator.update).parameters)
        self.assertEqual(parameters,['self','raw','days'])
        source=inspect.getsource(core.CoastBiasEstimator)
        self.assertNotIn('sensor.bias',source)
        self.assertNotIn('true_state',source)

    def test_estimation_does_not_change_mass_or_position(self):
        estimator=core.CoastBiasEstimator(1e6)
        estimator.bias_estimate=np.array([5.,-7.])
        original=self.state.copy()
        output=estimator.corrected(self.state)
        np.testing.assert_array_equal(self.state,original)
        np.testing.assert_array_equal(output[:2],original[:2])
        self.assertEqual(output[4],original[4])

    def test_insufficient_interval_does_not_fabricate_estimate(self):
        estimator=core.CoastBiasEstimator(1e6)
        estimator.update(self.state,0)
        estimator.update(self.state,1/core.DAY)
        self.assertEqual(len(estimator.records),0)
        self.assertTrue(np.isinf(estimator.bound_mps))

    def test_same_measurements_produce_same_estimate(self):
        end=core.model.propagate(self.state,30*core.DAY).y[:,-1]
        first,second=core.CoastBiasEstimator(1e6),core.CoastBiasEstimator(1e6)
        for estimator in (first,second):
            estimator.update(self.state,0)
            estimator.update(end,30)
        np.testing.assert_array_equal(first.bias_estimate,second.bias_estimate)

    def test_sensor_bias_drift_is_observation_only(self):
        case=core.ExperimentCase(4,velocity_bias_drift_per_day=.2)
        sensor=core.Sensor(case)
        original=self.state.copy()
        observed=sensor.measure(self.state,10)
        np.testing.assert_array_equal(original,self.state)
        np.testing.assert_allclose(observed[2:4]-self.state[2:4],sensor.bias[2:4]+[2,-2])

    def test_bias_rate_is_extrapolated_to_current_epoch(self):
        case=core.ExperimentCase(9,navigation_velocity_bound=15,velocity_bias_drift_per_day=.4)
        sensor=core.Sensor(case)
        estimator=core.CoastBiasEstimator(case.position_bound,True)
        state=self.state.copy()
        for days in range(0,181,30):
            if days:
                state=core.model.propagate(state,30*core.DAY).y[:,-1]
            estimator.update(sensor.measure(state,days),days)
        actual=sensor.bias[2:4]+.4*180*sensor.drift_direction
        self.assertLess(np.linalg.norm(estimator.bias_estimate-actual),.001)

    def test_invalid_measurement_rejected(self):
        estimator=core.CoastBiasEstimator(1e6)
        with self.assertRaises(ValueError):
            estimator.update([0,0,float('nan'),0,722],0)

    def test_reversed_clock_rejected(self):
        estimator=core.CoastBiasEstimator(1e6)
        estimator.update(self.state,10)
        with self.assertRaises(ValueError):
            estimator.update(self.state,9)

    def test_raw_capture_matches_frozen_v1_case(self):
        root=Path(__file__).resolve().parents[1]
        path=root/'development_results/closed_loop_transfer_v1_2026-10-23_revision5/runs.jsonl'
        if not path.exists():
            self.skipTest('Frozen v1 result unavailable')
        old=next(json.loads(line) for line in path.read_text().splitlines()
                 if json.loads(line)['scenario']['seed']==43 and not json.loads(line)['feedback'])
        nominal=core.v1.plan_transfer(core.model.S0,0,110)
        new=core.simulate(core.ExperimentCase(43,250,.5,5e5,.6,30),nominal,False)
        np.testing.assert_allclose(new['final_state'],old['final_state'],rtol=1e-12,atol=1e-6)
        self.assertEqual(new['capture'],old['capture'])


class RecordedResultsTests(unittest.TestCase):
    def test_experiment_pairs_and_reserve_floor(self):
        root=Path(__file__).resolve().parents[1]
        for path in (root/'development_results').glob('estimated_capture_v2_2026-10-23_*'):
            if not (path/'summary.json').exists():
                continue
            summary=json.loads((path/'summary.json').read_text())
            self.assertEqual(summary['protected_before'],summary['protected_after'])
            rows=[json.loads(line) for line in (path/'runs.jsonl').read_text().splitlines()]
            for first,second in zip(rows[::2],rows[1::2]):
                self.assertEqual(first['case'],second['case'])
                self.assertEqual(first['initial_state'],second['initial_state'])
                self.assertEqual(first['departure_state'],second['departure_state'])
                self.assertEqual(first['delivered_error'],second['delivered_error'])
                self.assertEqual([s['true_state'] for s in first['state_samples']],
                                 [s['true_state'] for s in second['state_samples']])
                for row in (first,second):
                    self.assertFalse(row['reserve_violation'])
                    self.assertGreaterEqual(row['minimum_reserve_kg'],80-1e-7)

    def test_snapshot_matches_frozen_configuration(self):
        root=Path(__file__).resolve().parents[1]
        for path in (root/'development_results').glob('estimated_capture_v2_2026-10-23_*'):
            config=json.loads((path/'configuration.json').read_text())
            self.assertEqual(hashlib.sha256((path/'implementation_snapshot.py').read_bytes()).hexdigest(),
                             config['source_hash'])

    def test_published_result_hashes(self):
        root=Path(__file__).resolve().parents[1]
        for path in (root/'development_results').glob('estimated_capture_v2_2026-10-23_*'):
            manifest=path/'artifact_hashes.json'
            if manifest.exists():
                for name,expected in json.loads(manifest.read_text()).items():
                    self.assertEqual(hashlib.sha256((path/name).read_bytes()).hexdigest(),expected)


if __name__=='__main__':
    unittest.main()
