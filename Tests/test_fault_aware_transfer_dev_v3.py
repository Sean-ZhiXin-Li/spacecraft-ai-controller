import hashlib
import inspect
import json
from pathlib import Path
import unittest

import numpy as np

from scripts import fault_aware_transfer_dev_v3 as core


class FaultIsolationTests(unittest.TestCase):
    def histories(self, drift=0.):
        estimators=[core.FaultAwareEstimator(5e5),core.FaultAwareEstimator(5e5)]
        sensor_case=core.FaultCase(seed=801,position_noise_bound=0,fault_days=120,fault_kind='sensor',velocity_bias_drift_per_day=drift)
        physical_case=core.FaultCase(seed=801,position_noise_bound=0,fault_days=120,fault_kind='physical',velocity_bias_drift_per_day=drift)
        sensors=[core.FaultSensor(sensor_case),core.FaultSensor(physical_case)]
        state=np.array(core.model.S0,dtype=float)
        for days in (0,30,60,90):
            if days:
                state=core.model.propagate(state,30*core.DAY).y[:,-1]
            for estimator,sensor in zip(estimators,sensors):
                estimator.update(sensor.measure(state,days),days)
        state=core.model.propagate(state,30*core.DAY).y[:,-1]
        physical_state=state.copy()
        physical_state[2:4]+=sensors[1].fault_vector
        return estimators,sensors,[state,physical_state]

    def test_instantaneous_fault_sources_are_indistinguishable(self):
        estimators,sensors,states=self.histories()
        measurements=[sensor.measure(state,120) for sensor,state in zip(sensors,states)]
        np.testing.assert_allclose(measurements[0],measurements[1],rtol=0,atol=1e-10)
        for estimator,raw in zip(estimators,measurements):
            estimator.update(raw,120)
            self.assertIsNotNone(estimator.pending)
            self.assertEqual(estimator.fault_records[-1]['classification'],'ambiguous_velocity_jump')

    def test_position_baseline_disambiguates_sensor_and_physical_fault(self):
        estimators,sensors,states=self.histories()
        for estimator,sensor,state in zip(estimators,sensors,states):
            estimator.update(sensor.measure(state,120),120)
            state=core.model.propagate(state,30*core.DAY).y[:,-1]
            estimator.update(sensor.measure(state,150),150)
            self.assertIsNone(estimator.pending)
        self.assertEqual(estimators[0].fault_records[-1]['classification'],'sensor_bias_jump')
        self.assertEqual(estimators[1].fault_records[-1]['classification'],'physical_state_change')
        self.assertEqual(estimators[0].fault_records[-1]['diagnosis_latency_days'],30)

    def test_estimator_has_no_fault_labels_or_truth_input(self):
        source=inspect.getsource(core.FaultAwareEstimator)
        self.assertNotIn('fault_kind',source)
        self.assertNotIn('sensor.bias',source)
        self.assertNotIn('true_state',source)

    def test_drift_is_retained_while_new_bias_history_rebuilds(self):
        estimators,sensors,states=self.histories(.2)
        estimator,sensor,state=estimators[0],sensors[0],states[0]
        estimator.update(sensor.measure(state,120),120)
        for days in (150,180):
            state=core.model.propagate(state,30*core.DAY).y[:,-1]
            estimator.update(sensor.measure(state,days),days)
        actual=sensor.bias[2:4]+sensor.fault_vector+sensor.drift_direction*.2*180
        self.assertLess(np.linalg.norm(estimator.bias_estimate-actual),.001)

    def test_measurements_are_owned(self):
        estimator=core.FaultAwareEstimator(5e5)
        state=np.array(core.model.S0)
        estimator.update(state,0)
        state[0]=123
        self.assertNotEqual(estimator.previous[1][0],123)

    def test_steady_coast_does_not_trigger_fault(self):
        estimator=core.FaultAwareEstimator(5e5)
        state=np.array(core.model.S0,dtype=float)
        for days in range(0,301,30):
            if days:
                state=core.model.propagate(state,30*core.DAY).y[:,-1]
            estimator.update(state,days)
        self.assertEqual(estimator.fault_records,[])

    def test_input_error_remains_explicit(self):
        estimator=core.FaultAwareEstimator(5e5)
        with self.assertRaises(ValueError):
            estimator.update([0,0,np.nan,0,722],0)

    def test_no_fault_reference_matches_v2_trend_capture(self):
        case=core.FaultCase(seed=811,velocity_bound=10,navigation_velocity_bound=15,
                            position_noise_bound=2e5,velocity_bias_drift_per_day=.2,
                            fault_amplitude=0)
        nominal=core.v1.plan_transfer(core.model.S0,0,110)
        reference=core.v2.simulate(case,nominal,True,True)
        wrapped=core.simulate(case,nominal,False)
        np.testing.assert_allclose(reference['final_state'],wrapped['final_state'],rtol=1e-12,atol=1e-5)
        self.assertEqual(reference['capture'],wrapped['capture'])


class RecordedFaultExperiments(unittest.TestCase):
    def test_paired_conditions_and_reserve(self):
        root=Path(__file__).resolve().parents[1]
        for directory in (root/'development_results').glob('fault_aware_v3_2026-10-23_*'):
            if not (directory/'summary.json').exists():
                continue
            summary=json.loads((directory/'summary.json').read_text())
            self.assertEqual(summary['protected_before'],summary['protected_after'])
            rows=[json.loads(line) for line in (directory/'runs.jsonl').read_text().splitlines()]
            for baseline,aware in zip(rows[::2],rows[1::2]):
                self.assertEqual(baseline['case'],aware['case'])
                self.assertEqual(baseline['initial_state'],aware['initial_state'])
                self.assertEqual(baseline['departure_state'],aware['departure_state'])
                self.assertEqual(baseline['fault_vector_for_evaluation'],aware['fault_vector_for_evaluation'])
                self.assertEqual(aware['burns_during_ambiguity'],0)
                self.assertTrue(all(not e['pending'] for e in aware['correction_events'] if e['decision']=='correction'))
                self.assertFalse(aware['reserve_violation'])
                self.assertGreaterEqual(aware['minimum_reserve_kg'],80-1e-7)

    def test_executed_source_hash_matches_config(self):
        root=Path(__file__).resolve().parents[1]
        for directory in (root/'development_results').glob('fault_aware_v3_2026-10-23_*'):
            config=json.loads((directory/'configuration.json').read_text())
            self.assertEqual(hashlib.sha256((directory/'implementation_snapshot.py').read_bytes()).hexdigest(),
                             config['source_sha256'])

    def test_report_hashes(self):
        root=Path(__file__).resolve().parents[1]
        for directory in (root/'development_results').glob('fault_aware_v3_2026-10-23_*'):
            manifest=directory/'artifact_hashes.json'
            if manifest.exists():
                for name,expected in json.loads(manifest.read_text()).items():
                    self.assertEqual(hashlib.sha256((directory/name).read_bytes()).hexdigest(),expected)


if __name__=='__main__':
    unittest.main()
