"""Numerical contracts for the isolated continuous development model."""
import math
import hashlib
import json
from pathlib import Path
import unittest

import numpy as np

from scripts import closed_loop_transfer_dev_v1 as core


class ClosedLoopTransferTests(unittest.TestCase):
    def test_predictor_matches_independent_continuous_crossing(self):
        state = np.array(core.model.S0)
        prediction = core.predict(state, 100, 20)
        er = state[:2]/np.linalg.norm(state[:2])
        state[2:4] += 100*er + 20*np.array([-er[1], er[0]])
        integrated = core.model.propagate(state, 1200*core.DAY, events=core.model.outward_crossing)
        self.assertLess(abs(integrated.t[-1]/core.DAY-prediction.coast_days), 1e-5)

    def test_arbitrary_state_rotation_covariance(self):
        state = np.array(core.model.S0)
        state = core.model.propagate(state, 100*core.DAY).y[:, -1]
        angle = .7
        rotation = np.array([[math.cos(angle), -math.sin(angle)],
                             [math.sin(angle), math.cos(angle)]])
        rotated = np.r_[rotation@state[:2], rotation@state[2:4], state[4]]
        first, second = core.predict(state), core.predict(rotated)
        self.assertAlmostEqual(first.coast_days, second.coast_days, places=7)
        self.assertAlmostEqual(first.total_dv, second.total_dv, places=7)

    def test_mass_budget_is_current_mass(self):
        state = np.array(core.model.S0)
        first = core.predict(state)
        state[4] -= 100
        second = core.predict(state)
        self.assertLess(second.predicted_final_mass, first.predicted_final_mass)

    def test_elapsed_mission_deadline_is_not_reset(self):
        self.assertIsNone(core.plan_transfer(core.model.S0, 1100, 80))

    def test_invalid_state_rejected(self):
        self.assertIsNone(core.plan_transfer([0, 1, 2, float('nan'), 722], 0, 80))
        self.assertIsNone(core.plan_transfer([0, 1, 2, 3], 0, 80))
        self.assertIsNone(core.plan_transfer([0, 0, 2, 3, 722], 0, 80))
        self.assertIsNone(core.predict([0,0,2,3,722]))

    def test_unavailable_fuel_returns_no_plan(self):
        state = np.array(core.model.S0)
        state[4] = core.model.DRY_MASS + 80
        self.assertIsNone(core.plan_transfer(state, 0, 80))

    def test_partial_burn_respects_operational_mass_floor(self):
        state = np.array(core.model.S0)
        state[4] = 380.1
        nav = core.Navigation(core.Scenario('test', 1, 0, 0, 0))
        result, duration, done, floor = core.burn(state, nav, 80, 1.3,
                                                goal=state[2:4]+[100, 100])
        self.assertFalse(done)
        self.assertGreater(duration, 0)
        self.assertGreaterEqual(floor, 380-1e-7)
        self.assertLessEqual(result[4], state[4])

    def test_navigation_returns_owned_biased_measurement(self):
        scenario = core.Scenario('test', 7, 10, 1, 1e6)
        nav = core.Navigation(scenario)
        state = np.array(core.model.S0)
        observed = nav.observe(state)
        self.assertFalse(np.array_equal(observed[:4], state[:4]))
        observed[0] = 0
        self.assertEqual(state[1], core.model.S0[1])
        self.assertEqual(nav.bias[4], 0)

    def test_navigation_reproducible(self):
        scenario = core.Scenario('test', 9, 100, 1, 1e6)
        np.testing.assert_array_equal(core.Navigation(scenario).bias,
                                      core.Navigation(scenario).bias)

    def test_surrogate_and_finite_capture_are_distinct(self):
        state = np.array(core.model.S0)
        self.assertGreater(core.model.eccentricity(state), .001)
        self.assertGreater(core.predict(state).capture_dv, 0)

    def test_replan_can_enter_feasible_deadline_from_late_initial_guess(self):
        state = np.array(core.model.S0)
        nominal = core.plan_transfer(state, 0, 110)
        goal = core.velocity_goal(state, nominal)
        nav = core.Navigation(core.Scenario('test',0,0,0,0))
        state, seconds, _, _ = core.burn(state,nav,80,1,goal)
        nominal_arrival = core.predict(state).coast_days
        state[3] -= 150
        self.assertGreater(core.predict(state).coast_days, nominal_arrival+30)
        correction = core.plan_transfer(state, seconds/core.DAY, 82,
                                       deadline_days=nominal_arrival+30,
                                       correction_limit=120)
        self.assertIsNotNone(correction)
        self.assertLessEqual(math.hypot(correction.dvr,correction.dvt),120+1e-5)
        self.assertGreaterEqual(correction.predicted_final_mass,382)

    def test_input_state_not_mutated_by_planning(self):
        state=np.array(core.model.S0)
        original=state.copy()
        core.plan_transfer(state,0,110)
        np.testing.assert_array_equal(state,original)


class RecordedExperimentContracts(unittest.TestCase):
    def test_fresh_validation_pairs_and_protected_bytes(self):
        root=Path(__file__).resolve().parents[1]
        directory=root/'development_results/closed_loop_transfer_v1_2026-10-23_revision4'
        if not directory.exists():
            self.skipTest('Fresh-seed development experiment has not been executed')
        summary=json.loads((directory/'summary.json').read_text())
        self.assertEqual(summary['protected_before'],summary['protected_after'])
        rows=[json.loads(line) for line in (directory/'runs.jsonl').read_text().splitlines()]
        self.assertEqual(len(rows),32)
        for name in {r['scenario']['name'] for r in rows}:
            pair=[r for r in rows if r['scenario']['name']==name]
            self.assertEqual(pair[0]['initial_state'],pair[1]['initial_state'])
            self.assertEqual(pair[0]['scenario'],pair[1]['scenario'])
            self.assertEqual(pair[0]['navigation_bias'],pair[1]['navigation_bias'])
            for run in pair:
                self.assertFalse(run['reserve_violation'])
                self.assertGreaterEqual(run['minimum_reserve_kg'],80-1e-7)
                self.assertLessEqual(run['final_state'][4],run['initial_state'][4])
                for event in run['replanning_events']:
                    if event['decision']=='bounded_correction':
                        self.assertLess(event['observed_state'][4],722)
                        self.assertGreater(event['elapsed_days'],0)

    def test_saved_implementation_and_artifact_hashes(self):
        root=Path(__file__).resolve().parents[1]
        for directory in (root/'development_results').glob('closed_loop_transfer_v1_2026-10-23*'):
            hashes_path=directory/'artifact_hashes.json'
            if not hashes_path.exists():
                continue
            for name,expected in json.loads(hashes_path.read_text()).items():
                self.assertEqual(hashlib.sha256((directory/name).read_bytes()).hexdigest(),expected)

    def test_fixed_departure_cruise_ablation(self):
        root=Path(__file__).resolve().parents[1]
        directory=root/'development_results/closed_loop_transfer_v1_2026-10-23_revision5'
        if not (directory/'summary.json').exists():
            self.skipTest('Cruise ablation not complete')
        rows=[json.loads(line) for line in (directory/'runs.jsonl').read_text().splitlines()]
        self.assertEqual(len(rows),48)
        for name in {r['scenario']['name'] for r in rows}:
            baseline=next(r for r in rows if r['scenario']['name']==name and not r['feedback'])
            feedback=next(r for r in rows if r['scenario']['name']==name and r['feedback'])
            self.assertEqual(baseline['initial_state'],feedback['initial_state'])
            self.assertEqual(baseline['delivered_velocity_error'],feedback['delivered_velocity_error'])
            self.assertEqual(baseline['trajectory_samples'][0]['true_state'],
                             feedback['trajectory_samples'][0]['true_state'])
            for event in feedback['replanning_events']:
                if event['decision']=='bounded_correction':
                    self.assertGreaterEqual(event['elapsed_days'],120)


if __name__ == '__main__':
    unittest.main()
