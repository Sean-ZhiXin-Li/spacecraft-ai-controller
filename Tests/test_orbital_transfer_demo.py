"""Check numerical recording fidelity, not visual appearance."""
import unittest
import numpy as np
from scripts import render_orbital_transfer_demo as demo


class DemoTests(unittest.TestCase):
    def test_recorder_preserves_original_propagator_and_endpoint(self):
        original = demo.m.model.propagate
        initial = np.array(demo.m.model.S0, dtype=float)
        expected = original(initial, 100., 1., 0., 1000.)
        with demo.Recorder() as recorder:
            actual = demo.m.model.propagate(initial, 100., 1., 0., 1000.)
        self.assertIs(demo.m.model.propagate, original)
        np.testing.assert_array_equal(actual.y[:, -1], expected.y[:, -1])
        np.testing.assert_array_equal(recorder.state_at(100.), expected.y[:, -1])
        self.assertEqual(recorder.segments[0]['phase'], 'departure')

    def test_recorder_rejects_extrapolated_frames(self):
        with demo.Recorder() as recorder:
            demo.m.model.propagate(np.array(demo.m.model.S0), 10.)
        with self.assertRaises(ValueError):
            recorder.state_at(11.)

    def test_mismatched_terminal_evidence_is_rejected(self):
        reference = {'final_state': [1., 2., 3., 4., 500.], 'completion_days': 5.,
                     'propellant_kg': 200., 'eccentricity': .0001,
                     'radius_error': .00001, 'survival_max_radius_error': .00002,
                     'minimum_propellant_kg': 200., 'status': 'capture',
                     'capture': True, 'survival': True, 'reserve_violation': False,
                     'physical_execution': True, 'departure_converged': True, 'crossing': True,
                     'crossing_state': [1., 2., 3., 4., 550.]}
        altered = dict(reference, final_state=[100., 2., 3., 4., 500.])
        with self.assertRaises(AssertionError):
            demo.validate_result(altered, reference)


if __name__ == '__main__':
    unittest.main()
