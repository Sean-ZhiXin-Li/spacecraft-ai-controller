import unittest
import hashlib
import json
from pathlib import Path
import numpy as np
from scripts import asynchronous_fault_dev_v4 as v4


class AsyncFaultTests(unittest.TestCase):
    def test_exact_physical_split(self):
        case=v4.FaultCase(seed=1,fault_kind='physical',fault_days=15)
        state=np.array(v4.model.S0,dtype=float)
        vector=np.array([40.,-20.])
        plant=v4.AsynchronousCoast(case,vector)
        result=plant.propagate(state,0,30*v4.DAY,())
        mid=v4.model.propagate(state,15*v4.DAY).y[:,-1].copy()
        mid[2:4]+=vector
        expected=v4.model.propagate(mid,15*v4.DAY).y[:,-1]
        np.testing.assert_array_equal(result.y[:,-1],expected)
        self.assertEqual(plant.applied_days,15)
        self.assertEqual(result.t[-1],30*v4.DAY)
        np.testing.assert_array_equal(state,v4.model.S0)

    def test_quarantine_does_not_admit_contaminated_fit(self):
        est=v4.QuarantinedEstimator(0)
        state=np.array(v4.model.S0,dtype=float)
        for day in (0,30,60,90):
            raw=v4.model.propagate(state,day*v4.DAY).y[:,-1] if day else state.copy()
            est.update(raw,day)
        old=[(t,b.copy(),w) for t,b,w in est.bias_history]
        raw=v4.model.propagate(state,120*v4.DAY).y[:,-1]
        raw[2:4]+=100
        est.update(raw,120)
        self.assertIsNotNone(est.pending)
        self.assertEqual(len(est.bias_history),len(old))
        for x,y in zip(est.bias_history,old):
            np.testing.assert_array_equal(x[1],y[1])
        self.assertEqual(len(est.quarantined_intervals),1)
        raw[:]=0
        self.assertNotEqual(est.previous[1][0],0)

    def test_skipped_epoch_fails_explicitly(self):
        plant=v4.AsynchronousCoast(v4.FaultCase(seed=1,fault_kind='physical',fault_days=10),[1,2])
        with self.assertRaises(RuntimeError):
            plant.propagate(np.array(v4.model.S0),11,v4.DAY,())

    def test_sensor_fault_does_not_change_plant(self):
        plant=v4.AsynchronousCoast(v4.FaultCase(seed=1,fault_kind='sensor',fault_days=10),[1,2])
        result=plant.propagate(np.array(v4.model.S0),0,30*v4.DAY,())
        expected=v4.model.propagate(v4.model.S0,30*v4.DAY)
        np.testing.assert_array_equal(result.y,expected.y)
        self.assertIsNone(plant.applied_days)

    def test_terminal_event_before_impulse_prevents_injection(self):
        plant=v4.AsynchronousCoast(v4.FaultCase(seed=1,fault_kind='physical',fault_days=15),[40,20])
        def stop(t,z):
            return t-10*v4.DAY
        stop.terminal=True
        stop.direction=1
        result=plant.propagate(np.array(v4.model.S0),0,30*v4.DAY,(stop,))
        self.assertAlmostEqual(result.t[-1]/v4.DAY,10)
        self.assertIsNone(plant.applied_days)

    def test_event_clock_is_continuous_across_impulse(self):
        plant=v4.AsynchronousCoast(v4.FaultCase(seed=1,fault_kind='physical',fault_days=15),[40,20])
        def stop(t,z):
            return t-22*v4.DAY
        stop.terminal=True
        stop.direction=1
        result=plant.propagate(np.array(v4.model.S0),0,30*v4.DAY,(stop,))
        self.assertAlmostEqual(result.t_events[0][0]/v4.DAY,22)
        self.assertAlmostEqual(result.t[-1]/v4.DAY,22)

    def test_no_fault_matches_v3(self):
        a=v4.QuarantinedEstimator(7e5)
        b=v4.v3.FaultAwareEstimator(7e5)
        state=np.array(v4.model.S0)
        for day in range(0,181,30):
            raw=v4.model.propagate(state,day*v4.DAY).y[:,-1] if day else state.copy()
            raw[2:4]+=np.array([5.,-3.])+day*.2
            a.update(raw,day)
            b.update(raw,day)
            np.testing.assert_array_equal(a.corrected(raw),b.corrected(raw))
        self.assertEqual(a.quarantined_intervals,[])

    def test_experiment_integrity_and_pairing(self):
        from scripts.report_asynchronous_fault_dev_v4 import analyze
        root=Path(__file__).resolve().parents[1]
        for label in ('development','holdout','sparse'):
            directory=root/f'development_results/asynchronous_fault_v4_2026-10-23_{label}'
            if not directory.exists():
                continue
            for filename in ('artifact_hashes.json','report_hashes.json'):
                hashes=json.loads((directory/filename).read_text())
                for name,digest in hashes.items():
                    self.assertEqual(hashlib.sha256((directory/name).read_bytes()).hexdigest(),digest)
            rows=[json.loads(line) for line in (directory/'runs.jsonl').read_text().splitlines()]
            report=analyze(rows)
            self.assertEqual(report,json.loads((directory/'comparison.json').read_text()))
            self.assertEqual(report['unequal_exposure'],0)
            summary=json.loads((directory/'summary.json').read_text())
            self.assertEqual(summary['protected_before'],summary['protected_after'])
            for row in rows:
                if row['actual_fault_epoch'] is not None:
                    self.assertEqual(row['actual_fault_epoch'],row['case']['fault_days'])
                self.assertFalse(row['reserve_violation'])
                self.assertEqual(row['burns_during_ambiguity'],0)


if __name__=='__main__':
    unittest.main()
