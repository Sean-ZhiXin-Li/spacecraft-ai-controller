import unittest
import math
import json
import hashlib
from pathlib import Path
import numpy as np
from scripts import multiregime_transfer_research as m


class TransferTests(unittest.TestCase):
    def state(self,ratio,vr,vt_ratio=1):
        r=ratio*m.model.TARGET_R
        return np.array([r,0,vr,vt_ratio*math.sqrt(m.model.MU/r),722])

    def test_analytic_matches_numerical_crossings(self):
        for ratio,vr in ((.98,400),(1.02,-400),(.999,-10),(1.001,10)):
            state=self.state(ratio,vr,.95 if ratio>1 else 1.05)
            p=m.elliptic_prediction(state)
            self.assertIsNotNone(p)
            def event(t,z):
                return np.linalg.norm(z[:2])-m.model.TARGET_R
            event.terminal=True
            event.direction=0
            sol=m.model.propagate(state,p.coast_days*m.DAY*1.01,events=event)
            self.assertTrue(len(sol.t_events[0]))
            self.assertAlmostEqual(sol.t_events[0][0]/m.DAY,p.coast_days,delta=1e-3)
            r,vr1,vt1=m.model.components(sol.y[:,-1])
            self.assertAlmostEqual(math.hypot(vr1,vt1-m.model.VC),p.capture_dv,delta=1e-3)

    def test_agrees_with_outward_quadrature(self):
        state=self.state(.98,400)
        p=m.elliptic_prediction(state)
        q=m.legacy.predict(state,max_days=200000)
        self.assertAlmostEqual(p.coast_days,q.coast_days,places=6)
        self.assertAlmostEqual(p.capture_dv,q.capture_dv,places=6)

    def test_unbound_is_explicitly_unsupported(self):
        self.assertIsNone(m.elliptic_prediction(self.state(1.02,10000)))

    def test_rotation_invariance(self):
        state=self.state(1.02,-400)
        rotated=state.copy()
        rotated[:2]=[-state[1],state[0]]
        rotated[2:4]=[-state[3],state[2]]
        self.assertEqual(m.elliptic_prediction(state),m.elliptic_prediction(rotated))

    def test_disjoint_seed_sets(self):
        a={r['seed'] for r in m.tasks('development')}
        b={r['seed'] for r in m.tasks('holdout')}
        self.assertFalse(a&b)

    def test_no_plan_executes_no_trajectory(self):
        r=m.execute(self.state(.98,400),None,1,1)
        self.assertFalse(r['physical_execution'])

    def test_invalid_cartesian_evidence_rejected(self):
        self.assertIsNone(m.elliptic_prediction([0,0,1,1,722]))
        self.assertIsNone(m.elliptic_prediction([1,1,float('nan'),1,722]))

    def test_outside_plan_finite_burn_capture(self):
        state=self.state(1.02,-200)
        plan=m.search(state,deadline=4400)
        self.assertIsNotNone(plan)
        self.assertGreaterEqual(plan.predicted_final_mass,410.5-1e-5)
        result=m.execute(state,plan,5,.7,4400)
        self.assertTrue(result['capture'])
        self.assertTrue(result['survival'])
        self.assertFalse(result['reserve_violation'])

    def test_recorded_experiments_and_protection(self):
        root=Path(__file__).resolve().parents[1]
        for label in ('development','extended_development','holdout'):
            directory=root/f'development_results/multiregime_transfer_2026-10-23_{label}'
            if not directory.exists():
                continue
            for filename in ('artifact_hashes.json','report_hashes.json'):
                if not (directory/filename).exists():
                    continue
                for name,digest in json.loads((directory/filename).read_text()).items():
                    self.assertEqual(hashlib.sha256((directory/name).read_bytes()).hexdigest(),digest)
            summary=json.loads((directory/'summary.json').read_text())
            self.assertEqual(summary['protected_before'],summary['protected_after'])
            rows=[json.loads(line) for line in (directory/'runs.jsonl').read_text().splitlines()]
            self.assertEqual(len(rows),summary['comparisons'])
            for i in range(0,len(rows),3):
                self.assertEqual(rows[i]['case'],rows[i+1]['case'])
                self.assertEqual(rows[i]['case'],rows[i+2]['case'])


if __name__=='__main__':
    unittest.main()
