"""Development v4: quarantine position intervals contaminated by asynchronous impulses."""
from __future__ import annotations
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from scripts import fault_aware_transfer_dev_v3 as v3
v2, v1, model, DAY = v3.v2, v3.v1, v3.model, v3.DAY
FaultCase, FaultSensor = v3.FaultCase, v3.FaultSensor


class QuarantinedEstimator(v3.FaultAwareEstimator):
    """Reject a suspect coast fit; retain its endpoint for independent confirmation."""
    def __init__(self, position_bound):
        super().__init__(position_bound)
        self.quarantined_intervals = []

    def update(self, raw, days):
        history = [(t, b.copy(), w) for t, b, w in self.bias_history]
        old_pending = self.pending
        prior = self._prior_bias(days) if self.previous else self.bias_estimate.copy()
        start = self.previous[0] if self.previous else None
        super().update(raw, days)
        if old_pending is None and self.pending is not None:
            self.bias_history = history
            self.bias_estimate = prior
            self.quarantined_intervals.append({'start_days': start, 'end_days': days,
                                               'reason': 'innovation_jump'})
            # The current raw endpoint remains available for the next clean interval.


class AsynchronousCoast:
    """Plant-only event splitting; fault epoch and vector never enter the estimator."""
    def __init__(self, case, vector):
        self.case, self.vector = case, np.array(vector).copy()
        self.applied_days = None

    def propagate(self, state, days, seconds, events):
        offset = (self.case.fault_days-days)*DAY
        physical = self.case.fault_kind in ('physical','combined')
        if physical and self.applied_days is None and 0 < offset <= seconds:
            first = model.propagate(state, offset, events=events)
            if first.status == 1:
                return first
            middle = first.y[:,-1].copy()
            middle[2:4] += self.vector
            self.applied_days = self.case.fault_days
            if seconds == offset:
                first.y[:,-1] = middle
                return first
            shifted=[]
            for event in events:
                def translated(t, z, original=event):
                    return original(t+offset, z)
                translated.terminal=event.terminal
                translated.direction=event.direction
                shifted.append(translated)
            second=model.propagate(middle, seconds-offset, events=tuple(shifted))
            return SimpleNamespace(y=np.column_stack((first.y,second.y)),
                t=np.r_[first.t, second.t+offset], status=second.status,
                t_events=[np.r_[a,b+offset] for a,b in zip(first.t_events,second.t_events)])
        if physical and self.applied_days is None and offset <= 0:
            raise RuntimeError('Physical fault epoch skipped by powered arc or runtime')
        return model.propagate(state,seconds,events=events)


def historical_snapshot(root):
    records=[]
    for pattern in ('closed_loop_transfer_v1_*','estimated_capture_v2_*','fault_aware_v3_*'):
        for directory in sorted((root/'development_results').glob(pattern)):
            for file in sorted(directory.rglob('*')):
                if file.is_file():
                    records.append((file.relative_to(root).as_posix(),
                                    hashlib.sha256(file.read_bytes()).hexdigest()))
    for file in sorted((root/'scripts').glob('*_dev_v[123].py')):
        records.append((file.relative_to(root).as_posix(),hashlib.sha256(file.read_bytes()).hexdigest()))
    return {'frozen':v3.historical_snapshot(root),
            'history_sha256':hashlib.sha256(json.dumps(records).encode()).hexdigest(),
            'history_files':len(records)}


def simulate(case, nominal, quarantine):
    rng=np.random.default_rng(case.seed)
    state=np.array(model.S0,dtype=float)
    state[2:4]+=rng.uniform(-case.velocity_bound,case.velocity_bound,2)
    initial=state.copy()
    delivered_error=rng.uniform(-case.velocity_bound,case.velocity_bound,2)
    sensor=FaultSensor(case)
    state,dt,departure_ok,floor=v1.burn(state,v2.GuidanceMeasurements(sensor,None,0),80,
                                      case.thrust_scale,v1.velocity_goal(np.array(model.S0),nominal))
    departure=state.copy()
    days=dt/DAY
    estimator=(QuarantinedEstimator if quarantine else v3.FaultAwareEstimator)(
        case.position_bound+case.position_noise_bound)
    plant=AsynchronousCoast(case, sensor.fault_vector)
    aware=True
    events=[]
    samples=[]
    delivered_done=False
    correction_dv=0.
    arrival_target=nominal.coast_days+30
    acquisition=False
    geometric=False
    capture_waits=0
    ambiguity_burns=0
    for index in range(100):
        if days>=120 and not delivered_done:
            state[2:4]+=delivered_error
            delivered_done=True
        raw=sensor.measure(state,days)
        estimator.update(raw,days)
        observed=estimator.corrected(raw)
        pending=bool(aware and estimator.pending is not None)
        forecast=v1.predict(observed,max_days=max(.001,1100-days))
        needs_correction=forecast is not None and days+forecast.coast_days>arrival_target
        correction=None
        if needs_correction and correction_dv<120 and not pending and not acquisition:
            correction=v1.plan_transfer(observed,days,82,deadline_days=arrival_target,
                                         correction_limit=120-correction_dv)
            if correction and np.hypot(correction.dvr,correction.dvt)>.1:
                mass_before=float(state[4])
                state,burn_seconds,ok,burn_floor=v1.burn(state,
                     v2.GuidanceMeasurements(sensor,estimator,days),80,case.thrust_scale,
                     v1.velocity_goal(observed,correction))
                days+=burn_seconds/DAY
                floor=min(floor,burn_floor)
                used=model.VE*np.log(mass_before/state[4])
                correction_dv+=used
                ambiguity_burns+=int(pending)
                events.append({'days':days,'decision':'correction','delivered_dv':float(used),
                               'plan':asdict(correction),'pending':pending,'converged':ok})
                estimator.previous=(days,sensor.measure(state,days))
        if pending:
            events.append({'days':days,'decision':'hold_for_position_confirmation'})
        samples.append({'days':days,'true_state':state.tolist(),'observed_state':observed.tolist(),
                        'pending':pending})
        if acquisition and not pending:
            break
        def crossing(t,z):
            return np.linalg.norm(sensor.measure(z,days+t/DAY)[:2])-model.TARGET_R
        crossing.terminal=True
        crossing.direction=1
        def real_crossing(t,z):
            return np.linalg.norm(z[:2])-model.TARGET_R
        real_crossing.terminal=False
        real_crossing.direction=0
        interval=7. if acquisition else case.observation_days
        interval=min(interval,1100-days)
        if interval<=0:
            break
        coast=plant.propagate(state, days, interval*DAY,
                 events=(real_crossing,) if acquisition else (crossing,real_crossing))
        state=coast.y[:,-1]
        days+=float(coast.t[-1])/DAY
        if acquisition:
            capture_waits+=1
            geometric|=bool(len(coast.t_events[0]))
        else:
            geometric|=bool(len(coast.t_events[1]))
            if len(coast.t_events[0]):
                acquisition=True
    mass_before_capture=float(state[4])
    if acquisition and not (aware and estimator.pending is not None):
        state,dt,converged,minimum=v1.burn(state,v2.GuidanceMeasurements(sensor,estimator,days),
                                           80,case.thrust_scale)
        floor=min(floor,minimum)
        days+=dt/DAY
    else:
        converged=False
    radius,vr,vt=model.components(state)
    ecc=model.eccentricity(state)
    captured=bool(converged and ecc<.001 and abs(radius/model.TARGET_R-1)<=.0025)
    hold=model.propagate(state,100*DAY,t_eval=np.linspace(0,100*DAY,101))
    max_error=max(abs(np.linalg.norm(s[:2])/model.TARGET_R-1) for s in hold.y.T)
    return {'case':asdict(case),'quarantine':quarantine,'initial_state':initial.tolist(),
            'departure_state':departure.tolist(),'fault_vector_for_evaluation':sensor.fault_vector.tolist(),
            'capture':captured,'terminal_eccentricity':ecc,'terminal_radial_velocity':vr,
            'terminal_tangential_error':vt-np.sqrt(model.MU/radius),
            'terminal_radius_error_ratio':abs(radius/model.TARGET_R-1),
            'remaining_propellant_kg':float(state[4]-300),'minimum_reserve_kg':floor-300,
            'reserve_violation':bool(floor<380-1e-7),'completion_days':days,
            'observed_crossing':acquisition,'geometric_crossing_during_coast':geometric,
            'survival':bool(captured and max_error<=.0025),'hold_max_error':max_error,
            'correction_dv':float(correction_dv),'correction_events':events,
            'fault_records':estimator.fault_records if aware else [],
            'estimator_records':estimator.records,'capture_wait_intervals':capture_waits,
            'burns_during_ambiguity':ambiguity_burns,
            'capture_propellant_used':mass_before_capture-state[4],
            'quarantined_intervals':getattr(estimator,'quarantined_intervals',[]),
            'actual_fault_epoch':plant.applied_days,'samples':samples,'final_state':state.tolist()}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment',choices=('development','holdout','sparse'),required=True)
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    output=root/f'development_results/asynchronous_fault_v4_2026-10-23_{args.experiment}'
    if output.exists():
        raise RuntimeError('Refusing to overwrite results')
    seeds=range(801,805) if args.experiment=='development' else range(901,913)
    cases=[FaultCase(seed=seed,velocity_bound=100,navigation_velocity_bound=15,
        position_bound=5e5,thrust_scale=.5,observation_days=30,
        velocity_bias_drift_per_day=.2,position_noise_bound=2e5,
        fault_kind=kind,fault_days=720+float(np.random.default_rng(seed+30000).uniform(2,28)),
        fault_amplitude=150) for seed in seeds for kind in ('sensor','physical','combined')]
    if args.experiment=='sparse':
        seeds=range(1001,1009)
        cases=[FaultCase(seed=seed,velocity_bound=100,navigation_velocity_bound=15,
            position_bound=5e5,thrust_scale=.5,observation_days=60,
            velocity_bias_drift_per_day=.2,position_noise_bound=2e5,
            fault_kind=kind,fault_days=600+float(np.random.default_rng(seed+30000).uniform(2,58)),
            fault_amplitude=150) for seed in seeds for kind in ('sensor','physical','combined')]
    before=historical_snapshot(root)
    output.mkdir(parents=True)
    config={'objective':'test whether excluding impulse-contaminated coast fits improves capture',
        'cases':[asdict(c) for c in cases],'baseline':'v3 estimator in identical asynchronous runtime',
        'reserve_kg':80,'deadline_days':1100,'capture_e_limit':.001,'radius_limit':.0025,
        'fault_epoch':'exact continuous-time split, independent of observation schedule',
        'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (output/'configuration.json').write_text(json.dumps(config,indent=2))
    (output/'implementation_snapshot.py').write_bytes(Path(__file__).read_bytes())
    nominal=v1.plan_transfer(model.S0,0,110)
    rows=[]
    for case in cases:
        pair=[]
        for quarantine in (False,True):
            result=simulate(case,nominal,quarantine)
            rows.append(result)
            pair.append(result)
            with (output/'runs.jsonl').open('a') as stream:
                stream.write(json.dumps(result,allow_nan=False)+'\n')
            print(case.seed,case.fault_kind,quarantine,'capture',result['capture'],
                  'e',round(result['terminal_eccentricity'],6),flush=True)
        assert pair[0]['initial_state']==pair[1]['initial_state']
        assert pair[0]['departure_state']==pair[1]['departure_state']
    after=historical_snapshot(root)
    if before!=after:
        raise RuntimeError('Historical evidence changed')
    summary={'pairs':len(cases),'runs':len(rows),'distinct_seeds':len(seeds),
             'protected_before':before,'protected_after':after,'groups':[]}
    for kind in ('sensor','physical','combined'):
        for quarantine in (False,True):
            group=[r for r in rows if r['case']['fault_kind']==kind and r['quarantine']==quarantine]
            summary['groups'].append({'kind':kind,'quarantine':quarantine,'n':len(group),
                'capture':sum(r['capture'] for r in group),'survival':sum(r['survival'] for r in group),
                'reserve_violations':sum(r['reserve_violation'] for r in group),
                'min_propellant':min(r['remaining_propellant_kg'] for r in group),
                'max_e':max(r['terminal_eccentricity'] for r in group)})
    (output/'summary.json').write_text(json.dumps(summary,indent=2))
    hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in output.iterdir() if p.is_file()}
    (output/'artifact_hashes.json').write_text(json.dumps(hashes,indent=2))
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    main()

