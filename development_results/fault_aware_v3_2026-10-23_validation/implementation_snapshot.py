"""Delayed position-confirmed fault isolation for continuous transfer/capture."""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

import numpy as np

from scripts import estimated_capture_dev_v2 as v2

v1, model, DAY = v2.v1, v2.model, v2.DAY


@dataclass(frozen=True)
class FaultCase(v2.ExperimentCase):
    fault_kind: str = 'sensor'
    fault_days: float = 660.
    fault_amplitude: float = 80.


class FaultSensor(v2.Sensor):
    def __init__(self, case):
        super().__init__(case)
        rng=np.random.default_rng(case.seed+20000)
        direction=rng.normal(size=2)
        self.fault_vector=direction/np.linalg.norm(direction)*case.fault_amplitude

    def measure(self, state, days):
        result=super().measure(state,days)
        if days>=self.case.fault_days and self.case.fault_kind in ('sensor','combined'):
            result[2:4]+=self.fault_vector
        return result


class FaultAwareEstimator(v2.CoastBiasEstimator):
    """Wait one unpowered position baseline before assigning a jump's cause."""
    def __init__(self, position_bound):
        super().__init__(position_bound,True)
        self.pending=None
        self.fault_records=[]

    def _prior_bias(self, days):
        if len(self.bias_history)<3:
            return self.bias_estimate.copy()
        history=self.bias_history[-6:]
        epochs=np.array([p[0]-days for p in history])
        weights=np.array([p[2] for p in history])
        design=np.column_stack((np.ones(len(history)),epochs))
        fit=np.linalg.lstsq(design*weights[:,None],
                            np.array([p[1] for p in history])*weights[:,None],rcond=None)[0]
        return fit[0]

    def update(self, raw, days):
        raw=np.asarray(raw).copy()
        previous=self.previous
        if previous is None:
            super().update(raw,days)
            return
        previous_days, previous_raw=previous
        dt=(days-previous_days)*DAY
        if dt<DAY:
            super().update(raw,days)
            return
        prior_previous_bias=self._prior_bias(previous_days)
        prior_current_bias=self._prior_bias(days)
        initial=previous_raw.copy()
        initial[2:4]-=prior_previous_bias
        expected=model.propagate(initial,dt).y[:,-1]
        innovation=raw[2:4]-expected[2:4]-prior_current_bias
        # Position displacement bound plus explicit engineering model-error allowance.
        gate=4*np.sqrt(2)*self.position_bound/dt+1.
        history_count=len(self.bias_history)
        super().update(raw,days)
        fitted=bool(len(self.bias_history)>history_count)
        if self.pending is not None and fitted:
            reference_days,fitted_bias,_=self.bias_history[-1]
            shift=fitted_bias-self.pending['prior_bias_at_detection']
            magnitude=float(np.linalg.norm(shift))
            if magnitude>gate:
                physical_residual=self.pending['innovation']-shift
                classification='sensor_bias_jump' if np.linalg.norm(physical_residual)<gate*2 else 'combined_or_model_mismatch'
                # Reset obsolete history; retain the pre-jump drift rate only as a prior.
                drift=(prior_current_bias-prior_previous_bias)/(days-previous_days)
                self.bias_history=[(reference_days,fitted_bias.copy(),dt/DAY)]
                self.bias_estimate=fitted_bias+drift*(days-reference_days)
            else:
                classification='physical_state_change'
            self.fault_records.append({'detected_days':self.pending['days'],
                 'confirmed_days':days,'classification':classification,
                 'bias_shift_mps':magnitude,'gate_mps':gate,
                 'diagnosis_latency_days':days-self.pending['days']})
            self.pending=None
            return
        if history_count>=3 and np.linalg.norm(innovation)>gate:
            self.pending={'days':days,'innovation':innovation.copy(),
                          'prior_bias_at_detection':prior_current_bias.copy()}
            self.fault_records.append({'detected_days':days,'classification':'ambiguous_velocity_jump',
                 'innovation_mps':float(np.linalg.norm(innovation)),'gate_mps':gate})


def historical_snapshot(root):
    base=v2.research_snapshot(root)
    records=[]
    for directory in sorted((root/'development_results').glob('estimated_capture_v2_*')):
        for file in sorted(directory.rglob('*')):
            if file.is_file():
                records.append((file.relative_to(root).as_posix(),hashlib.sha256(file.read_bytes()).hexdigest()))
    for name in ('scripts/estimated_capture_dev_v2.py','scripts/report_estimated_capture_dev_v2.py'):
        records.append((name,hashlib.sha256((root/name).read_bytes()).hexdigest()))
    return {**base,'v2_sha256':hashlib.sha256(json.dumps(records).encode()).hexdigest(),
            'v2_file_count':len(records)}


def simulate(case, nominal, aware):
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
    estimator=(FaultAwareEstimator(case.position_bound+case.position_noise_bound) if aware else
               v2.CoastBiasEstimator(case.position_bound+case.position_noise_bound,True))
    events=[]
    samples=[]
    impulse_done=False
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
        if days>=case.fault_days and not impulse_done:
            if case.fault_kind in ('physical','combined'):
                state[2:4]+=sensor.fault_vector
            impulse_done=True
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
        coast=model.propagate(state,interval*DAY,
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
    if acquisition:
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
    return {'case':asdict(case),'aware':aware,'initial_state':initial.tolist(),
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
            'samples':samples,'final_state':state.tolist()}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment',choices=('discovery','validation','stress'),required=True)
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    output=root/f'development_results/fault_aware_v3_2026-10-23_{args.experiment}'
    if output.exists():
        raise RuntimeError('Refusing to overwrite results')
    seeds=range(401,405) if args.experiment=='discovery' else range(501,513)
    cases=[FaultCase(seed=seed,velocity_bound=50,navigation_velocity_bound=15,position_bound=5e5,
                     thrust_scale=.7,observation_days=30,velocity_bias_drift_per_day=.02,
                     position_noise_bound=2e5,fault_kind=kind,fault_days=660,fault_amplitude=80)
           for seed in seeds for kind in ('sensor','physical')]
    if args.experiment=='stress':
        cases=[FaultCase(seed=seed,velocity_bound=100,navigation_velocity_bound=15,position_bound=5e5,
                         thrust_scale=.5,observation_days=30,velocity_bias_drift_per_day=.2,
                         position_noise_bound=2e5,fault_kind=kind,fault_days=720,fault_amplitude=150)
               for seed in range(601,609) for kind in ('sensor','physical','combined')]
    before=historical_snapshot(root)
    output.mkdir(parents=True)
    config={'objective':'improve late-fault capture and avoid unconfirmed physical interventions',
            'cases':[asdict(c) for c in cases],'reserve_kg':80,'planning_reserve_kg':110,
            'capture_e_limit':.001,'radius_limit':.0025,'deadline_days':1100,
            'confirmation_interval':'next cruise observation or 7 days after target acquisition',
            'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (output/'configuration.json').write_text(json.dumps(config,indent=2))
    (output/'implementation_snapshot.py').write_bytes(Path(__file__).read_bytes())
    nominal=v1.plan_transfer(model.S0,0,110)
    rows=[]
    for case in cases:
        pair=[]
        for aware in (False,True):
            result=simulate(case,nominal,aware)
            pair.append(result)
            rows.append(result)
            with (output/'runs.jsonl').open('a') as stream:
                stream.write(json.dumps(result,allow_nan=False)+'\n')
            print(case.seed,case.fault_kind,aware,'capture',result['capture'],
                  'e',round(result['terminal_eccentricity'],6),'dv',round(result['correction_dv'],2),
                  'diagnoses',[r['classification'] for r in result['fault_records']],flush=True)
        assert pair[0]['departure_state']==pair[1]['departure_state']
    after=historical_snapshot(root)
    if before!=after:
        raise RuntimeError('Historical evidence changed')
    summary={'pairs':len(cases),'runs':len(rows),'protected_before':before,'protected_after':after,
             'groups':[]}
    for aware in (False,True):
        subset=[r for r in rows if r['aware']==aware]
        summary['groups'].append({'aware':aware,'n':len(subset),
                'captures':sum(r['capture'] for r in subset),'survival':sum(r['survival'] for r in subset),
                'reserve_violations':sum(r['reserve_violation'] for r in subset),
                'correction_traces':sum(r['correction_dv']>0 for r in subset),
                'ambiguous_burns':sum(r['burns_during_ambiguity'] for r in subset)})
    (output/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    main()
