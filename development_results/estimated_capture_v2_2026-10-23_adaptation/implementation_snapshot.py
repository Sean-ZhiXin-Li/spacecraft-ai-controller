"""Position-aided velocity-bias estimation for fuel-limited capture research."""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy.optimize import root

from scripts import closed_loop_transfer_dev_v1 as v1

model = v1.model
DAY = v1.DAY


@dataclass(frozen=True)
class ExperimentCase:
    seed: int
    velocity_bound: float = 100.
    navigation_velocity_bound: float = 5.
    position_bound: float = 5e5
    thrust_scale: float = .8
    observation_days: float = 30.
    velocity_bias_drift_per_day: float = 0.
    position_noise_bound: float = 0.


class Sensor:
    """Plant-facing measurement model; no bias value is given to the estimator."""
    def __init__(self, case):
        rng = np.random.default_rng(case.seed+10000)
        self.case = case
        self.bias = np.r_[rng.uniform(-1,1,2)*case.position_bound,
                          rng.uniform(-1,1,2)*case.navigation_velocity_bound,0.]
        self.drift_direction = np.array([1.,-1.])

    def measure(self, state, days):
        result = np.asarray(state).copy()+self.bias
        result[2:4] += self.case.velocity_bias_drift_per_day*days*self.drift_direction
        # Bounded reproducible observation error, shared across policies at the same epoch.
        result[:2] += self.case.position_noise_bound*np.array([
            np.sin(days*.137+self.case.seed),np.cos(days*.193+self.case.seed)])
        return result


class CoastBiasEstimator:
    """Infer velocity bias from position evolution across a known unpowered interval.

    Only measurement vectors, timestamps and the declared model enter this object.
    A constant position translation largely cancels from displacement, while velocity
    bias accumulates into a measurable position-prediction residual.
    """
    def __init__(self, position_bound, track_drift=False):
        self.position_bound = position_bound
        self.previous = None
        self.bias_estimate = np.zeros(2)
        self.bound_mps = float('inf')
        self.records = []
        self.track_drift=track_drift
        self.bias_history=[]

    def update(self, raw, days):
        raw = np.asarray(raw).copy()
        if self.previous is not None:
            previous_days, previous = self.previous
            dt=(days-previous_days)*DAY
            if dt >= DAY:
                def residual(bias):
                    initial=previous.copy()
                    initial[2:4] -= bias
                    final=model.propagate(initial,dt).y[:,-1]
                    return (final[:2]-raw[:2])/dt
                fit=root(residual,self.bias_estimate,method='hybr',options={'eps':1e-6})
                residual_norm=float(np.linalg.norm(residual(fit.x)))
                accepted=bool(np.isfinite(fit.x).all() and residual_norm<.01)
                if accepted:
                    self.bias_estimate=np.array(fit.x)
                    self.bias_history.append((previous_days,np.array(fit.x),dt/DAY))
                    if self.track_drift and len(self.bias_history)>=3:
                        # Fit time-stamped bias estimates, weighting by position-baseline
                        # duration so short noisy terminal intervals cannot dominate.
                        history=self.bias_history[-6:]
                        epochs=np.array([p[0] for p in history])
                        weights=np.array([p[2] for p in history])
                        design=np.column_stack((np.ones(len(history)),epochs-days))
                        values=np.array([p[1] for p in history])
                        coefficients=np.linalg.lstsq(design*weights[:,None],
                                                     values*weights[:,None],rcond=None)[0]
                        self.bias_estimate=coefficients[0]
                    # Conservative displacement bound; excludes unmodeled bias drift.
                    self.bound_mps=2*np.sqrt(2)*self.position_bound/dt+.05
                self.records.append({'days':days,'interval_days':dt/DAY,
                                     'bias_estimate':self.bias_estimate.tolist(),
                                     'displacement_residual_mps':residual_norm,
                                     'accepted':accepted,'declared_bound_mps':self.bound_mps
                                     if np.isfinite(self.bound_mps) else None})
        self.previous=(days,raw)

    def corrected(self, raw):
        result=np.asarray(raw).copy()
        result[2:4]-=self.bias_estimate
        return result


class GuidanceMeasurements:
    """Finite-burn measurement adapter. Bias estimate is held during the powered arc."""
    def __init__(self,sensor,estimator,days):
        self.sensor,self.estimator,self.days=sensor,estimator,days

    def observe(self,state):
        raw=self.sensor.measure(state,self.days)
        return self.estimator.corrected(raw) if self.estimator is not None else raw


def simulate(case,nominal,estimated,track_drift=False):
    rng=np.random.default_rng(case.seed)
    state=np.array(model.S0,dtype=float)
    state[2:4]+=rng.uniform(-case.velocity_bound,case.velocity_bound,2)
    initial=state.copy()
    disturbance=rng.uniform(-case.velocity_bound,case.velocity_bound,2)
    sensor=Sensor(case)
    # Both arms intentionally use the exact same raw-sensor departure controller.
    departure_navigation=GuidanceMeasurements(sensor,None,0.)
    state,dt,departure_ok,min_mass=v1.burn(state,departure_navigation,80.,case.thrust_scale,
                                          v1.velocity_goal(np.array(model.S0),nominal))
    days=dt/DAY
    departure_state=state.copy()
    estimator=CoastBiasEstimator(case.position_bound+case.position_noise_bound,track_drift) if estimated else None
    applied=False
    crossing=False
    samples=[]
    forecasts=[]
    true_crossed=False
    for index in range(80):
        if days>=120 and not applied:
            state[2:4]+=disturbance
            applied=True
        raw=sensor.measure(state,days)
        if estimator is not None:
            estimator.update(raw,days)
            observation=estimator.corrected(raw)
        else:
            observation=raw
        forecast=v1.predict(observation,max_days=max(.001,1100-days))
        forecasts.append({'days':days,'estimated_state':observation.tolist(),
                          'predicted_capture_dv':forecast.capture_dv if forecast else None,
                          'predicted_final_mass':forecast.predicted_final_mass if forecast else None})
        samples.append({'days':days,'true_state':state.tolist(),'raw_observation':raw.tolist(),
                        'estimated_state':observation.tolist()})
        def observed_crossing(t,z):
            return np.linalg.norm(sensor.measure(z,days+t/DAY)[:2])-model.TARGET_R
        observed_crossing.direction=1
        observed_crossing.terminal=True
        def truth_crossing(t,z):
            return np.linalg.norm(z[:2])-model.TARGET_R
        truth_crossing.direction=0
        truth_crossing.terminal=False
        duration=min(case.observation_days,1100-days)
        if duration<=0:
            break
        coast=model.propagate(state,duration*DAY,events=(observed_crossing,truth_crossing))
        true_crossed|=bool(len(coast.t_events[1]))
        state=coast.y[:,-1]
        days+=coast.t[-1]/DAY
        if len(coast.t_events[0]):
            crossing=True
            break
    mass_before_capture=float(state[4])
    if crossing:
        if estimator is not None:
            estimator.update(sensor.measure(state,days),days)
        nav=GuidanceMeasurements(sensor,estimator,days)
        state,dt,converged,floor=v1.burn(state,nav,80.,case.thrust_scale)
        min_mass=min(min_mass,floor)
        days+=dt/DAY
    else:
        converged=False
    radius,vr,vt=model.components(state)
    ecc=model.eccentricity(state)
    captured=bool(converged and ecc<.001 and abs(radius/model.TARGET_R-1)<=.0025)
    hold=model.propagate(state,100*DAY,t_eval=np.linspace(0,100*DAY,101),events=truth_crossing)
    true_crossed|=bool(len(hold.t_events[0]))
    max_radius_error=max(abs(np.linalg.norm(s[:2])/model.TARGET_R-1) for s in hold.y.T)
    # Bias truth is used only for post-run scoring, after guidance has completed.
    actual_bias=sensor.bias[2:4]+case.velocity_bias_drift_per_day*days*sensor.drift_direction
    bias_error=float(np.linalg.norm(estimator.bias_estimate-actual_bias)) if estimated else None
    return {'case':asdict(case),'estimated_navigation':estimated,
            'policy':'trend' if track_drift else ('static' if estimated else 'raw'),
            'initial_state':initial.tolist(),
            'departure_state':departure_state.tolist(),'delivered_error':disturbance.tolist(),
            'sensor_bias_for_evaluation_only':sensor.bias.tolist(),'observed_crossing':crossing,
            'geometric_crossing':true_crossed,'capture':captured,'guidance_converged':converged,
            'terminal_radius_error_ratio':abs(radius/model.TARGET_R-1),'terminal_radial_velocity':vr,
            'terminal_tangential_error':vt-np.sqrt(model.MU/radius),'terminal_eccentricity':ecc,
            'remaining_propellant_kg':float(state[4]-model.DRY_MASS),
            'minimum_reserve_kg':min_mass-model.DRY_MASS,'reserve_violation':bool(min_mass<380-1e-7),
            'completion_days':days,'capture_propellant_used':mass_before_capture-state[4],
            'survived_100days_sampled':bool(captured and max_radius_error<=.0025),
            'hold_max_radius_error_ratio':max_radius_error,'bias_error_mps':bias_error,
            'estimator_records':estimator.records if estimator else [],
            'forecasts':forecasts,'state_samples':samples,'final_state':state.tolist()}


def research_snapshot(root):
    records=[]
    # Includes untracked historical v1 development artifacts, not just frozen Git evidence.
    for path in sorted((root/'development_results').glob('closed_loop_transfer_v1_*')):
        for file in sorted(path.rglob('*')):
            if file.is_file():
                records.append([file.relative_to(root).as_posix(),hashlib.sha256(file.read_bytes()).hexdigest()])
    for name in ('scripts/closed_loop_transfer_dev_v1.py','scripts/report_closed_loop_transfer_dev_v1.py'):
        records.append([name,hashlib.sha256((root/name).read_bytes()).hexdigest()])
    return {'frozen':v1.protected_snapshot(root),
            'v1_development_sha256':hashlib.sha256(json.dumps(records,sort_keys=True).encode()).hexdigest(),
            'v1_file_count':len(records)}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment',choices=('discovery','validation','stress','adaptation','holdout'),required=True)
    args=parser.parse_args()
    root_path=Path(__file__).resolve().parents[1]
    output=root_path/f'development_results/estimated_capture_v2_2026-10-23_{args.experiment}'
    if output.exists():
        raise RuntimeError('Refusing to overwrite research results')
    if args.experiment=='discovery':
        cases=[ExperimentCase(10,100,5,1e7,.8,120)]
        cases += [ExperimentCase(seed,100,bias) for seed in range(51,55) for bias in (5.,15.)]
        cases += [ExperimentCase(seed,250,.5,5e5,.6,30) for seed in (43,44)]
    elif args.experiment=='validation':
        cases=[ExperimentCase(seed,100,bias,5e5,.7,30,0.,2e5)
               for seed in range(101,113) for bias in (5.,15.)]
    elif args.experiment=='stress':
        cases=[ExperimentCase(seed,250,15,5e5,.5,30,drift,2e5)
               for seed in range(201,209) for drift in (0.,.02,.2)]
    elif args.experiment=='adaptation':
        cases=[ExperimentCase(seed,250,15,5e5,.5,30,.2,2e5) for seed in range(201,209)]
    else:
        cases=[ExperimentCase(seed,150,15,5e5,.6,30,drift,2e5)
               for seed in range(301,313) for drift in (.2,.4)]
    before=research_snapshot(root_path)
    output.mkdir(parents=True)
    config={'objective':'improve capture under velocity bias at unchanged fuel floor and terminal criteria',
            'cases':[asdict(c) for c in cases],'planning_reserve_kg':110,'hard_reserve_kg':80,
            'deadline_days':1100,'hold_days':100,'independent_terminal_capture_threshold_e':.001,
            'terminal_radius_error_limit':.0025,'estimator':'two-body position shooting across unpowered arcs',
            'arms':['static','trend'] if args.experiment in ('adaptation','holdout') else ['raw','static'],
            'source_hash':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (output/'configuration.json').write_text(json.dumps(config,indent=2))
    (output/'implementation_snapshot.py').write_bytes(Path(__file__).read_bytes())
    nominal=v1.plan_transfer(model.S0,0,110)
    if nominal is None:
        raise RuntimeError('Nominal plan unavailable')
    started=time.monotonic()
    rows=[]
    for case in cases:
        pair=[]
        for policy in config['arms']:
            estimated=policy!='raw'
            result=simulate(case,nominal,estimated,policy=='trend')
            rows.append(result)
            pair.append(result)
            with (output/'runs.jsonl').open('a') as stream:
                stream.write(json.dumps(result,allow_nan=False)+'\n')
            print(case.seed,case.navigation_velocity_bound,case.velocity_bias_drift_per_day,policy,
                  'capture',result['capture'],'e',round(result['terminal_eccentricity'],7),
                  'fuel',round(result['remaining_propellant_kg'],3),'bias_error',result['bias_error_mps'],flush=True)
        assert pair[0]['initial_state']==pair[1]['initial_state']
        assert pair[0]['departure_state']==pair[1]['departure_state']
        assert pair[0]['delivered_error']==pair[1]['delivered_error']
    after=research_snapshot(root_path)
    if before!=after:
        raise RuntimeError('Historical evidence changed')
    summary={'pairs':len(cases),'runs':len(rows),'elapsed_seconds':time.monotonic()-started,
             'protected_before':before,'protected_after':after,'groups':[]}
    for policy in config['arms']:
        subset=[r for r in rows if r['policy']==policy]
        summary['groups'].append({'policy':policy,'n':len(subset),
                 'capture_count':sum(r['capture'] for r in subset),
                 'survival_count':sum(r['survived_100days_sampled'] for r in subset),
                 'reserve_violations':sum(r['reserve_violation'] for r in subset),
                 'worst_eccentricity':max(r['terminal_eccentricity'] for r in subset)})
    (output/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    main()
