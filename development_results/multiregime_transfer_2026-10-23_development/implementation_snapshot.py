"""Bidirectional elliptic transfer search with independent finite-thrust execution."""
from dataclasses import asdict
from pathlib import Path
import argparse
import hashlib
import json
import math
import time
import numpy as np
from scipy.optimize import minimize
from scripts import closed_loop_transfer_dev_v1 as legacy
from scripts import asynchronous_fault_dev_v4 as history

model, DAY = legacy.model, legacy.DAY


def elliptic_prediction(state, dvr=0., dvt=0., max_days=200000.):
    """Earliest future radius intersection, including inward arcs and turning points."""
    state=np.asarray(state,dtype=float)
    if state.shape!=(5,) or not np.isfinite(state).all() or state[4]<=0:
        return None
    r,vr,vt=model.components(state)
    vr,vt=vr+dvr,vt+dvt
    if r<=0 or vt<=0:
        return None
    energy=(vr*vr+vt*vt)/2-model.MU/r
    if energy>=0:
        return None
    a=-model.MU/(2*energy)
    h=r*vt
    e2=1-h*h/(model.MU*a)
    if e2<=1e-14 or e2>=1:
        return None
    e=math.sqrt(e2)
    target_cos=(1-model.TARGET_R/a)/e
    if abs(target_cos)>=1:
        return None
    initial_E=math.atan2(r*vr/math.sqrt(model.MU*a),(1-r/a))
    initial_M=initial_E-e*math.sin(initial_E)
    target_E=math.acos(target_cos)
    n=math.sqrt(model.MU/a**3)
    times=[((E-e*math.sin(E)-initial_M)%(2*math.pi))/n/DAY for E in (target_E,-target_E)]
    days=min(t for t in times if t>1e-8) if any(t>1e-8 for t in times) else math.inf
    if days>max_days:
        return None
    vt_target=h/model.TARGET_R
    vr2=2*(energy+model.MU/model.TARGET_R)-vt_target**2
    if vr2<0:
        return None
    capture=math.hypot(math.sqrt(vr2),model.VC-vt_target)
    total=math.hypot(dvr,dvt)+capture
    return legacy.Plan(float(dvr),float(dvt),days,capture,total,state[4]*math.exp(-total/model.VE))


def search(state, general=True, reserve=110., deadline=1100.):
    """Same scaled search/seeds for both prediction models; no global-optimality claim."""
    state=np.asarray(state,dtype=float)
    if state[4]<=model.DRY_MASS+reserve+.5:
        return None
    r,vr,vt=model.components(state)
    budget=model.VE*math.log(state[4]/(model.DRY_MASS+reserve+.5))
    predictor=elliptic_prediction if general else legacy.predict
    def prediction(x):
        return predictor(state,*(1000*np.asarray(x)),max_days=200000.)
    def objective(x):
        p=prediction(x)
        return p.coast_days/1000 if p else 1e5
    def margins(x):
        p=prediction(x)
        return np.array([(budget-p.total_dv)/1000,(deadline-p.coast_days)/1000,
            (1.90*model.VC-math.hypot(vr+1000*x[0],vt+1000*x[1]))/1000]) if p else np.array([-1e3]*3)
    seeds=[(0.,0.),(.35,.1),(.7,.15),(-.35,.1),(-.7,.15),
           (-vr/1000+.4,(model.VC-vt)/1000),(-vr/1000-.4,(model.VC-vt)/1000)]
    candidates=[]
    for seed in seeds:
        result=minimize(objective,np.clip(seed,-2,2),method='SLSQP',bounds=[(-2,2)]*2,
            constraints=[{'type':'ineq','fun':margins}],options={'maxiter':120,'ftol':1e-9})
        p=prediction(result.x)
        if p and min(margins(result.x))>=-1e-8:
            candidates.append(p)
    return min(candidates,key=lambda p:p.coast_days) if candidates else None


def execute(state, plan, seed, thrust_scale):
    """Measurement-only guidance; plant state used independently for evaluation."""
    if plan is None:
        return {'status':'no_plan','capture':False,'survival':False,'physical_execution':False}
    scenario=legacy.Scenario('multiregime',seed,0,.2,1e5,thrust_scale)
    nav=legacy.Navigation(scenario)
    initial=np.array(state,dtype=float)
    state,seconds,ok,floor=legacy.burn(initial,nav,80,thrust_scale,
                                     legacy.velocity_goal(nav.observe(initial),plan))
    def crossing(t,z):
        return np.linalg.norm(nav.observe(z)[:2])-model.TARGET_R
    crossing.terminal=True
    crossing.direction=0
    coast=model.propagate(state,max(0,1100*DAY-seconds),events=crossing)
    state=coast.y[:,-1]
    crossing_found=bool(len(coast.t_events[0]))
    days=(seconds+coast.t[-1])/DAY
    crossing_state=state.copy()
    if crossing_found:
        state,dt,converged,minimum=legacy.burn(state,nav,80,thrust_scale)
        floor=min(floor,minimum)
        days+=dt/DAY
    else:
        converged=False
    radius,vr,vt=model.components(state)
    ecc=model.eccentricity(state)
    captured=bool(converged and ecc<.001 and abs(radius/model.TARGET_R-1)<=.0025 and days<=1100)
    hold=model.propagate(state,100*DAY,t_eval=np.linspace(0,100*DAY,101))
    error=max(abs(np.linalg.norm(z[:2])/model.TARGET_R-1) for z in hold.y.T)
    return {'status':'capture' if captured else ('terminal_failure' if crossing_found else 'no_crossing'),
        'capture':captured,'survival':bool(captured and error<=.0025),'physical_execution':True,
        'departure_converged':ok,'crossing':crossing_found,'crossing_state':crossing_state.tolist(),
        'eccentricity':ecc,'radius_error':abs(radius/model.TARGET_R-1),
        'completion_days':days,'propellant_kg':state[4]-300,'minimum_propellant_kg':floor-300,
        'reserve_violation':bool(floor<380-1e-7),'survival_max_radius_error':error,
        'final_state':state.tolist()}


def tasks(split):
    seeds=range(1101,1109) if split=='development' else range(1201,1225)
    result=[]
    for seed in seeds:
        rng=np.random.default_rng(seed)
        outside=bool(seed%2)
        ratio=float(rng.uniform(1.01,1.06) if outside else rng.uniform(.94,.99))
        radius=ratio*model.TARGET_R
        vr=float(rng.uniform(-350,350))
        vt=float(math.sqrt(model.MU/radius)*rng.uniform(.85,1.08))
        angle=float(rng.uniform(-math.pi,math.pi))
        er=np.array([math.cos(angle),math.sin(angle)])
        et=np.array([-er[1],er[0]])
        state=np.r_[radius*er,vr*er+vt*et,722.]
        result.append({'seed':seed,'outside':outside,'state':state.tolist(),'radius_ratio':ratio,
                       'initial_vr':vr,'initial_vt':vt,'thrust_scale':float(rng.uniform(.5,1.))})
    return result


def snapshot(root):
    previous=history.historical_snapshot(root)
    files=list((root/'development_results').glob('asynchronous_fault_v4_*/*'))
    files+=list((root/'scripts').glob('*asynchronous_fault*'))
    hashes=[(p.relative_to(root).as_posix(),hashlib.sha256(p.read_bytes()).hexdigest())
            for p in sorted(files) if p.is_file()]
    return {'previous':previous,'asynchronous_sha256':hashlib.sha256(json.dumps(hashes).encode()).hexdigest()}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment',choices=('development','holdout'),required=True)
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    output=root/f'development_results/multiregime_transfer_2026-10-23_{args.experiment}'
    if output.exists():
        raise RuntimeError('Output already exists')
    cases=tasks(args.experiment)
    before=snapshot(root)
    output.mkdir(parents=True)
    (output/'configuration.json').write_text(json.dumps({'cases':cases,'hypothesis':
        'bidirectional Kepler intersection search expands capture beyond outward-only geometry',
        'planning_reserve':110,'operational_reserve':80,'deadline_days':1100,
        'capture_e':.001,'capture_radius_ratio':.0025,'survival_days':100},indent=2))
    (output/'implementation_snapshot.py').write_bytes(Path(__file__).read_bytes())
    rows=[]
    for case in cases:
        for method in ('legacy','matched_outward','bidirectional'):
            state=np.array(case['state'])
            nav=legacy.Navigation(legacy.Scenario('multiregime',case['seed'],0,.2,1e5))
            observed=nav.observe(state)
            start=time.perf_counter()
            plan=legacy.plan_transfer(observed,0,110) if method=='legacy' else search(observed,method=='bidirectional')
            elapsed=time.perf_counter()-start
            row={'case':case,'method':method,'plan':asdict(plan) if plan else None,'planning_seconds':elapsed,
                 'result':execute(state,plan,case['seed'],case['thrust_scale'])}
            rows.append(row)
            with (output/'runs.jsonl').open('a') as f:
                f.write(json.dumps(row,allow_nan=False)+'\n')
            print(case['seed'],method,row['result']['status'],round(elapsed,3),flush=True)
    after=snapshot(root)
    assert before==after
    summary={'tasks':len(cases),'comparisons':len(rows),'protected_before':before,'protected_after':after,'groups':[]}
    for method in ('legacy','matched_outward','bidirectional'):
        group=[r for r in rows if r['method']==method]
        summary['groups'].append({'method':method,'plans':sum(r['plan'] is not None for r in group),
            'captures':sum(r['result']['capture'] for r in group),'survival':sum(r['result']['survival'] for r in group),
            'physical_executions':sum(r['result']['physical_execution'] for r in group),
            'reserve_violations':sum(r['result'].get('reserve_violation',False) for r in group),
            'median_planning_seconds':float(np.median([r['planning_seconds'] for r in group]))})
    (output/'summary.json').write_text(json.dumps(summary,indent=2))
    (output/'artifact_hashes.json').write_text(json.dumps({p.name:hashlib.sha256(p.read_bytes()).hexdigest()
        for p in output.iterdir() if p.is_file()},indent=2))
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    main()
