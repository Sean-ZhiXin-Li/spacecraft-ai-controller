"""Isolated variable-mass, observation-driven transfer research prototype."""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import subprocess
import time

import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize

from scripts import dev_fuel_aware_transfer_optimizer_v0 as model

DAY = 86400.0


@dataclass(frozen=True)
class Scenario:
    name: str
    seed: int
    velocity_bound: float
    nav_velocity_bound: float = 0.2
    nav_position_bound: float = 1e5
    thrust_scale: float = 1.0
    observation_days: float = 60.0


@dataclass(frozen=True)
class Plan:
    dvr: float
    dvt: float
    coast_days: float
    capture_dv: float
    total_dv: float
    predicted_final_mass: float


def predict(state, dvr=0.0, dvt=0.0, max_days=1200.0):
    """Two-body outward radial quadrature from arbitrary current Cartesian state."""
    r, vr, vt = model.components(state)
    vr, vt = vr + dvr, vt + dvt
    if r >= model.TARGET_R or vr <= 1e-3 or vt <= 0:
        return None
    h = r * vt
    energy = (vr * vr + vt * vt) / 2 - model.MU / r

    def vr2(u):
        radius = model.TARGET_R * u
        return 2 * (energy + model.MU / radius) - (h / radius) ** 2

    if min(vr2(r / model.TARGET_R), vr2(1.0)) <= 0:
        return None
    try:
        days = quad(lambda u: model.TARGET_R / math.sqrt(vr2(u)),
                    r / model.TARGET_R, 1.0, epsrel=1e-9)[0] / DAY
    except (ValueError, ZeroDivisionError):
        return None
    if not math.isfinite(days) or days > max_days:
        return None
    capture = math.hypot(math.sqrt(vr2(1.0)), model.VC - h / model.TARGET_R)
    total = math.hypot(dvr, dvt) + capture
    return Plan(float(dvr), float(dvt), days, capture, total,
                float(state[4]) * math.exp(-total / model.VE))


def plan_transfer(state, elapsed_days, reserve, deadline_days=1100.0,
                  correction_limit=None):
    """Search current mass budget, including terminal capture; never reset mission time."""
    state = np.asarray(state, dtype=float)
    if state.shape != (5,) or not np.isfinite(state).all() or state[4] <= model.DRY_MASS + reserve:
        return None
    horizon = deadline_days - elapsed_days
    if horizon <= 0:
        return None
    budget = model.VE * math.log(state[4] / (model.DRY_MASS + reserve + 0.5))
    r, vr, vt = model.components(state)
    limit = 2000.0 if correction_limit is None else correction_limit

    def prediction(x):
        return predict(state, *x, max_days=horizon)

    def objective(x):
        p = prediction(x)
        if p is None:
            return 1e5
        # Initial planning minimizes time; corrective planning minimizes intervention fuel.
        return p.coast_days if correction_limit is None else math.hypot(*x) + 1e-4 * p.coast_days

    def margin(x):
        p = prediction(x)
        return -1e4 if p is None else budget - p.total_dv

    seeds = [(0., 0.), (350., 100.), (550., 250.), (700., 150.)]
    if correction_limit is not None:
        seeds = [(0., 0.), (-limit / 2, 0.), (0., limit / 2), (-limit / 2, limit / 3)]
    candidates = []
    for seed in seeds:
        result = minimize(objective, np.clip(seed, -limit, limit), method="SLSQP",
                          bounds=[(-limit, limit), (-limit, limit)],
                          constraints=[
                              {"type": "ineq", "fun": margin},
                              {"type": "ineq", "fun": lambda x: 1.90 * model.VC - math.hypot(vr + x[0], vt + x[1])},
                              {"type": "ineq", "fun": lambda x: limit - math.hypot(*x)}
                          ] if correction_limit is not None else [
                              {"type": "ineq", "fun": margin},
                              {"type": "ineq", "fun": lambda x: 1.90 * model.VC - math.hypot(vr + x[0], vt + x[1])}],
                          options={"maxiter": 150, "ftol": 1e-8})
        p = prediction(result.x)
        if p and margin(result.x) >= -1e-5 and math.hypot(vr+p.dvr, vt+p.dvt) <= 1.90*model.VC:
            if correction_limit is None or math.hypot(p.dvr, p.dvt) <= limit + 1e-6:
                candidates.append((objective(result.x), p))
    return min(candidates, key=lambda entry: entry[0])[1] if candidates else None


class Navigation:
    """Bounded deterministic measurement bias. Only this interface exposes state to guidance."""
    def __init__(self, scenario):
        rng = np.random.default_rng(scenario.seed + 10000)
        self.bias = np.r_[rng.uniform(-1, 1, 2) * scenario.nav_position_bound,
                          rng.uniform(-1, 1, 2) * scenario.nav_velocity_bound, 0.]

    def observe(self, true_state):
        # Mass is assumed known from calibrated propellant telemetry in this first prototype.
        return np.asarray(true_state).copy() + self.bias


def burn(true_state, navigation, reserve, thrust_scale, goal=None, max_steps=80):
    """Held-direction finite burns. The actual integrated mass floor limits partial burns."""
    state = np.asarray(true_state).copy()
    elapsed = 0.
    minimum_mass = float(state[4])
    for _ in range(max_steps):
        observed = navigation.observe(state)
        if goal is None:
            r = math.hypot(*observed[:2])
            target = math.sqrt(model.MU / r) * np.array([-observed[1], observed[0]]) / r
        else:
            target = goal
        error = target - observed[2:4]
        dv = float(np.linalg.norm(error))
        if dv < .001:
            return state, elapsed, True, minimum_mass
        usable = max(0., state[4] - model.DRY_MASS - reserve)
        if usable < 1e-8:
            return state, elapsed, False, minimum_mass
        # Nominal duration; unknown actual thrust changes delivered impulse and consumed mass.
        duration = min(model.GUIDANCE_DT,
                       state[4] * (1 - math.exp(-dv/model.VE)) / model.MDOT,
                       usable / (model.MDOT * thrust_scale))
        if duration < 1e-9:
            return state, elapsed, False, minimum_mass
        out = model.propagate(state, duration, *(error/dv), model.THRUST_N * thrust_scale)
        state = out.y[:, -1]
        elapsed += duration
        minimum_mass = min(minimum_mass, float(state[4]))
        if state[4] < model.DRY_MASS + reserve - 1e-7:
            raise RuntimeError("Continuous propagation breached operational reserve")
    return state, elapsed, False, minimum_mass


def velocity_goal(observed, plan):
    er = observed[:2] / np.linalg.norm(observed[:2])
    et = np.array([-er[1], er[0]])
    return observed[2:4] + plan.dvr*er + plan.dvt*et


def execute(scenario, nominal, planning_reserve, operational_reserve, feedback, revision=2):
    rng = np.random.default_rng(scenario.seed)
    state = np.array(model.S0, dtype=float)
    perturbation = rng.uniform(-scenario.velocity_bound, scenario.velocity_bound, 2)
    state[2:4] += perturbation
    initial_state = state.copy()
    nav = Navigation(scenario)
    events = []
    deadline = nominal.coast_days + 30.0 if revision >= 2 else 1100.0
    departure_plan = nominal
    if feedback and revision >= 2 and scenario.nav_velocity_bound <= 2:
        # Observe before committing scarce propellant; the same planning reserve is retained.
        departure_plan = plan_transfer(nav.observe(state), 0., planning_reserve,
                                      deadline_days=deadline)
        events.append({"elapsed_days":0., "decision":"replan_departure_from_observation"
                       if departure_plan else "infeasible_development_coast_fallback",
                       "reasons":["observed_initial_state_differs_from_nominal"],
                       "observed_state":nav.observe(state).tolist(),
                       "corrective_plan":asdict(departure_plan) if departure_plan else None})
        if departure_plan is None:
            # Failure is explicit: coast without spending departure fuel, then assess capture.
            departure_plan = Plan(0,0,0,0,0,float(state[4]))
    # Same precomputed nominal inertial target in both policies.
    goal = velocity_goal(nav.observe(state), departure_plan) if feedback and revision >= 2 else velocity_goal(np.array(model.S0), nominal)
    state, seconds, departure_ok, min_mass = burn(state, nav, operational_reserve,
                                                 scenario.thrust_scale, goal)
    elapsed = seconds / DAY
    if revision >= 2:
        # A deterministic delivered-velocity execution error, distinct from navigation bias.
        state[2:4] += rng.uniform(-scenario.velocity_bound, scenario.velocity_bound, 2)
    reference = np.array(model.S0, dtype=float)
    reference, nominal_seconds, _, _ = burn(reference, Navigation(Scenario("reference", 0, 0, 0, 0)),
                                          operational_reserve, 1., goal)
    predicted_arrival = nominal.coast_days + nominal_seconds / DAY
    correction_dv = 0.
    crossing = False
    mode = "coast"
    trajectory = []
    for observation_index in range(60):
        observation = nav.observe(state)
        forecast = predict(observation, max_days=max(.001,deadline-elapsed))
        reference_prediction = model.propagate(reference, max(0., elapsed-nominal_seconds/DAY)*DAY).y[:, -1]
        residual = float(np.linalg.norm(observation[2:4]-reference_prediction[2:4]))
        uncertainty = 2 * math.sqrt(2) * scenario.nav_velocity_bound
        if forecast:
            fuel_margin = forecast.predicted_final_mass - model.DRY_MASS - operational_reserve
            arrival_error = elapsed + forecast.coast_days - predicted_arrival
        else:
            fuel_margin, arrival_error = None, None
        reasons = []
        if forecast is None:
            reasons.append("no_predicted_crossing_before_deadline")
        elif fuel_margin < 2.0:
            reasons.append("predicted_capture_reserve_shortfall")
        if residual > uncertainty + 1.0:
            reasons.append("velocity_prediction_residual_exceeds_navigation_envelope")
        if feedback and reasons and correction_dv < 120 and len([e for e in events if e['decision'] in ('bounded_correction','replan_remaining_transfer')]) < 3:
            if scenario.nav_velocity_bound > 2.0:
                decision, correction = "preserve_fuel_navigation_uncertain", None
            else:
                correction = plan_transfer(observation, elapsed, operational_reserve+2,
                                           deadline_days=deadline,
                                           correction_limit=120-correction_dv)
                decision = "replan_remaining_transfer" if correction else "infeasible_development_coast_fallback"
                if correction and math.hypot(correction.dvr, correction.dvt) < .01:
                    decision = "continue_existing_plan"
                elif correction:
                    before_mass = state[4]
                    state, dt, ok, floor = burn(state, nav, operational_reserve,
                                               scenario.thrust_scale, velocity_goal(observation, correction))
                    elapsed += dt / DAY
                    min_mass = min(min_mass, floor)
                    correction_dv += model.VE * math.log(before_mass/state[4])
                    if not ok:
                        decision = "correction_failed_development_coast_fallback"
                    else:
                        decision = "bounded_correction"
                    # New reference is the estimated post-correction state, never hidden truth.
                    reference = nav.observe(state)
                    nominal_seconds = elapsed * DAY
                    predicted_arrival = elapsed + correction.coast_days
            events.append({"elapsed_days": elapsed, "decision": decision, "reasons": reasons,
                           "forecast": asdict(forecast) if forecast else None,
                           "observed_state": observation.tolist(), "velocity_residual": residual,
                           "predicted_reserve_margin_kg": fuel_margin,
                           "arrival_error_days": arrival_error,
                           "corrective_plan": asdict(correction) if correction else None})
        trajectory.append({"days": elapsed, "true_state": state.tolist(),
                           "observed_state": nav.observe(state).tolist()})
        def observed_crossing(t, z):
            return np.linalg.norm(nav.observe(z)[:2])-model.TARGET_R
        observed_crossing.terminal = True
        observed_crossing.direction = 1
        def true_crossing_event(t, z):
            return np.linalg.norm(z[:2])-model.TARGET_R
        true_crossing_event.terminal = False
        true_crossing_event.direction = 0
        dt_days = min(scenario.observation_days, 1100-elapsed)
        if dt_days <= 0:
            mode = "deadline_exceeded"
            break
        coast = model.propagate(state, dt_days * DAY, events=(observed_crossing,true_crossing_event))
        if len(coast.t_events[1]):
            trajectory.append({"days":elapsed+float(coast.t_events[1][0])/DAY,
                               "true_state":coast.y_events[1][0].tolist(),
                               "observed_state":nav.observe(coast.y_events[1][0]).tolist(),
                               "verified_true_crossing":True})
        state = coast.y[:, -1]
        elapsed += float(coast.t[-1])/DAY
        if len(coast.t_events[0]):
            crossing = True
            mode = "capture"
            break
    if crossing:
        state, dt, capture_ok, floor = burn(state, nav, operational_reserve, scenario.thrust_scale)
        elapsed += dt/DAY
        min_mass = min(min_mass, floor)
    else:
        capture_ok = False
    r, vr, vt = model.components(state)
    eccentricity = model.eccentricity(state)
    captured = capture_ok and eccentricity < .001 and abs(r/model.TARGET_R-1) <= .0025
    hold = model.propagate(state, 100*DAY, t_eval=np.linspace(0,100*DAY,101))
    survival_error = max(abs(np.linalg.norm(z[:2])/model.TARGET_R-1) for z in hold.y.T)
    radii = [np.linalg.norm(np.asarray(p['true_state'])[:2]) for p in trajectory] + [r]
    true_crossing = any(a < model.TARGET_R <= b for a,b in zip(radii,radii[1:])) or any(p.get('verified_true_crossing') for p in trajectory)
    return {"scenario": asdict(scenario), "feedback": feedback,
            "planning_reserve_kg": planning_reserve, "operational_reserve_kg": operational_reserve,
            "initial_state": initial_state.tolist(), "initial_velocity_error": perturbation.tolist(),
            "navigation_bias": nav.bias.tolist(), "departure_ok": departure_ok,
            "observed_crossing": crossing, "geometric_crossing": true_crossing,
            "capture": bool(captured), "terminal_radius_error_ratio": abs(r/model.TARGET_R-1),
            "terminal_radial_velocity": vr, "terminal_tangential_error": vt-math.sqrt(model.MU/r),
            "terminal_eccentricity": eccentricity, "remaining_propellant_kg": float(state[4]-model.DRY_MASS),
            "minimum_reserve_kg": min_mass-model.DRY_MASS,
            "reserve_violation": bool(min_mass < model.DRY_MASS+operational_reserve-1e-7),
            "completion_days": elapsed, "correction_dv": correction_dv,
            "deadline_days":deadline,"completed_before_deadline":bool(captured and elapsed<=deadline),
            "replanning_events": events, "terminal_mode": mode,
            "hold_100d_max_radius_error_ratio": survival_error,
            "survived_100d_sampled": bool(captured and survival_error <= .0025),
            "final_state": state.tolist(), "trajectory_samples": trajectory}


def protected_snapshot(root):
    names = subprocess.check_output(['git','-c',f'safe.directory={root.as_posix()}',
                                     'ls-files','analysis','runtime_assurance','simulator'],cwd=root).decode().splitlines()
    digest = hashlib.sha256()
    for name in sorted(names):
        digest.update(name.encode())
        digest.update(hashlib.sha256((root/name).read_bytes()).digest())
    return {"sha256": digest.hexdigest(), "file_count": len(names)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', action='store_true')
    parser.add_argument('--revision',type=int,choices=(1,2),default=2)
    args = parser.parse_args()
    if not args.experiment:
        parser.print_help()
        return
    root = Path(__file__).resolve().parents[1]
    output = root/('development_results/closed_loop_transfer_v1_2026-10-23' + ('_revision2' if args.revision==2 else ''))
    if output.exists():
        raise RuntimeError('Refusing to overwrite experiment')
    before = protected_snapshot(root)
    scenarios = [Scenario('nominal',0,0,0,0)]
    for seed in range(1,5):
        scenarios.append(Scenario(f'modest_{seed}',seed,10,.2,1e5, .9 if seed%2 else 1.,60))
    for seed in range(5,9):
        scenarios.append(Scenario(f'hard_{seed}',seed,100,1.,1e6,.75,90))
    scenarios.extend([Scenario('extreme_9',9,250,1,1e6,.6,120),
                      Scenario('uncertain_nav',10,100,5,1e7,.8,120),
                      Scenario('extreme_11',11,400,1,1e6,.5,120)])
    output.mkdir(parents=True)
    (output/'implementation_snapshot.py').write_bytes(Path(__file__).read_bytes())
    config = {"revision":args.revision,"scenarios": [asdict(s) for s in scenarios], "reserve_pairs": [[80,80],[110,80]],
              "implementation_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "deadline_contract":"nominal crossing plus 30 days" if args.revision==2 else "1100 days",
              "deadline_days":1100,"correction_limit_mps":120,"max_corrections":3,
              "capture_forecast_buffer_kg":2,"velocity_residual_margin_mps":1,
              "navigation_confidence_limit_mps":2,
              "model":"2D two-body variable mass; development-only; no frozen Final Veto authority"}
    (output/'configuration.json').write_text(json.dumps(config,indent=2))
    started = time.monotonic()
    rows=[]
    for reserve in (80.,110.):
        nominal=plan_transfer(model.S0,0,reserve)
        if nominal is None:
            raise RuntimeError('Nominal planner failed')
        print('PLAN',reserve,asdict(nominal),flush=True)
        for scenario in scenarios:
            for feedback in (False,True):
                row=execute(scenario,nominal,reserve,80.,feedback,args.revision)
                rows.append(row)
                with (output/'runs.jsonl').open('a') as stream:
                    stream.write(json.dumps(row,allow_nan=False)+'\n')
                print(reserve,scenario.name,feedback,'capture',row['capture'],'fuel',round(row['remaining_propellant_kg'],3),
                      'days',round(row['completion_days'],2),'correction',round(row['correction_dv'],2),flush=True)
    after=protected_snapshot(root)
    if before!=after:
        raise RuntimeError('Protected sources changed')
    summary={"protected_before":before,"protected_after":after,"runtime_seconds":time.monotonic()-started,
             "runs":len(rows),"paired_scenarios":len(rows)//2,"source_head":subprocess.check_output(
                 ['git','-c',f'safe.directory={root.as_posix()}','rev-parse','HEAD'],cwd=root).decode().strip(),
             "groups":[]}
    for reserve in (80.,110.):
        for feedback in (False,True):
            subset=[r for r in rows if r['planning_reserve_kg']==reserve and r['feedback']==feedback]
            summary['groups'].append({"planning_reserve_kg":reserve,"feedback":feedback,"n":len(subset),
                 "capture_count":sum(r['capture'] for r in subset),
                 "deadline_capture_count":sum(r['completed_before_deadline'] for r in subset),
                 "survival_count":sum(r['survived_100d_sampled'] for r in subset),
                 "reserve_violations":sum(r['reserve_violation'] for r in subset),
                 "worst_eccentricity":max(r['terminal_eccentricity'] for r in subset)})
    (output/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    main()
