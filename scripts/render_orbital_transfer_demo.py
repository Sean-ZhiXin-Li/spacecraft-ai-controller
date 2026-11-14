"""Replay one recorded finite-thrust mission; render real DOP853 states only.

Run: python -m scripts.render_orbital_transfer_demo
Check existing outputs without integration/rendering: add --verify-only.
"""
from pathlib import Path
from dataclasses import asdict
from unittest.mock import patch
import argparse
import csv
import hashlib
import json
import subprocess
import sys

import numpy as np
from scripts import multiregime_transfer_research as m

ROOT = Path(__file__).resolve().parents[1]
SOURCE_COMMIT = '1ae4745b7d02deac7ff17c3d34b923d6d3c55c0f'
DATA = ROOT / 'development_results/multiregime_transfer_2026-10-23_holdout'
OUTPUT = ROOT / 'docs/assets/orbital_transfer_1201'
PHASES = ('departure', 'coast', 'capture', 'survival')
COLORS = ('#0072B2', '#009E73', '#D55E00', '#CC79A7')
FRAMES = (15, 75, 10, 20)


def sha(data):
    return hashlib.sha256(data).hexdigest()


class Recorder:
    """Temporarily observe the original propagator; preserve calls and endpoints.

    Dense output evaluates the integrator's own polynomial, not a curve fitted
    between sparse published states. Controls, tolerances and events stay intact.
    """
    def __init__(self):
        self.segments = []
        self.seconds = 0.

    def __enter__(self):
        self.original = m.model.propagate
        self.patcher = patch.object(m.model, 'propagate', self.propagate)
        self.patcher.start()
        return self

    def __exit__(self, *args):
        self.patcher.stop()

    def propagate(self, state, seconds, *args, **kwargs):
        phase = ('survival' if 't_eval' in kwargs else 'coast') if not args else (
            'capture' if any(s['phase'] == 'coast' for s in self.segments) else 'departure')
        result = self.original(state, seconds, *args, **{**kwargs, 'dense_output': True})
        self.segments.append(dict(start=self.seconds, end=self.seconds+float(result.t[-1]),
                                  phase=phase, solution=result))
        self.seconds += float(result.t[-1])
        return result

    def state_at(self, seconds):
        if not self.segments or not 0 <= seconds <= self.seconds:
            raise ValueError('Frame time outside integrated mission')
        index = min(np.searchsorted([s['end'] for s in self.segments], seconds), len(self.segments)-1)
        segment = self.segments[index]
        return segment['solution'].sol(seconds-segment['start'])


def validate_result(actual, reference):
    """Tight reproducibility checks; these are not physical error bounds."""
    deltas = {}
    for name in ('final_state', 'crossing_state'):
        difference = np.abs(np.array(actual[name])-reference[name])
        assert np.all(difference <= [1., 1., 1e-5, 1e-5, 1e-6]), (name, difference)
        deltas[name] = difference.tolist()
    tolerances = dict(completion_days=1e-6, propellant_kg=1e-6, minimum_propellant_kg=1e-6,
                      eccentricity=1e-9, radius_error=1e-10, survival_max_radius_error=1e-10)
    for name, tolerance in tolerances.items():
        deltas[name] = abs(actual[name]-reference[name])
        assert deltas[name] <= tolerance, (name, deltas[name])
    for name in ('status', 'capture', 'survival', 'reserve_violation', 'physical_execution',
                 'departure_converged', 'crossing'):
        assert actual[name] == reference[name], name
    return dict(absolute_differences=deltas, scalar_tolerances=tolerances,
                state_tolerances=['1 m', '1 m', '1e-5 m/s', '1e-5 m/s', '1e-6 kg'],
                rationale='Roundoff-scale replay agreement; time allowance 0.0864 s is below the 100 s guidance interval. Not integration/physical uncertainty bounds.')


def inputs():
    # Git may normalize historical Windows JSON bytes. Accept only an original
    # manifest digest reproduced by exact LF/CRLF conversion, and disclose which.
    checks = {}
    for name in ('artifact_hashes.json', 'report_hashes.json'):
        for filename, expected in json.loads((DATA/name).read_text()).items():
            raw = (DATA/filename).read_bytes()
            lf = raw.replace(b'\r\n', b'\n')
            forms = {'exact': raw, 'LF': lf, 'CRLF': lf.replace(b'\n', b'\r\n')}
            matches = [kind for kind, data in forms.items() if sha(data) == expected]
            assert matches, f'Original evidence hash mismatch: {filename}'
            checks[filename] = dict(recorded_sha256=expected, checkout_sha256=sha(raw),
                                    matching_byte_form=matches[0])
    rows = [json.loads(line) for line in (DATA/'runs.jsonl').read_text().splitlines()]
    assert len(rows) == 72
    selected = {r['method']: r for r in rows if r['case']['seed'] == 1201}
    outward, bidir = selected['matched_outward'], selected['bidirectional']
    config = json.loads((DATA/'configuration.json').read_text())
    assert outward['case'] == bidir['case'] == next(c for c in config['cases'] if c['seed'] == 1201)
    assert outward['plan'] is None and outward['result']['status'] == 'no_plan'
    assert not outward['result']['physical_execution']
    assert (config['planning_reserve'], config['operational_reserve'], config['deadline_days']) == (110, 80, 4400)
    totals = {method: sum(r['result']['capture'] and r['result']['survival'] for r in rows if r['method'] == method)
              for method in ('matched_outward', 'bidirectional')}
    assert totals == dict(matched_outward=11, bidirectional=24)
    paths = [DATA/n for n in ('configuration.json', 'runs.jsonl', 'summary.json', 'comparison.json', 'artifact_hashes.json', 'report_hashes.json', 'implementation_snapshot.py')]
    paths += [ROOT/'scripts'/n for n in ('multiregime_transfer_research.py', 'closed_loop_transfer_dev_v1.py', 'dev_fuel_aware_transfer_optimizer_v0.py')]
    identities = {}
    for path in paths:
        relative = path.relative_to(ROOT).as_posix()
        committed = subprocess.check_output(['git', 'show', SOURCE_COMMIT+':'+relative], cwd=ROOT)
        assert path.read_bytes().replace(b'\r\n', b'\n') == committed.replace(b'\r\n', b'\n'), relative
        identities[relative] = dict(checkout_sha256=sha(path.read_bytes()), source_commit_blob_sha256=sha(committed))
    return bidir, config, dict(original_manifests=checks, input_identities=identities, holdout_capture_and_survival=totals)


def replay():
    record, config, integrity = inputs()
    case = record['case']
    initial = np.array(case['state'])
    nav = m.legacy.Navigation(m.legacy.Scenario('multiregime', 1201, 0, .2, 1e5))
    # One cheap matched-planner check; no new optimization campaign.
    assert m.search(nav.observe(initial), general=False, reserve=110, deadline=4400) is None
    with Recorder() as recorder:
        actual = m.execute(initial, m.legacy.Plan(**record['plan']), 1201, case['thrust_scale'], 4400)
    validation = validate_result(actual, record['result'])
    phase_ranges = {p: [min(s['start'] for s in recorder.segments if s['phase'] == p),
                        max(s['end'] for s in recorder.segments if s['phase'] == p)] for p in PHASES}
    frame_times = np.concatenate([np.linspace(*phase_ranges[p], n, endpoint=p == 'survival') for p, n in zip(PHASES, FRAMES)])
    times = list(frame_times)
    for s in recorder.segments:
        step = 10. if s['phase'] in ('departure', 'capture') else (m.DAY if s['phase'] == 'survival' else 2*m.DAY)
        times.extend(s['start']+s['solution'].t)
        times.extend(np.linspace(s['start'], s['end'], max(2, int(np.ceil((s['end']-s['start'])/step))+1)))
    times = np.unique(times)
    states = np.array([recorder.state_at(t) for t in times])
    assert np.all(np.diff(times) > 0)
    np.testing.assert_allclose(states[0], initial, rtol=0, atol=0)
    capture_index = np.searchsorted(times, phase_ranges['capture'][1])
    np.testing.assert_allclose(states[capture_index], record['result']['final_state'], rtol=0, atol=1.)
    phases = np.array([PHASES[min(np.searchsorted([phase_ranges[p][1] for p in PHASES], t), 3)] for t in times])
    frame_indices = np.searchsorted(times, frame_times)
    return times, states, phases, frame_indices, dict(source_commit=SOURCE_COMMIT, seed=1201,
        method='bidirectional', reference_method='matched_outward', configuration=config,
        recorded_plan=record['plan'], recorded_outcome=record['result'], replay_outcome=actual,
        validation=validation, integrity=integrity, phase_ranges_seconds=phase_ranges)


def render(times, states, phases, frames, manifest):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Rectangle
    from matplotlib.animation import FuncAnimation, PillowWriter
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11, 'axes.spines.top': False,
                         'axes.spines.right': False, 'axes.labelcolor': '#24364b', 'text.color': '#24364b'})
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), gridspec_kw={'height_ratios': [1.3, .65]})
    fig.subplots_adjust(top=.82, bottom=.14, left=.09, right=.97, hspace=.47, wspace=.3)
    fig.suptitle('An inward transfer that outward-only search excludes', y=.98, fontsize=19, weight='bold')
    fig.text(.5, .937, 'Seed 1201  |  simplified 2D solar two-body model  |  explicit planning + finite-thrust execution', ha='center', fontsize=11)
    fig.text(.5, .897, 'Matched outward: NO PLAN / no trajectory    •    Bidirectional: capture + sampled survival', ha='center', fontsize=12, weight='bold')
    clock = fig.text(.5, .855, '', ha='center', fontsize=12, family='monospace')
    full, zoom, radial, fuel = axes.flat
    xy = states[:, :2]/1e12
    pad = .12
    lo, hi = xy.min(axis=0)-pad, xy.max(axis=0)+pad
    side = max(hi-lo)
    center = (lo+hi)/2
    lo, hi = center-side/2, center+side/2
    orbit_lines, dots = [], []
    for ax in (full, zoom):
        ax.add_patch(Circle((0, 0), m.model.TARGET_R/1e12, fill=False, ls='--', lw=1.2, color='#8794a3'))
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlabel('Inertial x (10¹² m)'); ax.set_ylabel('Inertial y (10¹² m)')
        ax.grid(alpha=.16)
        orbit_lines.append([ax.plot([], [], color=c, lw=2.6, label=p.capitalize())[0] for p, c in zip(PHASES, COLORS)])
        dots.append(ax.plot([], [], 'o', color='#172c48', ms=6, zorder=8)[0])
        ax.plot(*xy[0], marker='^', color='#172c48', ms=6, zorder=6)
    full.set(xlim=(-8.4, 8.4), ylim=(-8.4, 8.4), title='Global frame · target radius 7.5 × 10¹² m')
    full.plot(0, 0, '*', color='#b47800', ms=11)
    full.annotate('Sun (point symbol)', (0, 0), xytext=(-7, 8), textcoords='offset points', fontsize=9)
    full.add_patch(Rectangle(lo, side, side, fill=False, color='#172c48', lw=1.2))
    zoom.set(xlim=(lo[0], hi[0]), ylim=(lo[1], hi[1]), title='Transfer detail · fixed scale throughout playback')
    zoom.legend(loc='upper left', fontsize=9, framealpha=.95)
    completion = manifest['recorded_outcome']['completion_days']
    days = times/m.DAY
    radial.plot(days, np.linalg.norm(states[:, :2], axis=1)/m.model.TARGET_R, color=COLORS[1], lw=1.8)
    radial.axhline(1, ls='--', color='#8794a3', lw=1)
    radial.set(ylabel='True radius / target radius', xlabel='Elapsed simulation days', title='Full replay diagnostics · linear time')
    fuel.plot(days, states[:, 4]-m.model.DRY_MASS, color=COLORS[0], lw=1.8)
    fuel.axhline(110, color=COLORS[2], ls=':', label='Planning reserve: 110 kg')
    fuel.axhline(80, color='#444444', ls='--', label='Operational reserve: 80 kg')
    fuel.set(ylim=(0, 450), ylabel='Propellant remaining (kg)', xlabel='Elapsed simulation days', title='Finite burns consume propellant')
    fuel.legend(loc='upper right', fontsize=9)
    cursors = []
    for ax in (radial, fuel):
        ax.axvline(completion, color=COLORS[2], lw=1, ls=':')
        ax.axvspan(completion, days[-1], color=COLORS[3], alpha=.12)
        ax.grid(alpha=.16)
        cursors.append(ax.axvline(0, color='#172c48', lw=1))
    fig.text(.5, .075, 'Shared contract: 4,400-day deadline · 110 kg planning / 80 kg operational reserve · target orbit dashed', ha='center', fontsize=10)
    fig.text(.5, .048, 'Playback: 12 s at 10 fps; phase timing is nonuniform. Burns expanded; the on-screen clock is actual simulation time.', ha='center', fontsize=10)
    fig.text(.5, .022, 'DOP853 solver states; fixed spatial scales. Diagnostics show the full replay. Sampled survival ≠ continuous safety; no_plan ≠ impossibility.', ha='center', fontsize=9)

    def update(index):
        for lines, dot in zip(orbit_lines, dots):
            for p, line in zip(PHASES, lines):
                mask = (phases == p) & (np.arange(len(times)) <= index)
                line.set_data(xy[mask, 0], xy[mask, 1])
            dot.set_data([xy[index, 0]], [xy[index, 1]])
        for cursor in cursors:
            cursor.set_xdata([days[index], days[index]])
        # Frame phase label is determined by actual phase start, including boundaries.
        phase = next((p for p in reversed(PHASES) if times[index] >= manifest['phase_ranges_seconds'][p][0]), 'departure')
        clock.set_text(f'{phase.upper():9s}  t = {days[index]:10.5f} days   propellant = {states[index,4]-300:7.3f} kg')
    update(len(times)-1)
    fig.savefig(OUTPUT/'transfer.png', dpi=160)
    animation = FuncAnimation(fig, update, frames=frames, interval=100, blit=False)
    animation.save(OUTPUT/'transfer.gif', writer=PillowWriter(fps=10), dpi=90)
    plt.close(fig)


def verify_existing():
    manifest = json.loads((OUTPUT/'provenance.json').read_text())
    inputs()
    assert sha(Path(__file__).read_bytes()) == manifest['render_script_sha256']
    for name, digest in manifest['output_sha256'].items():
        assert sha((OUTPUT/name).read_bytes()) == digest, name
    validate_result(manifest['replay_outcome'], manifest['recorded_outcome'])
    print('Verified source identities, original manifests, replay metrics and output hashes.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    if args.verify_only:
        verify_existing(); return
    times, states, phases, frames, manifest = replay()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT/'trajectory.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream, lineterminator='\n')
        writer.writerow(['seconds', 'x_m', 'y_m', 'vx_mps', 'vy_mps', 'mass_kg', 'phase', 'animation_frame'])
        frame_map = {int(index): frame for frame, index in enumerate(frames)}
        writer.writerows([float(t), *state.tolist(), phase, frame_map.get(index, -1)]
                         for index, (t, state, phase) in enumerate(zip(times, states, phases)))
    manifest.update(render_script='scripts/render_orbital_transfer_demo.py',
        render_script_sha256=sha(Path(__file__).read_bytes()),
        command='python -m scripts.render_orbital_transfer_demo',
        versions=dict(python=sys.version.split()[0], numpy=np.__version__, scipy=__import__('scipy').__version__,
                      matplotlib=__import__('matplotlib').__version__, pillow=__import__('PIL').__version__),
        model=dict(state='[x_m,y_m,vx_mps,vy_mps,mass_kg]', frame='inertial Cartesian XY, Sun at origin',
                   integrator='SciPy solve_ivp DOP853', rtol=1e-11, atol=[.01,.01,1e-7,1e-7,1e-7],
                   thrust_newtons=2000, thrust_scale=record_scale(manifest), guidance_seconds=100,
                   dense_output='Solver-provided polynomial on each actual integration interval; not endpoint interpolation or ideal Kepler substitution.'),
        sampling=dict(count=len(times), strictly_increasing=True, trace='Union of native solver nodes and grids <=10 s in burns, <=2 days in coast, <=1 day in survival, plus animation timestamps.',
                      frames=120, fps=10, playback_seconds=12, phase_frame_counts=dict(zip(PHASES, FRAMES)),
                      time_mapping='Nonuniform phase playback; 1.5 s departure, 7.5 s coast, 1 s capture, 2 s survival. Actual simulation days displayed.'),
        limitations=['One selected successful seed, not an additional benchmark.',
                     'Matched outward has no plan and executes no trajectory.',
                     'Capture: guidance convergence, e<0.001, relative radius error<=0.0025, deadline<=4400 days.',
                     'Survival: 101 samples across 100 unpowered days; not continuous safety.',
                     'Simplified planar solar two-body model; explicit control, not RL or flight-ready GNC.'])
    render(times, states, phases, frames, manifest)
    manifest['output_sha256'] = {n: sha((OUTPUT/n).read_bytes()) for n in ('trajectory.csv','transfer.png','transfer.gif')}
    (OUTPUT/'provenance.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8', newline='\n')
    verify_existing()
    print(json.dumps(dict(samples=len(times), validation=manifest['validation'], phase_ranges_seconds=manifest['phase_ranges_seconds']), indent=2))


def record_scale(manifest):
    return next(c['thrust_scale'] for c in manifest['configuration']['cases'] if c['seed'] == 1201)


if __name__ == '__main__':
    main()
