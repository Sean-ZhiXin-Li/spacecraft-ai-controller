"""Offline comparison and figures for every recorded development iteration."""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    root=Path(__file__).resolve().parents[1]
    directories=sorted((root/'development_results').glob('closed_loop_transfer_v1_2026-10-23*'))
    comparisons=[]
    for directory in directories:
        if not (directory/'summary.json').exists():
            continue
        rows=[json.loads(line) for line in (directory/'runs.jsonl').read_text().splitlines()]
        groups={}
        for row in rows:
            groups.setdefault((row['planning_reserve_kg'],row['scenario']['name']),{})[row['feedback']]=row
        paired=[]
        for (reserve,name),arms in groups.items():
            baseline,feedback=arms[False],arms[True]
            assert baseline['initial_state']==feedback['initial_state']
            assert baseline['scenario']==feedback['scenario']
            assert baseline['navigation_bias']==feedback['navigation_bias']
            assert baseline['operational_reserve_kg']==feedback['operational_reserve_kg']==80
            paired.append({'reserve':reserve,'scenario':name,
                           'baseline_capture':baseline['capture'],'feedback_capture':feedback['capture'],
                           'time_delta_days':feedback['completion_days']-baseline['completion_days'],
                           'propellant_delta_kg':feedback['remaining_propellant_kg']-baseline['remaining_propellant_kg'],
                           'correction_dv':feedback['correction_dv'],
                           'decisions':[e['decision'] for e in feedback['replanning_events']]})
        worst=max(rows,key=lambda r:r['terminal_eccentricity'])
        report={'directory':directory.name,'pairs':len(paired),
                'baseline_capture':sum(r['baseline_capture'] for r in paired),
                'feedback_capture':sum(r['feedback_capture'] for r in paired),
                'correction_traces':sum(r['correction_dv']>0 for r in paired),
                'max_correction_dv':max(r['correction_dv'] for r in paired),
                'max_time_penalty_days':max(r['time_delta_days'] for r in paired),
                'max_time_saving_days':-min(r['time_delta_days'] for r in paired),
                'reserve_violations':sum(r['reserve_violation'] for r in rows),
                'worst_case':{'scenario':worst['scenario']['name'],'feedback':worst['feedback'],
                              'eccentricity':worst['terminal_eccentricity']},
                'paired_results':paired}
        if directory.name.endswith('revision5'):
            report['fixed_departure_equal'] = all(
                arms[False]['trajectory_samples'][0]['true_state']==arms[True]['trajectory_samples'][0]['true_state']
                for arms in groups.values())
            report['all_corrections_after_120days'] = all(
                event['elapsed_days']>=120 for row in rows
                for event in row['replanning_events'] if event['decision']=='bounded_correction')
            report['correction_cases'] = [
                row for row in paired if row['correction_dv']>0]
        comparisons.append(report)
        (directory/'paired_comparison.json').write_text(json.dumps(report,indent=2))
    latest=directories[-1]
    rows=[json.loads(line) for line in (latest/'runs.jsonl').read_text().splitlines()]
    labels=sorted({r['scenario']['name'] for r in rows})
    fig,axes=plt.subplots(3,1,figsize=(11,10),sharex=True,layout='constrained')
    for feedback,color,label,offset in [(False,'#777777','Nominal transfer',-.18),(True,'#007e86','Observed-state replanning',.18)]:
        ordered=[next(r for r in rows if r['feedback']==feedback and r['scenario']['name']==name) for name in labels]
        x=np.arange(len(labels))+offset
        bars=axes[0].bar(x,[r['completion_days'] for r in ordered],width=.35,color=color,label=label)
        for bar,run in zip(bars,ordered):
            if not run['capture']:
                bar.set_hatch('xx')
                bar.set_edgecolor('#be3737')
        axes[1].bar(x,[r['remaining_propellant_kg'] for r in ordered],width=.35,color=color)
        axes[2].bar(x,[r['correction_dv'] for r in ordered],width=.35,color=color)
    axes[0].set_ylabel('Terminal epoch (days; hatched = capture failed)')
    axes[0].legend()
    axes[1].set_ylabel('Remaining propellant (kg)')
    axes[1].axhline(80,color='#be3737',linestyle='--',label='Hard operational reserve')
    axes[1].legend()
    axes[2].set_ylabel('Correction delta-v (m/s)')
    axes[2].set_xticks(range(len(labels)),labels,rotation=60,ha='right')
    fig.suptitle('Fixed departure, disturbance at day 120: 110 kg planning / 80 kg hard reserve')
    fig.savefig(latest/'paired_validation.png',dpi=160)
    plt.close(fig)
    (latest/'sprint_comparison.json').write_text(json.dumps(comparisons,indent=2))
    # Every source snapshot and raw run is retained for all iterations.
    for directory in directories:
        hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                for p in directory.iterdir() if p.is_file() and p.name!='artifact_hashes.json'}
        (directory/'artifact_hashes.json').write_text(json.dumps(hashes,indent=2,sort_keys=True))
    print(json.dumps([{k:v for k,v in row.items() if k!='paired_results'} for row in comparisons],indent=2))


if __name__=='__main__':
    main()
