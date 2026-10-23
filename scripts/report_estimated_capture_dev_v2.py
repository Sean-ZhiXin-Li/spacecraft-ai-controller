"""Read-only numerical comparisons, plus new figures, for v2 capture estimation."""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    root=Path(__file__).resolve().parents[1]
    reports=[]
    for directory in sorted((root/'development_results').glob('estimated_capture_v2_2026-10-23_*')):
        rows=[json.loads(line) for line in (directory/'runs.jsonl').read_text().splitlines()]
        pairs=list(zip(rows[::2],rows[1::2]))
        for first,second in pairs:
            assert first['case']==second['case']
            assert first['initial_state']==second['initial_state']
            assert first['departure_state']==second['departure_state']
        changes=[second['remaining_propellant_kg']-first['remaining_propellant_kg'] for first,second in pairs]
        summary={'experiment':directory.name,'pairs':len(pairs),
                 'distinct_seeds':len({row['case']['seed'] for row in rows}),
                 'baseline_capture':sum(first['capture'] for first,second in pairs),
                 'new_capture':sum(second['capture'] for first,second in pairs),
                 'rescued':sum(not first['capture'] and second['capture'] for first,second in pairs),
                 'regressed':sum(first['capture'] and not second['capture'] for first,second in pairs),
                 'baseline_survival':sum(first['survived_100days_sampled'] for first,second in pairs),
                 'new_survival':sum(second['survived_100days_sampled'] for first,second in pairs),
                 'baseline_worst_e':max(first['terminal_eccentricity'] for first,second in pairs),
                 'new_worst_e':max(second['terminal_eccentricity'] for first,second in pairs),
                 'max_extra_fuel_kg':max(0.,-min(changes)),'max_fuel_saved_kg':max(0.,max(changes)),
                 'minimum_final_fuel_kg':min(row['remaining_propellant_kg'] for row in rows),
                 'reserve_violations':sum(row['reserve_violation'] for row in rows),
                 'max_new_bias_error_mps':max(second['bias_error_mps'] for first,second in pairs),
                 'conditions_are_correlated_within_seed':True,
                 'subgroups':[]}
        for bias,drift in sorted({(r['case']['navigation_velocity_bound'],r['case']['velocity_bias_drift_per_day']) for r in rows}):
            subset=[p for p in pairs if p[0]['case']['navigation_velocity_bound']==bias
                    and p[0]['case']['velocity_bias_drift_per_day']==drift]
            summary['subgroups'].append({'initial_bias_bound':bias,'drift_mps_per_day':drift,
                 'pairs':len(subset),'baseline_capture':sum(p[0]['capture'] for p in subset),
                 'new_capture':sum(p[1]['capture'] for p in subset)})
        (directory/'comparison.json').write_text(json.dumps(summary,indent=2))
        if directory.name.endswith(('validation','holdout')):
            fig,axes=plt.subplots(2,1,figsize=(11,7),layout='constrained')
            for index,label,color in [(0,'Reference','#888888'),(1,'Position-aided estimate','#007e86')]:
                subset=[pair[index] for pair in pairs]
                axes[0].plot(range(len(pairs)),[r['terminal_eccentricity'] for r in subset],
                             'o-',color=color,label=label)
                axes[1].plot(range(len(pairs)),[r['remaining_propellant_kg'] for r in subset],
                             'o-',color=color,label=label)
            axes[0].set_yscale('log')
            axes[0].axhline(.001,color='#b03030',linestyle='--',label='Capture eccentricity limit')
            axes[0].set_ylabel('Terminal eccentricity')
            axes[0].legend()
            axes[1].axhline(80,color='#b03030',linestyle='--',label='Operational reserve')
            axes[1].set_ylabel('Remaining propellant (kg)')
            axes[1].set_xlabel('Predeclared scenario index (two conditions per seed)')
            axes[1].legend()
            fig.suptitle('Unseen seeds: '+('bias-rate tracking' if directory.name.endswith('holdout') else 'constant-bias correction'))
            fig.savefig(directory/'comparison.png',dpi=150)
            plt.close(fig)
        reports.append(summary)
        hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in directory.iterdir()
                if p.is_file() and p.name!='artifact_hashes.json'}
        (directory/'artifact_hashes.json').write_text(json.dumps(hashes,indent=2,sort_keys=True))
    print(json.dumps(reports,indent=2))


if __name__=='__main__':
    main()
