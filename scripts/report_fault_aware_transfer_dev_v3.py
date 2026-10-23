"""Offline comparisons and plots for position-confirmed fault handling."""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    root=Path(__file__).resolve().parents[1]
    for directory in sorted((root/'development_results').glob('fault_aware_v3_2026-10-23_*')):
        if not (directory/'summary.json').exists():
            continue
        rows=[json.loads(line) for line in (directory/'runs.jsonl').read_text().splitlines()]
        pairs=list(zip(rows[::2],rows[1::2]))
        matched=0
        unconfirmed=[]
        latencies=[]
        matrix={}
        for baseline,aware in pairs:
            assert baseline['case']==aware['case']
            assert baseline['departure_state']==aware['departure_state']
            case=aware['case']
            expected={'sensor':'sensor_bias_jump','physical':'physical_state_change',
                      'combined':'combined_or_model_mismatch'}[case['fault_kind']]
            diagnoses=[r for r in aware['fault_records'] if r['detected_days']>=case['fault_days']
                       and r['classification']!='ambiguous_velocity_jump']
            if diagnoses:
                diagnosis=diagnoses[0]
                matched+=int(diagnosis['classification']==expected)
                latencies.append(diagnosis['diagnosis_latency_days'])
                key=case['fault_kind']+' -> '+diagnosis['classification']
                matrix[key]=matrix.get(key,0)+1
            else:
                unconfirmed.append(case)
        report={'experiment':directory.name,'pairs':len(pairs),
             'distinct_seeds':len({r['case']['seed'] for r in rows}),
             'baseline_capture':sum(a['capture'] for a,b in pairs),
             'aware_capture':sum(b['capture'] for a,b in pairs),
             'baseline_survival':sum(a['survival'] for a,b in pairs),
             'aware_survival':sum(b['survival'] for a,b in pairs),
             'rescued':sum(not a['capture'] and b['capture'] for a,b in pairs),
             'regressed':sum(a['capture'] and not b['capture'] for a,b in pairs),
             'diagnosis_matches':matched,'diagnosis_denominator':len(pairs),
             'unconfirmed_cases':unconfirmed,'classification_matrix':matrix,
             'diagnosis_latency_range_days':[min(latencies),max(latencies)] if latencies else None,
             'reserve_violations':sum(r['reserve_violation'] for r in rows),
             'aware_ambiguity_burns':sum(b['burns_during_ambiguity'] for a,b in pairs),
             'aware_sensor_only_correction_traces':sum(b['correction_dv']>0 and b['case']['fault_kind']=='sensor' for a,b in pairs),
             'baseline_sensor_only_correction_traces':sum(a['correction_dv']>0 and a['case']['fault_kind']=='sensor' for a,b in pairs),
             'aware_max_e':max(b['terminal_eccentricity'] for a,b in pairs),
             'baseline_max_e':max(a['terminal_eccentricity'] for a,b in pairs),
             'minimum_final_propellant':min(r['remaining_propellant_kg'] for r in rows),
             'max_extra_propellant_used':max(0.,max(a['remaining_propellant_kg']-b['remaining_propellant_kg'] for a,b in pairs)),
             'max_propellant_saved':max(0.,max(b['remaining_propellant_kg']-a['remaining_propellant_kg'] for a,b in pairs)),
             'max_capture_delay_days':max(b['completion_days']-a['completion_days'] for a,b in pairs),
             'fault_classes_share_seeds':True,'subgroups':[]}
        for kind in sorted({r['case']['fault_kind'] for r in rows}):
            subset=[(a,b) for a,b in pairs if a['case']['fault_kind']==kind]
            report['subgroups'].append({'kind':kind,'pairs':len(subset),
                 'baseline_capture':sum(a['capture'] for a,b in subset),
                 'aware_capture':sum(b['capture'] for a,b in subset),
                 'baseline_correction_count':sum(a['correction_dv']>0 for a,b in subset),
                 'aware_correction_count':sum(b['correction_dv']>0 for a,b in subset)})
        (directory/'comparison.json').write_text(json.dumps(report,indent=2))
        if directory.name.endswith('holdout'):
            fig,axes=plt.subplots(2,1,figsize=(12,7),layout='constrained')
            for arm,color,label in [(0,'#888888','v2 trend + shared replanning'),(1,'#007e86','Position-confirmed fault handling')]:
                subset=[p[arm] for p in pairs]
                axes[0].plot(range(len(pairs)),[r['terminal_eccentricity'] for r in subset],'o-',color=color,label=label)
                axes[1].plot(range(len(pairs)),[r['remaining_propellant_kg'] for r in subset],'o-',color=color,label=label)
            axes[0].set_yscale('log')
            axes[0].axhline(.001,color='#b03030',linestyle='--',label='Capture eccentricity limit')
            axes[0].set_ylabel('Terminal eccentricity')
            axes[0].legend()
            axes[1].axhline(80,color='#b03030',linestyle='--',label='Hard operational reserve')
            axes[1].set_ylabel('Remaining propellant (kg)')
            axes[1].set_xlabel('Scenario index: sensor, physical, combined for each unseen seed')
            axes[1].legend()
            fig.suptitle('Late-fault holdout: same departures, disturbances, resources, and terminal predicates')
            fig.savefig(directory/'comparison.png',dpi=150)
            plt.close(fig)
        hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in directory.iterdir()
                if p.is_file() and p.name!='artifact_hashes.json'}
        (directory/'artifact_hashes.json').write_text(json.dumps(hashes,indent=2,sort_keys=True))
        print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
