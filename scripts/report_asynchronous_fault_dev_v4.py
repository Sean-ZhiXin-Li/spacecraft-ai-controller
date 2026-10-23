"""Evidence-aware reporting; unexposed missions are reported separately."""
import hashlib
import json
from pathlib import Path
import numpy as np


def analyze(rows):
    pairs=list(zip(rows[::2],rows[1::2]))
    for a,b in pairs:
        assert a['case']==b['case'] and not a['quarantine'] and b['quarantine']
        assert a['initial_state']==b['initial_state'] and a['departure_state']==b['departure_state']
    def exposed(row):
        return (row['actual_fault_epoch'] is not None if row['case']['fault_kind']!='sensor'
                else any(s['days']>=row['case']['fault_days'] for s in row['samples']))
    exposed_pairs=[(a,b) for a,b in pairs if exposed(a) and exposed(b)]
    result={'pairs':len(pairs),'exposed_pairs':len(exposed_pairs),
        'unexposed':[{'seed':a['case']['seed'],'kind':a['case']['fault_kind']}
                     for a,b in pairs if not exposed(a) and not exposed(b)],
        'unequal_exposure':sum(exposed(a)!=exposed(b) for a,b in pairs),
        'reference_capture':sum(a['capture'] for a,b in pairs),
        'quarantine_capture':sum(b['capture'] for a,b in pairs),
        'rescued':sum(b['capture'] and not a['capture'] for a,b in pairs),
        'regressed':sum(a['capture'] and not b['capture'] for a,b in pairs),
        'reserve_violations':sum(r['reserve_violation'] for r in rows),
        'ambiguity_burns':sum(r['burns_during_ambiguity'] for r in rows),
        'quarantine_correction_traces':sum(b['correction_dv']>0 for a,b in pairs),
        'reference_correction_traces':sum(a['correction_dv']>0 for a,b in pairs),
        'max_extra_fuel_kg':max(a['remaining_propellant_kg']-b['remaining_propellant_kg'] for a,b in pairs),
        'max_fuel_saved_kg':max(b['remaining_propellant_kg']-a['remaining_propellant_kg'] for a,b in pairs),
        'max_completion_delay_days':max(b['completion_days']-a['completion_days'] for a,b in pairs),
        'subgroups':[]}
    for kind in ('sensor','physical','combined'):
        group=[(a,b) for a,b in exposed_pairs if a['case']['fault_kind']==kind]
        result['subgroups'].append({'kind':kind,'exposed_pairs':len(group),
             'reference_capture':sum(a['capture'] for a,b in group),
             'quarantine_capture':sum(b['capture'] for a,b in group)})
    for flag in (False,True):
        subset=[r for r in rows if r['quarantine']==flag and exposed(r)]
        records=[(r,f) for r in subset for f in r['fault_records']
                 if f['detected_days']>=r['case']['fault_days'] and 'confirmed_days' in f]
        result['quarantine' if flag else 'reference']={
            'confirmed_late_records':len(records),
            'classifications':{k:sum(f['classification']==k for r,f in records)
                for k in ('sensor_bias_jump','physical_state_change','combined_or_model_mismatch')},
            'onset_to_confirmation_days':[min(f['confirmed_days']-r['case']['fault_days'] for r,f in records),
                                          max(f['confirmed_days']-r['case']['fault_days'] for r,f in records)] if records else None,
            'max_terminal_velocity_estimation_error_mps':max(
                np.linalg.norm(np.array(r['samples'][-1]['observed_state'])[2:4]-
                               np.array(r['samples'][-1]['true_state'])[2:4]) for r in subset)}
    return result


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root=Path(__file__).resolve().parents[1]
    for label in ('development','holdout','sparse'):
        directory=root/f'development_results/asynchronous_fault_v4_2026-10-23_{label}'
        recorded=json.loads((directory/'artifact_hashes.json').read_text())
        assert all(hashlib.sha256((directory/name).read_bytes()).hexdigest()==digest for name,digest in recorded.items())
        rows=[json.loads(line) for line in (directory/'runs.jsonl').read_text().splitlines()]
        result=analyze(rows)
        path=directory/'comparison.json'
        path.write_text(json.dumps(result,indent=2))
        print(label,json.dumps(result),flush=True)
        if label=='holdout':
            fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
            for flag,color,name in ((False,'#7b7b7b','v3 reference'),(True,'#148473','Quarantined intervals')):
                subset=[r for r in rows if r['quarantine']==flag]
                x=np.arange(len(subset))
                axes[0].scatter(x,[r['terminal_eccentricity'] for r in subset],s=22,color=color,label=name)
                axes[1].scatter(x,[r['remaining_propellant_kg'] for r in subset],s=22,color=color)
            axes[0].set_yscale('log')
            axes[0].axhline(.001,color='firebrick',linestyle='--',label='Capture limit')
            axes[0].set_ylabel('Terminal eccentricity')
            axes[0].legend(fontsize=8)
            axes[1].axhline(80,color='firebrick',linestyle='--')
            axes[1].set_ylabel('Remaining propellant (kg); floor = 80')
            for ax in axes:
                ax.set_xlabel('Paired condition index (12 seeds x 3 classes)')
            fig.suptitle('Asynchronous faults: 30 exposed pairs, 6 unexposed pairs')
            fig.savefig(directory/'comparison.png',dpi=160)
            plt.close(fig)
        names=['comparison.json']+(['comparison.png'] if label=='holdout' else [])
        (directory/'report_hashes.json').write_text(json.dumps(
            {name:hashlib.sha256((directory/name).read_bytes()).hexdigest() for name in names},indent=2))


if __name__=='__main__':
    main()
