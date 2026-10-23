"""Read recorded comparisons and diagnose the retained development no-plan case."""
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
import numpy as np
from scripts import multiregime_transfer_research as m


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root=Path(__file__).resolve().parents[1]
    directory=root/'development_results/multiregime_transfer_2026-10-23_holdout'
    hashes=json.loads((directory/'artifact_hashes.json').read_text())
    assert all(hashlib.sha256((directory/n).read_bytes()).hexdigest()==h for n,h in hashes.items())
    rows=[json.loads(line) for line in (directory/'runs.jsonl').read_text().splitlines()]
    groups={method:[r for r in rows if r['method']==method]
            for method in ('legacy','matched_outward','bidirectional')}
    new=groups['bidirectional']
    common=[(a,b) for a,b in zip(groups['matched_outward'],new) if a['result']['capture'] and b['result']['capture']]
    report={'holdout_tasks':len(new),'outside_captures':sum(r['case']['outside'] and r['result']['capture'] for r in new),
        'inside_captures':sum(not r['case']['outside'] and r['result']['capture'] for r in new),
        'min_final_propellant_kg':min(r['result']['propellant_kg'] for r in new),
        'max_eccentricity':max(r['result']['eccentricity'] for r in new),
        'completion_days_range':[min(r['result']['completion_days'] for r in new),max(r['result']['completion_days'] for r in new)],
        'common_capture_pairs':len(common),
        'max_absolute_common_completion_difference_days':max(abs(a['result']['completion_days']-b['result']['completion_days']) for a,b in common),
        'max_absolute_common_propellant_difference_kg':max(abs(a['result']['propellant_kg']-b['result']['propellant_kg']) for a,b in common),
        'max_prediction_vs_execution_arrival_error_days':max(abs(r['plan']['coast_days']-r['result']['completion_days']) for r in new),
        'legacy_limitation':'legacy prediction search retains an internal 2000-day cutoff; matched_outward is the primary control',
        'not_global_infeasibility':'no_plan denotes failure of this finite multi-start search'}
    case=m.tasks('development')[3]
    nav=m.legacy.Navigation(m.legacy.Scenario('multiregime',case['seed'],0,.2,1e5))
    candidate=m.search(nav.observe(np.array(case['state'])),True,deadline=10000)
    report['development_failure_diagnostic']={'seed':case['seed'],'original_deadline_days':4400,
        'diagnostic_deadline_days':10000,'candidate':asdict(candidate) if candidate else None,
        'physical_execution':False,'included_in_success_counts':False}
    (directory/'comparison.json').write_text(json.dumps(report,indent=2))
    fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    for ax,method in zip(axes,('matched_outward','bidirectional')):
        for success,marker,color in ((False,'x','#b94141'),(True,'o','#167b72')):
            subset=[r for r in groups[method] if r['result']['capture']==success]
            ax.scatter([r['case']['radius_ratio'] for r in subset],[r['case']['initial_vr'] for r in subset],
                       marker=marker,color=color,label='Capture + survival' if success else 'No plan')
        ax.axvline(1,color='#777777',linestyle=':')
        ax.axhline(0,color='#cccccc',linewidth=.7)
        ax.set_xlabel('Initial radius / target radius')
        ax.set_ylabel('Initial radial velocity (m/s)')
        ax.set_title(method.replace('_',' ').title())
        ax.legend(fontsize=8)
    fig.suptitle('24 unseen states; shared 4400-day deadline and 80 kg reserve')
    fig.savefig(directory/'comparison.png',dpi=160)
    plt.close(fig)
    (directory/'report_hashes.json').write_text(json.dumps({n:hashlib.sha256((directory/n).read_bytes()).hexdigest()
        for n in ('comparison.json','comparison.png')},indent=2))
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
