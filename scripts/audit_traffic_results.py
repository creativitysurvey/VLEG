"""Independent offline audit of a completed Experiment 6 archive (no model calls)."""
import argparse
import collections
import csv
import hashlib
import json
import math
import random
import re
import statistics
import subprocess
from pathlib import Path
import numpy as np

METHODS = ('raw', 'single_step', 'vleg', 'raw_on_verified', 'global_summary')
SEEDS = list(range(20261020, 20261025))


def load(path):
    return json.loads(Path(path).read_text())


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def score(records):
    cm = {l: {p:0 for p in ('BENIGN','MALICIOUS','INVALID')} for l in ('BENIGN','MALICIOUS')}
    for r in records:
        cm[r['truth']][r['prediction']] += 1
    f1=[]
    for label in cm:
        tp=cm[label][label];fn=sum(cm[label].values())-tp
        fp=sum(cm[other][label] for other in cm if other != label)
        f1.append(2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.)
    lat=sorted(r['seconds'] for r in records)
    return {'n':len(records),'confusion':cm,
            'accuracy':sum(cm[l][l] for l in cm)/len(records),
            'macro_f1':sum(f1)/2,
            'attack_recall':cm['MALICIOUS']['MALICIOUS']/sum(cm['MALICIOUS'].values()),
            'benign_recall':cm['BENIGN']['BENIGN']/sum(cm['BENIGN'].values()),
            'false_positive_rate':cm['BENIGN']['MALICIOUS']/sum(cm['BENIGN'].values()),
            'invalid_predictions':sum(cm[l]['INVALID'] for l in cm),
            'latency_mean_s':statistics.mean(lat),'latency_p95_s':lat[math.ceil(.95*len(lat))-1],
            'prompt_tokens':sum(r['prompt_tokens'] for r in records),
            'completion_tokens':sum(r['completion_tokens'] for r in records)}


def equal_numeric(a,b):
    if isinstance(a,dict):
        assert set(a)==set(b),(set(a),set(b))
        for key in a:equal_numeric(a[key],b[key])
    elif isinstance(a,list):
        assert len(a)==len(b)
        for x,y in zip(a,b):equal_numeric(x,y)
    elif isinstance(a,(int,float)):
        assert math.isclose(a,b,rel_tol=1e-11,abs_tol=1e-11),(a,b)
    else:assert a==b,(a,b)


def audit(root,out,archive,repo):
    root=Path(root);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    assert load(root/'status.json')['status']=='complete'
    runs=[];files=[];configs=[];model_records=[];all_prompts=[]
    for stage,seed,n in [('pilot',20261012,100)]+[('formal',s,1000) for s in SEEDS]:
        folder=root/f'{stage}_{seed}'
        for name in ('prepared.json','checkpoint.json','predictions.json','summary.json','reflection_audit.csv'):
            p=folder/name;assert p.is_file(),p
            files.append({'path':str(p.relative_to(root)), 'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
        bundle=load(folder/'prepared.json');state=load(folder/'checkpoint.json');summary=load(folder/'summary.json');preds=load(folder/'predictions.json')
        assert bundle['fingerprint']==digest({k:v for k,v in bundle.items() if k!='fingerprint'})
        assert state['status']==summary['status']=='complete'
        assert bundle['manifest']['seed']==seed
        assert state['config']==summary['config']
        assert state['manifest']==summary['manifest']==bundle['manifest']
        assert state['config']['bundle_fingerprint']==bundle['fingerprint']
        assert state['config']['K']==state['config']['k']==3
        assert state['config']['num_ctx']==32768
        assert len(bundle['memory'])==len(state['memory'])==32
        assert len(bundle['test'])==n and len(preds)==n*len(METHODS)
        assert len({r['id'] for r in bundle['test']})==n
        for row in bundle['memory']+bundle['test']:assert row['id']==digest(row['features'])
        memory_ids={r['id'] for r in bundle['memory']};test_ids={r['id'] for r in bundle['test']}
        assert not (memory_ids & test_ids)
        assert memory_ids=={m['id'] for m in state['memory']}
        expected={(m,key) for m in METHODS for key in test_ids}
        by={(r['method'],r['id']):r for r in preds};truth={r['id']:r for r in bundle['test']}
        assert len(by)==len(preds) and set(by)==expected
        assert { (r['method'],r['id']):r for r in state['predictions'].values()}==by
        assert set(state['retrieval'])==test_ids
        mem={m['id']:m for m in state['memory']}
        histories=[]
        for m in state['memory']:
            assert 1<=len(m['history'])<=3
            assert m['single_step']==m['history'][0]['reflection']
            assert all(type(h['verification']['valid']) is bool for h in m['history'])
            assert (m['vleg'] is not None)==m['history'][-1]['verification']['valid']
            if m['vleg'] is not None:assert m['vleg']==m['history'][-1]['reflection']
            histories.extend(m['history'])
        identical=changed=admission_changed=0
        for key in test_ids:
            selection=state['retrieval'][key]['ids']
            assert len(selection)==3 and len(set(selection))==3 and set(selection)<=memory_ids
            admitted=[i for i in selection if mem[i]['vleg'] is not None]
            for method in METHODS:
                r=by[method,key]
                assert r['truth']==truth[key]['label'] and r['attack_category']==truth[key]['attack_category']
                assert r['retrieved_ids']==selection
                expected_count=0 if method=='global_summary' else len(admitted) if method in ('vleg','raw_on_verified') else 3
                assert r['effective_examples']==expected_count
                reply=r['reply'].strip().upper()
                assert r['prediction']==(reply if reply in ('BENIGN','MALICIOUS') else 'INVALID')
                assert r['calls']==1 and r['seconds']>=0 and r['context_characters']<=18000
            if len(admitted)==3:
                identical+=1;changed+=by['raw',key]['prediction']!=by['raw_on_verified',key]['prediction']
            else:admission_changed+=1
        recomputed={m:score([r for r in preds if r['method']==m]) for m in METHODS}
        equal_numeric(recomputed,summary['metrics'])
        coverage=sum(m['vleg'] is not None for m in state['memory'])/32
        assert coverage==summary['vleg_memory_coverage']
        for method,history in [('single_step',[m['history'][0] for m in state['memory']]),('vleg',histories)]:
            stages=('generator_cost',) if method=='single_step' else ('generator_cost','verifier_cost')
            for metric in ('seconds','prompt_tokens','completion_tokens','calls'):
                val=sum(h[st][metric] for h in history for st in stages)
                assert math.isclose(val,summary['consolidation_cost'][method][metric],rel_tol=1e-11)
        with (folder/'reflection_audit.csv').open(newline='') as f:
            human=list(csv.DictReader(f))
        reviewed=sum(bool(r['label_supported'] or r['fields_supported'] or r['overgeneralization'] or r['reviewer_notes']) for r in human)
        model_records.append(summary['model_metadata']);configs.append({k:v for k,v in summary['config'].items() if k!='bundle_fingerprint'})
        runs.append({'stage':stage,'seed':seed,'metrics':recomputed,'coverage':coverage,
                     'construction':summary['consolidation_cost'],'predictions':preds,'truth':truth,
                     'identical_raw_prompt_pairs':identical,'different_predictions_on_identical_prompts':changed,
                     'queries_affected_by_admission':admission_changed,'manual_reviewed_rows':reviewed,
                     'runtime':summary['runtime']})
    assert all(c==configs[0] for c in configs)
    assert all(m==model_records[0] and m['status']=='recorded' for m in model_records)
    formal=runs[1:];ids=sorted(formal[0]['truth']);pilot_ids=set(runs[0]['truth'])
    assert not pilot_ids & set(ids)
    assert all(r['truth']==formal[0]['truth'] for r in formal)
    # Verify the frozen source, not the audited repository after new scripts are added.
    commit=formal[0]['runtime']['git_commit']
    assert all(r['runtime']['git_commit']==commit for r in runs)
    paths=subprocess.check_output(['git','ls-tree','-r','--name-only',commit],cwd=repo,text=True).splitlines()
    paths=sorted(p for p in paths if p.startswith('traffic/') and p.endswith('.py') and p.count('/')==1)+['core/llm_client.py','core/verification_response.py']
    frozen={p:subprocess.check_output(['git','show',commit+':'+p],cwd=repo,text=True) for p in paths}
    assert digest(frozen)==configs[0]['code_sha256']
    means={m:{metric:{'mean':statistics.mean(r['metrics'][m][metric] for r in formal),
                     'stdev':statistics.stdev(r['metrics'][m][metric] for r in formal)}
              for metric in ('accuracy','macro_f1','attack_recall','false_positive_rate','latency_mean_s','latency_p95_s','prompt_tokens','completion_tokens')}
           for m in METHODS}
    uploaded=load(root/'formal_aggregate.json')
    for index,run in enumerate(formal):
        for method in METHODS:
            relevant=[r for r in run['predictions'] if r['method']==method and r['truth']=='MALICIOUS']
            per_attack={category:sum(r['prediction']=='MALICIOUS' for r in relevant if r['attack_category']==category)/sum(r['attack_category']==category for r in relevant) for category in sorted({r['attack_category'] for r in relevant})}
            equal_numeric(per_attack,uploaded['per_attack_recall_by_seed'][index][method])
    for method in METHODS:
        for metric in uploaded['seed_metrics'][method]:equal_numeric(means[method][metric],uploaded['seed_metrics'][method][metric])
    truth=np.array([formal[0]['truth'][key]['label']=='MALICIOUS' for key in ids])
    indices=[{(r['method'],r['id']):r for r in seed['predictions']} for seed in formal]
    codes={'BENIGN':0,'MALICIOUS':1,'INVALID':2}
    outputs=np.array([[[codes[index[m,key]['prediction']] for key in ids] for m in METHODS] for index in indices])
    comparisons=[(m,'raw') for m in METHODS if m!='raw']+[('vleg','single_step'),('vleg','raw_on_verified')]
    boot={a+'_vs_'+b:[] for a,b in comparisons};rng=random.Random(0)
    for _ in range(1000):
        draw=rng.choices(ids,k=len(ids));counts=collections.Counter(draw);weights=np.array([counts.get(key,0) for key in ids])
        f1=np.zeros((5,5))
        for label in (0,1):
            correct=(outputs==label)&(truth==label)
            tp=np.sum(correct*weights,axis=2);fp=np.sum(((outputs==label)&(truth!=label))*weights,axis=2)
            fn=np.sum(((outputs!=label)&(truth==label))*weights,axis=2)
            denominator=2*tp+fp+fn
            f1+=np.divide(2*tp,denominator,out=np.zeros_like(tp,dtype=float),where=denominator!=0)/2
        for a,b in comparisons:boot[a+'_vs_'+b].append(float(np.mean(f1[:,METHODS.index(a)]-f1[:,METHODS.index(b)])))
    paired={}
    for a,b in comparisons:
        key=a+'_vs_'+b;values=sorted(boot[key]);delta=[r['metrics'][a]['macro_f1']-r['metrics'][b]['macro_f1'] for r in formal]
        paired[key]={'mean_macro_f1_delta':statistics.mean(delta),'per_seed_delta':delta,
                     'conditional_flow_bootstrap_95pct_ci':[values[24],values[974]]}
        if b=='raw':
            source=uploaded['paired_macro_f1_delta_vs_raw'][a]
            equal_numeric(paired[key]['mean_macro_f1_delta'],source['mean'])
            equal_numeric(paired[key]['conditional_flow_bootstrap_95pct_ci'],source['conditional_test_cluster_95pct_ci'])
    costs={m:{metric:{'mean':statistics.mean(r['construction'][m][metric] for r in formal),
                    'stdev':statistics.stdev(r['construction'][m][metric] for r in formal)}
              for metric in ('seconds','prompt_tokens','completion_tokens','calls')} for m in ('single_step','vleg','global_summary')}
    attack_counts=dict(collections.Counter(r['attack_category'] for r in formal[0]['truth'].values()))
    cm_major=[{'truth':formal[0]['truth'][key]['label'],'prediction':'BENIGN','seconds':0.,'prompt_tokens':0,'completion_tokens':0} for key in ids]
    baseline=score(cm_major)
    notes=['Confidence intervals resample the same 1000 feature fingerprints synchronously across methods/seeds; no session IDs are available. These are conditional unadjusted intervals, not five independent test sets or production-network guarantees.',
           'Fixed method order and model prompt caching confound chat-latency comparisons; report timings descriptively, not as causal speed advantages.',
           'Temperature-zero calls differ on some identical raw/raw-on-verified prompts; admission effects of a few percentage points cannot be interpreted without accounting for this variability.',
           'Manual reflection-audit worksheets are blank; no completed human quality or causal audit is claimed.',
           'Feature-conflict exclusion and deduplication change the target population; results concern the cleaned representation population.']
    report={'status':'passed','source_archive_sha256':hashlib.sha256(Path(archive).read_bytes()).hexdigest(),
            'source_commit':commit,'source_code_sha256':configs[0]['code_sha256'],
            'config':configs[0],'model_metadata':model_records[0],
            'formal_unique_test_flows':1000,'formal_predictions':25000,'pilot_predictions':500,
            'class_counts':dict(collections.Counter(r['label'] for r in formal[0]['truth'].values())),
            'attack_counts':attack_counts,'mean_metrics':means,'paired_comparisons':paired,
            'construction_cost':costs,'always_benign_reference':baseline,
            'runs':[{k:v for k,v in r.items() if k not in ('predictions','truth')} for r in runs],
            'mean_coverage':statistics.mean(r['coverage'] for r in formal),
            'formal_invalid_counts':{m:sum(r['metrics'][m]['invalid_predictions'] for r in formal) for m in METHODS},
            'identical_prompt_pairs':sum(r['identical_raw_prompt_pairs'] for r in formal),
            'different_predictions_on_identical_prompts':sum(r['different_predictions_on_identical_prompts'] for r in formal),
            'notes':notes,'files':files}
    (out/'audit.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    (out/'formal_aggregate.json').write_text(json.dumps(uploaded,indent=2)+'\n')
    with (out/'metrics_by_seed.csv').open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(['seed','method','accuracy','macro_f1','attack_recall','false_positive_rate','invalid','chat_mean_s'])
        for r in formal:
            for m in METHODS:
                v=r['metrics'][m];writer.writerow([r['seed'],m,v['accuracy'],v['macro_f1'],v['attack_recall'],v['false_positive_rate'],v['invalid_predictions'],v['latency_mean_s']])
    print(json.dumps({k:report[k] for k in ('status','formal_predictions','mean_coverage','formal_invalid_counts','paired_comparisons')},indent=2))
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',required=True);p.add_argument('--output',required=True);p.add_argument('--archive',required=True);p.add_argument('--repo',default='.')
    a=p.parse_args();audit(a.root,a.output,a.archive,a.repo)
