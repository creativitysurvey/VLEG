"""Offline evidence audit; machine checks are not a human explanation audit.
Run with --flow-root pointing to the original completed batch archive.
No evaluation predictions are changed and no model calls are made.
"""
import argparse, csv, hashlib, json, operator, re, statistics, subprocess
from collections import Counter
from pathlib import Path

OPS={'==':operator.eq,'!=':operator.ne,'>':operator.gt,'<':operator.lt,'>=':operator.ge,'<=':operator.le}
ATOM=re.compile(r'`?(\w+)`?\s*(==|!=|>=|<=|>>|>|<|\bis\b)\s*("[^"]*"|\d+(?:\.\d+)?|`?\w+`?)',re.I)
def parse_rule(text,features):
    match=re.search(r'\bif\b(.*?)\bthen\b',text,re.I)
    if not match:return None,'no complete IF/THEN condition'
    clause=match.group(1);atoms=[];cursor=0
    for m in ATOM.finditer(clause):
        gap=clause[cursor:m.start()]
        if re.sub(r'\band\b|[\s(),]','',gap,flags=re.I):return None,'unsupported condition syntax'
        key,op,value=m.groups();value=value.strip('`"');cursor=m.end()
        if key not in features or op=='>>':return None,'unknown field or ambiguous >> comparison'
        if value in features:rhs=('field',value)
        else:
            try:rhs=('literal',float(value))
            except ValueError:rhs=('literal',value)
        atoms.append((key,'==' if op.lower()=='is' else op,rhs))
    if re.sub(r'[\s(),]','',clause[cursor:]) or not atoms:return None,'unparsed condition suffix'
    return atoms,''
def applies(atoms,features):
    return all(OPS[op](features[key],features[rhs[1]] if rhs[0]=='field' else rhs[1]) for key,op,rhs in atoms)
def label(text):
    after=re.split(r'\bthen\b',text,flags=re.I)[-1]
    found=re.findall(r'\bBENIGN\b|\bMALICIOUS\b',after)
    return found[0] if found else None

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--flow-root',type=Path,required=True);ap.add_argument('--output',type=Path,default=Path('results/explanation_provenance_20261013'));a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    checkpoints=[];historical={}
    for p in sorted(a.flow_root.glob('formal_*/checkpoint.json')):
        d=json.loads(p.read_text());checkpoints.append((p,d))
        for m in d['memory']:
            f,l=m['raw'].split('\nHistorical label: ');historical[m['id']]=(json.loads(f),l)
    rows=[]
    for p,d in checkpoints:
        for m in d['memory']:
            features,true=historical[m['id']]
            for method in ['single_step','vleg']:
                text=m[method]
                if text is None:continue
                atoms,error=parse_rule(text,features);pred=label(text)
                matches=[];conflicts=[]
                if atoms is not None:
                    for hid,(f,l) in historical.items():
                        if applies(atoms,f):
                            matches.append(hid)
                            if l!=pred and hid!=m['id']:conflicts.append(hid)
                rows.append({'seed':p.parent.name,'memory_id':m['id'],'condition':method,'reflection':text,'historical_label':true,'stated_label':pred,'label_matches_source':pred==true,'condition_parseable':atoms is not None,'source_conditions_true':applies(atoms,features) if atoms is not None else '', 'syntax_limit':error,'other_history_matches':len([x for x in matches if x!=m['id']]),'history_counterexamples':len(conflicts),'first_counterexample_id':conflicts[0] if conflicts else '', 'generalization_status':'counterexample_found' if conflicts else 'not_established','review_type':'deterministic_assisted_screen','human_reviewer':'','human_fields_supported':'','human_overgeneralization':'','human_notes':''})
    with (a.output/'reflection_screen.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    totals={}
    for method in ['single_step','vleg']:
        rr=[r for r in rows if r['condition']==method];totals[method]={'n':len(rr),'source_label_matches':sum(r['label_matches_source'] for r in rr),'parseable':sum(r['condition_parseable'] for r in rr),'source_conditions_true':sum(r['source_conditions_true'] is True for r in rr),'source_conditions_false':sum(r['source_conditions_true'] is False for r in rr),'rules_with_historical_counterexamples':sum(r['history_counterexamples']>0 for r in rr)}
    (a.output/'reflection_screen_summary.json').write_text(json.dumps({'review_type':'deterministic_assisted_screen_not_human_audit','historical_unique_flows':len(historical),'test_flows_used':0,'human_reviews_completed':0,'methods':totals,'limits':['Only explicit conjunctive IF/THEN atoms evaluated; ambiguous syntax is unresolved.','Counterexamples use the union of historical draws, excluding the source of each rule; not independent validation.','No counterexample found does not establish causal or universal validity.','No model-output or metric changes. Human columns intentionally blank.']},indent=2))
    repo=Path(__file__).resolve().parents[1]
    def git(*args):return subprocess.check_output(['git',*args],cwd=repo)
    original=git('rev-parse','fa33091').decode().strip();frozen=git('rev-parse','7c39efe').decode().strip();inventory=[];derived={}
    for tag,rev in [('archived_7b',original),('latest_archived_70b',frozen)]:
        destination=a.output/tag;destination.mkdir(exist_ok=True)
        for path in git('ls-tree','-r','--name-only',rev,'outputs').decode().splitlines():
            if not path.endswith('.json'):continue
            data=git('show',rev+':'+path);(destination/Path(path).name).write_bytes(data)
            inventory.append({'snapshot':tag,'commit':rev,'path':path,'sha256':hashlib.sha256(data).hexdigest()})
        def load(name):return json.loads(git('show',rev+':outputs/'+name))
        exp1=load('experiment1_multiseed_results.json');exp2=load('experiment2_multiseed_results.json');exp3=load('experiment3_multiseed_results.json');exp4=load('experiment4_multiseed_results.json');exp5=load('experiment5_results.json')
        d={'model_declaration':re.search(r'^MODEL_NAME\s*=\s*(.*)$',git('show',rev+':core/llm_client.py').decode(),re.M).group(1),'exp1_pooled_level3':{k:statistics.mean(x['results'][k]['accuracy'] for x in exp1['level3_by_seed']) for k in ['VLEG','HDR','Reflexion','ICL']},'exp1_seed_values':exp1['level3_by_seed'],'exp2_pooled_level3':{k:statistics.mean(x['modality_results'][k]['accuracy'] for x in exp2['level3_by_seed']) for k in ['hybrid','abstract_only','raw_only']},'exp3_delta_mean':statistics.mean(x['delta_overall'] for x in exp3),'exp3_delta_sample_sd':statistics.stdev(x['delta_overall'] for x in exp3),'exp4_by_seed':exp4['level3_by_seed'],'exp5_keyword_recount':[],'per_item_accuracy_checks':0,'per_item_accuracy_failures':[]}
        def walk(x,path=''):
            if isinstance(x,dict):
                items=x.get('per_item')
                if items and 'accuracy' in x:
                    flags=[i['predicted']==i['combo']['label'] for i in items];acc=sum(flags)/len(flags);d['per_item_accuracy_checks']+=1
                    if abs(acc-x['accuracy'])>1e-10 or any(i['correct']!=b for i,b in zip(items,flags)):d['per_item_accuracy_failures'].append(path)
                for k,v in x.items():walk(v,path+'/'+k)
            elif isinstance(x,list):
                for i,v in enumerate(x):walk(v,path+'/'+str(i))
        for p in destination.glob('*.json'):walk(json.loads(p.read_text()),p.name)
        import sys
        sys.path.insert(0,str(repo))
        from core.task_env import all_combinations, make_level3_split
        for x in exp5:
            details=x['details']
            # Ground truth pairing for 7B is reconstructed from verbatim audit-script lists; 70B file lacks source combos.
            combos=all_combinations() if x['level']=='Level2' else make_level3_split(seed=42)[0]
            if len(details)<len(combos):
                combos=[c for i,c in enumerate(combos) if not (tag=='archived_7b' and x['level']=='Level2' and x['strategy']=='VLEG' and i==6)]
            conclusions=[re.findall(r'\bANOMALY\b|\bNORMAL\b',t['reflection'])[-1] for t in details]
            label_matches=[l==c['label'] for l,c in zip(conclusions,combos)]
            flags=[bool(re.search(r'\b(texture|size|small|large|striped|solid)\b',t['reflection'],re.I)) for t in details]
            flagged=sum(bool(re.search(r'\b(texture|size|small|large|striped|solid)\b',t['reflection'],re.I)) for t in details)
            d['exp5_keyword_recount'].append({'level':x['level'],'strategy':x['strategy'],'n':len(details),'mentions_irrelevant_attributes':flagged,'source_label_matches_reconstructed_order':sum(label_matches),'keyword_mentions_among_label_correct':sum(b and c for b,c in zip(flags,label_matches)),'original_llm_hallucination_flags':sum(t['hallucinated'] for t in details)})
        derived[tag]=d
    (a.output/'symbolic_provenance.json').write_text(json.dumps({'snapshots':inventory,'derived':derived,'limits':['Git records establish archived byte provenance, not execution attestation.','Old snapshots omit exact model digests, runtime manifests and raw verifier replies.','7B and 70B baseline implementations differ; scale cannot be isolated.','The 7B audit script contains manually transcribed reflections; it is not a complete original checkpoint.','Old bool(string) parser risks cannot be quantified without original replies.']},indent=2))
    print(json.dumps(totals,indent=2));print('historical flows',len(historical));print('derived symbolic',[(k,v['exp1_pooled_level3'],v['exp3_delta_mean'],v['per_item_accuracy_checks'],v['per_item_accuracy_failures']) for k,v in derived.items()])
if __name__=='__main__':main()
