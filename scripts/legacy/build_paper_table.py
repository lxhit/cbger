#!/usr/bin/env python3
"""Build the camera-ready PBGER-v0.8 table and paired user-cluster statistics."""
from __future__ import annotations
import json, math
from collections import defaultdict
from pathlib import Path
import numpy as np
from cbger.io import read_jsonl

R=Path(__file__).resolve().parents[2]; SEEDS=(20260831,20260832,20260833)
DATA=R/'data/processed/pbger_v0_8_lite/pbger_v0_8.jsonl'
BASE={
 'PR-Net':'outputs/v08/formal/prnet/raw/seed_{seed}.jsonl',
 'QD-DETR':'outputs/v08/formal/qddetr/raw/seed_{seed}.jsonl',
 'TR-DETR':'outputs/v08/formal/tr_detr/raw/seed_{seed}.jsonl',
 'FlashVTG':'outputs/v08/formal/flashvtg/raw/seed_{seed}.jsonl',
 'MQVTG':'outputs/v08/formal/mqvtg/raw/seed_{seed}.jsonl'}
SUM={
 'Frozen CLIP similarity':'outputs/v08/cross_ladder/clip_similarity/summary.json',
 'PR-Net':'outputs/v08/formal/prnet/summary.json','QD-DETR':'outputs/v08/formal/qddetr/summary.json',
 'TR-DETR':'outputs/v08/formal/tr_detr/summary.json','FlashVTG':'outputs/v08/formal/flashvtg/summary.json',
 'MQVTG':'outputs/v08/formal/mqvtg/summary.json','PBGER-Lite':'outputs/v08/pbger_lite/lite/summary.json',
 'PBGER-Lite + Structured CF':'outputs/v08/pbger_lite/lite_cf/summary.json'}
PB='outputs/v08/pbger_lite/lite_cf/raw/seed_{seed}.jsonl'

def scoremap(x): return {y['segment_id']:float(y['score']) for y in x['ranked_candidates']}
def mean_score(x): return float(np.mean([float(y['score']) for y in x['ranked_candidates']]))
def raw_score(x): return float(x['sample_score'])
def measures(rows,dataset,scorefn):
 pred={x['sample_id']:x for x in rows if x['split']=='test'}; out={}
 pos=[x for x in dataset.values() if x['split']=='test' and int(x['relevance'])==1]
 neg={x['counterfactual_of']:x for x in dataset.values() if x['split']=='test' and x.get('counterfactual_of')}
 for x in pos:
  p=pred[x['sample_id']]; n=neg[x['sample_id']]; cp=pred[n['sample_id']]
  evid={z['segment_id'] for z in x['timeline'] if z['role']=='evidence'}
  ranks=[i+1 for i,z in enumerate(p['ranked_candidates']) if z['segment_id'] in evid]
  ps,cs=scoremap(p),scoremap(cp); removed=n['provenance']['removed_segment_id']; replacement=n['provenance']['replacement_segment_id']
  out[x['sample_id']]={'user_id':x['user_id'],'mrr':1/min(ranks) if ranks else 0.0,
    'pair_accuracy':float(scorefn(p)>scorefn(cp)),'intervention_consistency':float(ps[removed]>cs[replacement])}
 return out
def cluster_boot(pb,base,key,rng,draws=10000):
 by=defaultdict(list)
 for s in SEEDS:
  for sid,x in pb[s].items(): by[x['user_id']].append(x[key]-base[s][sid][key])
 users=sorted(by); obs=float(np.mean([v for u in users for v in by[u]])); vals=[]
 for _ in range(draws):
  pick=rng.choice(users,len(users),replace=True); vals.append(float(np.mean([v for u in pick for v in by[u]])))
 vals=np.asarray(vals); p=min(1.0,2*min(float((vals<=0).mean()),float((vals>=0).mean())))
 return {'delta':obs,'ci95':[float(x) for x in np.quantile(vals,[.025,.975])],'p':p,'clusters':len(users)}
def fmt(x): return f"{x['mean']:.4f} ± {x['std']:.4f}"

def main():
 dataset={x['sample_id']:x for x in read_jsonl(DATA)}
 summaries={k:json.loads((R/v).read_text()) for k,v in SUM.items()}
 mean_report=json.loads((R/'outputs/v08/mean_pooling_final/results.json').read_text())['backbones']['CLIP']
 cf=summaries['PBGER-Lite + Structured CF']; final=json.loads(json.dumps(cf)); final['method']='PBGER-final (Structured CF + Mean)'
 final['metrics']['pair_accuracy']=mean_report['pair_accuracy']['cf_mean']; summaries['PBGER-final']=final
 pb={s:measures(list(read_jsonl(R/PB.format(seed=s))),dataset,mean_score) for s in SEEDS}
 rng=np.random.default_rng(20260831); significance={}
 for name,path in BASE.items():
  base={s:measures(list(read_jsonl(R/path.format(seed=s))),dataset,raw_score) for s in SEEDS}
  significance[name]={k:cluster_boot(pb,base,k,rng) for k in ('mrr','pair_accuracy','intervention_consistency')}
 order=('Frozen CLIP similarity','PR-Net','QD-DETR','TR-DETR','FlashVTG','MQVTG','PBGER-Lite','PBGER-Lite + Structured CF','PBGER-final')
 metrics=('mrr','ndcg_at_1','ndcg_at_3','ndcg_at_5','pair_accuracy','intervention_consistency')
 out=R/'outputs/v08/paper_main'; out.mkdir(parents=True,exist_ok=True)
 report={'dataset':'PBGER-v0.8','seeds':SEEDS,'methods':{k:summaries[k] for k in order},'significance_vs_pbger_final':significance}
 (out/'main_table.json').write_text(json.dumps(report,indent=2))
 lines=['# PBGER-v0.8 paper main table','', '| Method | MRR | NDCG@1 | NDCG@3 | NDCG@5 | PairAcc | Intervention |','|---|---:|---:|---:|---:|---:|---:|']
 for name in order:
  m=summaries[name]['metrics']; lines.append('| '+name+' | '+' | '.join(fmt(m[x]) for x in metrics)+' |')
 lines += ['','## Paired user-cluster bootstrap: PBGER-final minus baseline','', '| Baseline | MRR Δ [95% CI], p | PairAcc Δ [95% CI], p | Intervention Δ [95% CI], p |','|---|---|---|---|']
 for name,x in significance.items():
  cells=[]
  for key in ('mrr','pair_accuracy','intervention_consistency'):
   z=x[key]; cells.append(f"{z['delta']:+.4f} [{z['ci95'][0]:+.4f},{z['ci95'][1]:+.4f}], p={z['p']:.4f}")
  lines.append('| '+name+' | '+' | '.join(cells)+' |')
 (out/'main_table.md').write_text('\n'.join(lines)+'\n'); print('\n'.join(lines))
 # Canonical Mean-pooling summary for downstream paper scripts.
 canon=R/'outputs/v08/pbger_final_mean'; canon.mkdir(parents=True,exist_ok=True); (canon/'summary.json').write_text(json.dumps(final,indent=2))
if __name__=='__main__': main()
