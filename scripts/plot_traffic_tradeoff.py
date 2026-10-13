"""Regenerate the vertically arranged public-flow figure from audited metrics."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
root=Path(__file__).resolve().parents[1]
r=json.loads((root/'results/traffic_20261013/audit.json').read_text())
methods=['raw','single_step','vleg','raw_on_verified','global_summary']
labels=['Raw flows','Single-step','VLEG','Raw admitted','Global summary']
colors=['#245b8a','#9d8c70','#bf4d3e','#629694','#846ba5']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'pdf.fonttype':42})
fig,axes=plt.subplots(2,1,figsize=(6.2,7.2))
x=np.arange(5);means=[100*r['mean_metrics'][m]['macro_f1']['mean'] for m in methods];errors=[100*r['mean_metrics'][m]['macro_f1']['stdev'] for m in methods]
axes[0].bar(x,means,color=colors,width=.62,yerr=errors,capsize=3)
for position,mean,error in zip(x,means,errors):
 axes[0].text(position,mean+error+1.5,f'{mean:.2f}',ha='center',va='bottom',fontsize=9)
axes[0].set_xticks(x,['Raw\nflows','Single\nstep','VLEG','Raw\nadmitted','Global\nsummary']);axes[0].set_ylim(0,85)
axes[0].set_ylabel('Macro-F1 (%)');axes[0].set_title('(a) Balanced classification',loc='left',fontsize=10)
axes[0].grid(axis='y',alpha=.2);axes[0].set_axisbelow(True)
for method,label,color in zip(methods,labels,colors):
 seeds=[run for run in r['runs'] if run['stage']=='formal']
 xx=[100*s['metrics'][method]['false_positive_rate'] for s in seeds];yy=[100*s['metrics'][method]['attack_recall'] for s in seeds]
 axes[1].scatter(xx,yy,s=18,color=color,alpha=.27)
 axes[1].scatter(np.mean(xx),np.mean(yy),s=64,color=color,marker='D',edgecolor='white',linewidth=.7,label=label)
axes[1].set_xlim(0,100);axes[1].set_ylim(0,105);axes[1].set_xlabel('False positive rate (%)');axes[1].set_ylabel('Attack recall (%)')
axes[1].set_title('(b) Recall / false-alarm trade-off',loc='left',fontsize=10)
axes[1].grid(alpha=.2);axes[1].legend(loc='lower right',fontsize=7,framealpha=.95)
for ax in axes:
 ax.spines['top'].set_visible(False);ax.spines['right'].set_visible(False)
fig.tight_layout()
for p in [root/'JCN_submission/exp6_tradeoff.pdf',root/'exp6_tradeoff.pdf']:
 fig.savefig(p,bbox_inches='tight')
fig.savefig(root/'results/traffic_20261013/exp6_tradeoff.png',dpi=200,bbox_inches='tight')
