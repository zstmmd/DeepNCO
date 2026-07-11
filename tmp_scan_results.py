import csv,json,math,os
from pathlib import Path
root=Path('result/stacks_s1_s9_user_repro_20260620')
baseline=json.load(open('experiments/configs/stacks_s1_s9_gurobi_baseline_20260620.json',encoding='utf-8'))['details']
base={r['case']:r for r in baseline}
rows=[]
for p in root.rglob('run_details.json'):
    try:
        data=json.load(open(p,encoding='utf-8'))
    except Exception:
        continue
    for r in (data if isinstance(data,list) else [data]):
        case=str(r.get('scale') or r.get('case') or '').upper()
        if not case.startswith('STACKS-S'): continue
        rows.append({
            'case':case,
            'dir':str(p.parent).replace('\\','/'),
            'cmax':float(r.get('model_cmax',float('nan'))),
            'runtime':float(r.get('runtime_sec',float('nan'))),
            'gap':float(r.get('model_gap',float('nan'))),
            'vars':r.get('model_var_count_total'),
            'arcs':r.get('u_arc_count'),
            'status':r.get('status'),
        })
for case in sorted(base):
    b=base[case]
    cand=[r for r in rows if r['case']==case]
    def score(r):
        return abs(r['cmax']-float(b['model_cmax']))*100000 + abs(r['runtime']-float(b['runtime_sec'])) + abs(r['gap']-float(b['model_gap']))*1000
    cand=sorted(cand,key=score)
    print('\n',case,'baseline',b)
    for r in cand[:8]: print(r)