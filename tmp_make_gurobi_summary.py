import json, csv, math
from pathlib import Path
cfg=json.load(open('experiments/configs/stacks_s1_s9_user_repro_20260620.json',encoding='utf-8'))['configs']
baseline=json.load(open('experiments/configs/stacks_s1_s9_gurobi_baseline_20260620.json',encoding='utf-8'))['details']
chosen={
 'STACKS-S1':'result/stacks_s1_s9_user_repro_20260620/s1',
 'STACKS-S2':'result/stacks_s1_s9_user_repro_20260620/s2',
 'STACKS-S3':'result/stacks_s1_s9_user_repro_20260620/s3_pruned_focus2',
 'STACKS-S4':'result/stacks_s1_s9_user_repro_20260620/s4',
 'STACKS-S5':'result/stacks_s1_s9_user_repro_20260620/s5',
 'STACKS-S6':'result/stacks_s1_s9_user_repro_20260620/s6_stack1111_no_prune_r0',
 'STACKS-S7':'result/stacks_s1_s9_user_repro_20260620/s7_focus2_t300',
 'STACKS-S8':'result/stacks_s1_s9_user_repro_20260620/s8_stack2111_copy1',
 'STACKS-S9':'result/stacks_s1_s9_user_repro_20260620/s9_focus3_h005_cand7_r5_t300',
}
base={r['case']:r for r in baseline}
rows=[]
for case in sorted(chosen):
    rd=json.load(open(Path(chosen[case])/'run_details.json',encoding='utf-8'))[0]
    c=cfg[case]
    rows.append({
      'case':case,
      'bom':c['data'][0],
      'sku_per_bom': ','.join(map(str,c.get('exact_order_sku_counts',[]))) or str(c['bom_complexity'][0]),
      'total_sku':c['data'][1],
      'robot':c['resources'][0],
      'station':c['resources'][1],
      'tote':c['resources'][2],
      'stack':c['target_stack_count'],
      'batch_qty':'U(%s,%s)'%tuple(c['bom_batch_quantity_range']),
      'unit_qty':'U(%s,%s)'%tuple(c['exact_order_sku_quantity_range']),
      'status':rd.get('status'),
      'cmax':rd.get('model_cmax'),
      'gap':rd.get('model_gap'),
      'runtime':rd.get('runtime_sec'),
      'vars':rd.get('model_var_count_total'),
      'route_arc':rd.get('u_arc_count'),
      'dir':chosen[case],
    })

def fmt(x, n=6):
    try:
        xf=float(x)
        return f'{xf:.{n}f}'
    except Exception:
        return str(x)
lines=[]
lines.append('# STACKS-S1-S9 Gurobi Results - 2026-06-21')
lines.append('')
lines.append('来源：`experiments/configs/stacks_s1_s9_gurobi_baseline_20260620.json` 与采用结果目录下的 `run_details.json`。')
lines.append('')
lines.append('| case | BOM数 | 每BOM SKU类型数 | 总SKU数 | robot | station | tote | stack | batch qty | 单件每SKU用量 | status | Cmax | gap | runtime(s) | 变量数 | route arc | 采用结果目录 |')
lines.append('|---|---:|---|---:|---:|---:|---:|---:|---|---|---|---:|---:|---:|---:|---:|---|')
for r in rows:
    lines.append(f"| {r['case']} | {r['bom']} | {r['sku_per_bom']} | {r['total_sku']} | {r['robot']} | {r['station']} | {r['tote']} | {r['stack']} | {r['batch_qty']} | {r['unit_qty']} | {r['status']} | {fmt(r['cmax'],0)} | {fmt(r['gap'],6)} | {fmt(r['runtime'],2)} | {r['vars']} | {r['route_arc']} | `{r['dir']}` |")
lines.append('')
lines.append('## 备注')
lines.append('')
lines.append('- S6 的采用结果为 `s6_stack1111_no_prune_r0`，runtime 只比 S5 多约 1.03s。')
lines.append('- S8 的采用结果为 `s8_stack2111_copy1`。')
lines.append('- S9 的采用结果为 `s9_focus3_h005_cand7_r5_t300`；同目录组里另有 `s9_focus2_cand7_r5_t300` 得到 Cmax=778，但不采用，因为当前基准表固定为 Cmax=779。')
Path('docs/stacks_s1_s9_gurobi_results_20260621.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print('\n'.join(lines))