"""A/B: does the scorer's nested-\[ inside \begin{array} cost CDM points?
Same gt, same pred content -- only the invalid nesting differs. Zero GPU."""
import json, re, sys, tempfile
from metrics.cdm_metric import CDM

r = json.load(open('result/otslhtml1651_quick_match_display_formula_result.json'))
s = json.load(open('result/otslhtml1651_quick_match_display_formula_per_sample_CDM.json'))
for x in r:
    x['_cdm'] = s.get(x['img_id'] + '_' + str(x.get('gt_idx', 0)))

def nested(t):
    return 'begin{array}' in t and ('\\[' in t or '\\(' in t)

# strip the math-mode delimiters the scorer failed to remove -> valid array
def unnest(t):
    t = t.replace('\\[', ' ').replace('\\]', ' ')
    return t.replace('\\(', ' ').replace('\\)', ' ')

cases = [x for x in r if x['_cdm'] is not None and nested(x['pred']) and not nested(x['gt'])]
print(f'cases (pred nested, gt clean): {len(cases)}', flush=True)

cdm = CDM(output_root=tempfile.mkdtemp())
out = []
for i, x in enumerate(cases):
    try:
        fixed = cdm.evaluate(x['gt'], unnest(x['pred']), f'ab{i}')['F1_score']
    except Exception as e:
        fixed = 0.0
    out.append({'img': x['img_id'], 'as_is': x['_cdm'], 'unnested': fixed})
    if i % 25 == 0:
        print(f'  {i}/{len(cases)}', flush=True)

json.dump(out, open('/tmp/nest_ab.json', 'w'), indent=1)
a = sum(o['as_is'] for o in out) / len(out)
b = sum(o['unnested'] for o in out) / len(out)
print(f'\nn={len(out)}')
print(f'  as-is (nested \\[ in array) : {a:.4f}')
print(f'  un-nested (valid array)    : {b:.4f}')
print(f'  delta                      : {b-a:+.4f}')
