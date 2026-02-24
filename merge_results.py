#!/usr/bin/env python3
"""
merge_results.py — run locally after both instances finish

Usage:
  python3 merge_results.py

Expects:
  ./results_TextFooler/paper_numbers.json
  ./results_DeepWordBug/paper_numbers.json

Produces:
  ./paper_numbers_merged.json
"""

import json, datetime
from pathlib import Path

ATTACKS = ['TextFooler', 'DeepWordBug']

print('Loading results...')
all_pn = {}
for atk in ATTACKS:
    p = Path(f'results_{atk}') / 'paper_numbers.json'
    if not p.exists():
        print(f'  MISSING: {p}')
        continue
    with open(p) as f:
        all_pn[atk] = json.load(f)
    kc = all_pn[atk]['key_claims']
    print(f"  {atk:12}: best_layer=L{kc['best_layer']}  "
          f"AUC={kc['best_auc']}  N={all_pn[atk]['metadata']['n_pairs']}")

if len(all_pn) < 2:
    print('\nNeed both results dirs. Run the missing instance first.')
    raise SystemExit(1)

base = list(all_pn.values())[0]

best_layers = {a: pn['key_claims']['best_layer'] for a, pn in all_pn.items()}
best_aucs   = {a: pn['key_claims']['best_auc']   for a, pn in all_pn.items()}
consistent  = len(set(best_layers.values())) == 1

if consistent:
    L = list(best_layers.values())[0]
    auc_str = ', '.join(f'{a} AUC={v}' for a, v in best_aucs.items())
    claim = (f'The Layer-{L} conditioning signal replicates across '
             f'word-level and character-level attacks ({auc_str}), '
             f'consistent with a decision-boundary geometry mechanism '
             f'independent of attack strategy.')
else:
    claim = ('Best detection layer differs across attacks: '
             + ', '.join(f'{a}=L{L}' for a, L in best_layers.items())
             + '. The conditioning signal may be attack-strategy-specific.')

merged = {
    'metadata': {
        **base['metadata'],
        'generated_at'    : datetime.datetime.now().isoformat(),
        'attacks_merged'  : list(all_pn.keys()),
        'notebook_version': 'v3-merged',
    },
    'table_1': {
        'original'  : base['table_1']['original'],
        'replicated': [{**pn['table_1']['replicated'], 'attack': a}
                       for a, pn in all_pn.items()],
    },
    'detection_results'  : {a: pn['detection_results']          for a, pn in all_pn.items()},
    'ablation_base_model': {a: pn.get('ablation_base_model', {}) for a, pn in all_pn.items()},
    'cosine_baseline'    : {a: pn['cosine_baseline']             for a, pn in all_pn.items()},
    'key_claims'         : {a: pn['key_claims']                  for a, pn in all_pn.items()},
    'figures'            : {a: pn['figures']                     for a, pn in all_pn.items()},
    'cross_attack_summary': {
        'best_layers'          : best_layers,
        'best_aucs'            : best_aucs,
        'best_layer_consistent': consistent,
        'cross_attack_claim'   : claim,
    },
}

out = Path('paper_numbers_merged.json')
with open(out, 'w') as f:
    json.dump(merged, f, indent=2)

print(f'\nMerged -> {out}')
print(f'Layer consistent across attacks: {consistent}')
print(f'\nCross-attack claim:')
print(f'  {claim}')
print(f'\nClaude Code usage:')
print(f'  pn = json.load(open("paper_numbers_merged.json"))')
print(f'  for row in pn["table_1"]["replicated"]: ...')
print(f'  pn["cross_attack_summary"]["cross_attack_claim"]')
