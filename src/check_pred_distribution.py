# check_pred_distribution.py
import json, sys
from collections import Counter
cnt = Counter(); N = 0
for line in open(sys.argv[1], 'r', encoding='utf-8'):
    o = json.loads(line)
    pred = (o.get('pred','') or '').strip()
    N += 1
    if pred == '':
        cnt['EMPTY'] += 1
    elif pred.upper() == 'CANNOTANSWER':
        cnt['CANNOTANSWER'] += 1
    else:
        cnt['NON_EMPTY'] += 1
print('Total:', N)
for k, v in cnt.items():
    print(k, v, f'({v/N:.1%})')
