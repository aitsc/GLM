import os
import json
from pprint import pprint

path = 'data/english_data/superglue/'
for ds in ['BoolQ', 'CB', 'COPA', 'MultiRC', 'ReCoRD', 'RTE', 'WiC', 'WSC']:
    print(ds)
    label = set()
    with open(os.path.join(path, ds, 'train.jsonl'), 'r', encoding='utf8') as r:
        all_json = []
        for line in r:
            if line.strip():
                j = json.loads(line)
                if 'label' in j:
                    label.add(j['label'])
                all_json.append(j)
    print(list(all_json[0].keys()))
    pprint(all_json[0])
    print(label)
    print()