import json
from pathlib import Path

nb_path = Path("notebooks/notebook_revised.ipynb")
with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

for idx in [39, 41, 47]:
    cell = nb['cells'][idx]
    print(f"=== CELL {idx} ===")
    for o in cell.get('outputs', []):
        if 'text' in o:
            print(''.join(o['text']))
        elif 'data' in o:
            for k, v in o['data'].items():
                if k == 'text/plain':
                    print(''.join(v))
