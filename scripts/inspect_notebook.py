import json
from pathlib import Path

nb_path = Path("notebooks/notebook_revised.ipynb")
with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

print(f"Total cells: {len(nb['cells'])}")
for i, cell in enumerate(nb["cells"]):
    src = "".join(cell.get("source", []))[:100].replace("\n", " ")
    outputs = cell.get("outputs", [])
    out_types = [o.get("output_type") for o in outputs]
    text_previews = []
    for o in outputs:
        if "text" in o:
            t = "".join(o["text"])[:80].replace("\n", " ")
            text_previews.append(t)
        elif "data" in o and "text/plain" in o["data"]:
            t = "".join(o["data"]["text/plain"])[:80].replace("\n", " ")
            text_previews.append(t)
    if outputs:
        print(f"Cell {i:02d}: {src}")
        if text_previews:
            print(f"   -> Out: {text_previews[:2]}")
