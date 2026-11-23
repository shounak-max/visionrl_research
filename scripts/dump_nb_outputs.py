import json
from pathlib import Path

nb_path = Path("notebooks/notebook_revised.ipynb")
with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

with open("notebook_all_outputs.txt", "w", encoding="utf-8") as out:
    for i, cell in enumerate(nb['cells']):
        out.write(f"\n{'='*80}\nCELL {i} ({cell.get('cell_type')})\n{'='*80}\n")
        # print first few lines of source
        src_lines = "".join(cell.get("source", [])).split("\n")
        out.write("SOURCE PREVIEW:\n")
        for sl in src_lines[:5]:
            out.write(f"  {sl}\n")
        out.write("OUTPUTS:\n")
        for o in cell.get("outputs", []):
            if "text" in o:
                out.write("".join(o["text"]))
            elif "data" in o:
                for k, v in o["data"].items():
                    if k == "text/plain":
                        out.write("".join(v))
                    elif k.startswith("image/"):
                        out.write(f"\n[IMAGE {k} ({len(''.join(v))} bytes)]\n")

print("Wrote notebook_all_outputs.txt")
