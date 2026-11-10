"""One-off smoke check: read_map over every folder of a CustomLevels directory.

Run in isolated mode: .venv/Scripts/python -I scripts/smoke_read_maps.py <custom_levels_dir>
"""
import sys
from collections import Counter
from pathlib import Path

from bsmaker.formats.reader import read_map


def main(root: Path) -> None:
    maps = exceptions = n_problems = 0
    schemas: Counter[str] = Counter()
    for folder in sorted(p for p in root.iterdir() if p.is_dir()):
        maps += 1
        try:
            m = read_map(folder)
        except Exception as e:  # noqa: BLE001 - smoke check reports everything
            exceptions += 1
            print("EXC", folder.name, type(e).__name__, e)
            continue
        n_problems += len(m.problems)
        schemas.update(d.schema for d in m.difficulties)
    print(f"maps={maps} exceptions={exceptions} problems={n_problems}")
    print(f"difficulties={sum(schemas.values())} per_schema={dict(schemas)}")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
