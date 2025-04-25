from configuration_evo2 import Evo2Config
import json, pathlib, inspect, itertools as it

cfg = Evo2Config.from_pretrained(pathlib.Path("."))      # loads your JSON

def walk(obj, prefix=""):
    if isinstance(obj, dict):
        for k, v in obj.items():  yield from walk(v, f"{prefix}.{k}" if prefix else k)
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj): yield from walk(v, f"{prefix}[{i}]")
    else:
        yield prefix, obj

bad = [(k, v) for k, v in walk(cfg.to_dict()) if isinstance(v, type)]
print("\n⚠️  Offending entries (expect zero):")
for k, v in bad: print(f"{k:>25}  ->  {v}")
