"""Generate the frozen Python json.dumps float-formatting fixture.

Run from the repository root with:
  cd benchmarks/baseline && ~/.local/bin/uv run python ../../spikes/tokenizer-parity/scripts/generate_float_cases.py
"""
import json
import math
import random
from pathlib import Path

random.seed(0x1A2B3C4D)
values = [
    -0.0,
    0.0,
    float.fromhex("0x0.0000000000001p-1022"),
    -float.fromhex("0x0.0000000000001p-1022"),
    1.0,
    -1.0,
    1e-4,
    1e-5,
    1e16,
    1e21,
]
for _ in range(1990):
    kind = random.randrange(6)
    if kind == 0:
        value = math.ldexp(random.uniform(-1, 1), random.randrange(-1074, 1024))
    elif kind == 1:
        value = random.uniform(-1e-4, 1e-4)
    elif kind == 2:
        value = random.uniform(-1e16, 1e16)
    elif kind == 3:
        value = math.copysign(float(random.randrange(0, 10**12)), random.choice((-1.0, 1.0)))
    elif kind == 4:
        value = math.copysign(10.0 ** random.uniform(-320, 308), random.choice((-1.0, 1.0)))
    else:
        value = random.uniform(-1e300, 1e300)
    if math.isfinite(value):
        values.append(value)
while len(values) < 2000:
    values.append(float(len(values)))
values = values[:2000]
output = [
    {"value": value, "expected": json.dumps(value, ensure_ascii=False)} for value in values
]
out = Path(__file__).resolve().parents[1] / "tests/fixtures/python-floats.json"
out.write_text(json.dumps(output, ensure_ascii=False, separators=(",", ":")) + "\n")
print(f"wrote {len(output)} cases to {out}")
