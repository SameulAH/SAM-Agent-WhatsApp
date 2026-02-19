import json

data = json.load(open("experiments/EXP-001/results.json"))

print("\n" + "=" * 80)
print("EXP-001: Sample Results")
print("=" * 80 + "\n")

for i, r in enumerate(data[:5], 1):
    print(f"[{i}] {r['prompt_id']}")
    print(f"    Input:   {r['input']}")
    print(f"    Status:  {r['status']}")
    print(f"    Latency: {r['latency_ms']:.0f}ms")
    output = r['output'][:100] if r['output'] else "(empty)"
    print(f"    Output:  {output}...")
    print()
