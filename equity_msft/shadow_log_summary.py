import json 
overrides = 0 
total = 0 
with open("shadow_grok_log.jsonl") as f:
    for line in f: 
        entry = json.loads(line)
        if entry["deterministic_action"] != entry["grok_action"]:
            overrides += 1
        total += 1
print(f"Overrides: {overrides} / {total} ({overrides/total:.2%})")