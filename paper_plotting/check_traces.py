import os
import json
from collections import defaultdict

trace_dir = "trace"
counts = defaultdict(lambda: {'total': 0, 'divergent': 0})

for root, dirs, files in os.walk(trace_dir):
    for f in files:
        if f.endswith('.json'):
            path = os.path.join(root, f)
            try:
                with open(path, 'r') as file:
                    data = json.load(file)
                meta = data.get('metadata', {})
                model = meta.get('model', 'Unknown')
                is_div = meta.get('is_divergent', False)
                counts[model]['total'] += 1
                if is_div:
                    counts[model]['divergent'] += 1
            except:
                pass

for k, v in counts.items():
    print(f"Model: {k} | Total Traces: {v['total']} | Divergent Traces: {v['divergent']}")
