
import json
import os
import glob
from tabulate import tabulate

def get_ppl(filepath):
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        # lm_eval structure: results['wikitext']['word_perplexity'] or similar
        # Check structure
        if 'results' in data and 'wikitext' in data['results']:
            # Handle potential key variations (e.g., with filter suffix)
            res = data['results']['wikitext']
            if 'word_perplexity,none' in res:
                return res['word_perplexity,none']
            return res.get('word_perplexity', 'N/A')
        return 'N/A'
    except Exception as e:
        return f"Error: {e}"

def main():
    base_dir = "diagnosis_8b"
    files = {
        "Baseline (Batch 1, Default)": "baseline_batch1.json",
        "Batch 4": "batch4.json",
        "Batch 8": "batch8.json",
        "Attn Eager (Math)": "attn_eager.json",
        "GEMM Deterministic": "gemm_deterministic.json"
    }

    results = []
    baseline_ppl = None

    # Load baseline first
    base_path = os.path.join(base_dir, files["Baseline (Batch 1, Default)"])
    if os.path.exists(base_path):
        baseline_ppl = get_ppl(base_path)
    
    headers = ["Configuration", "Perplexity", "Diff from Baseline", "Status"]

    for name, filename in files.items():
        filepath = os.path.join(base_dir, filename)
        if not os.path.exists(filepath):
            results.append([name, "Missing", "-", "Failed/Not Run"])
            continue
        
        ppl = get_ppl(filepath)
        status = "OK"
        diff = "-"
        
        if isinstance(ppl, float) and isinstance(baseline_ppl, float):
            if ppl == baseline_ppl:
                diff = "0.0 (Identical)"
                status = "Match"
            else:
                delta = ppl - baseline_ppl
                diff = f"{delta:+.6f}"
                status = "MISMATCH" if abs(delta) > 1e-6 else "Match (<1e-6)"
        
        results.append([name, ppl, diff, status])

    print("\n=== Diagnosis Analysis (Qwen3 8B) ===")
    print(tabulate(results, headers=headers, floatfmt=".6f"))

if __name__ == "__main__":
    main()
