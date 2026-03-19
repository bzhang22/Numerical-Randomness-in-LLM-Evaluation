import pandas as pd

def check_data():
    df = pd.read_csv('pairwise_compare.csv')
    
    with open('data_integrity_report.txt', 'w') as f:
        f.write("=== DATA INTEGRITY REPORT ===\n")
        
        # Check 1: Missing Benchmarks per Model
        f.write("\n--- Model Benchmark Coverage ---\n")
        coverage = df.groupby(['model', 'benchmark']).size().unstack(fill_value=0)
        f.write(coverage.to_string() + "\n")
        
        # Check 2: Missing Attention Comparisons
        f.write("\n--- Missing Attention Comparisons ---\n")
        attn_df = df[df['config_A'].isin(['eager', 'sdpa', 'flash_attention_2']) | df['config_B'].isin(['eager', 'sdpa', 'flash_attention_2'])]
        if not attn_df.empty:
            attn_coverage = attn_df.groupby(['model', 'config_A', 'config_B']).size()
            f.write(attn_coverage.to_string() + "\n")
        else:
            f.write("No attention data found.\n")
            
        # Check 3: Extreme Anomalies (0% exact match)
        f.write("\n--- Exact Match Rates (Warning if 0%) ---\n")
        match_rates = df.groupby(['model', 'benchmark', 'config_A', 'config_B'])['exact_match'].mean() * 100
        anomalies = match_rates[match_rates < 1.0]
        if not anomalies.empty:
            f.write("CRITICAL: Found comparisons with <1% exact match rate!\n")
            f.write(anomalies.to_string() + "\n")
        else:
            f.write("No extreme 0% failures found.\n")

if __name__ == '__main__':
    check_data()
