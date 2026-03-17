import pandas as pd
import sys
df = pd.read_csv("pairwise_compare.csv")
prec_pairs = [("float32", "float16"), ("float32", "bfloat16"), ("float16", "bfloat16")]
valid_pairs = []
for a, b in prec_pairs:
    valid_pairs.append((a, b))
    valid_pairs.append((b, a))

mask = df.apply(lambda row: (row['config_A'], row['config_B']) in valid_pairs, axis=1)
prec_df = df[mask].copy()

prec_df['sequence_mismatch_rate'] = (1.0 - prec_df['exact_match']) * 100

def get_color_pair(row):
    mapping = {"float32": "FP32", "float16": "FP16", "bfloat16": "BF16"}
    a, b = row['config_A'], row['config_B']
    if mapping[a] == "FP32" and mapping[b] == "FP16" or mapping[b] == "FP32" and mapping[a] == "FP16":
        return "FP32 vs FP16"
    if mapping[a] == "FP32" and mapping[b] == "BF16" or mapping[b] == "FP32" and mapping[a] == "BF16":
        return "FP32 vs BF16"
    return "FP16 vs BF16"

prec_df['pair'] = prec_df.apply(get_color_pair, axis=1)

with open("out_p2.txt", "w") as f:
    f.write("--- group by pair ---\n")
    f.write(str(prec_df.groupby(['benchmark', 'pair', 'model'])['sequence_mismatch_rate'].mean()))
    f.write("\n--- max sequence mismatch rate ---\n")
    f.write(str(prec_df['sequence_mismatch_rate'].max()))
