import argparse
import json
import time
import torch
import numpy as np
from tqdm import tqdm
import os
import sys

# reuse the DatasetLoader from run_benchmark
from run_benchmark import DatasetLoader

def get_choice_probs(model, tokenizer, prompt, choices, device):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    last_logits = outputs.logits[0, -1, :]
    
    choice_logits = []
    valid_choices = []
    
    for c in choices:
        token_ids = tokenizer.encode(c, add_special_tokens=False)
        if not token_ids: continue
        target_id = token_ids[-1]
        valid_choices.append(c)
        choice_logits.append(last_logits[target_id].item())
        
    if not choice_logits:
        return {}
        
    choice_logits = torch.tensor(choice_logits)
    probs = torch.softmax(choice_logits, dim=0).tolist()
    return dict(zip(valid_choices, probs))

def apply_fp32_mitigation(model, variant, base_dtype, layer_split="all"):
    if variant in ["bf16_baseline", "fp16_baseline", "fp32_reference"]:
        return

    def pre_hook(module, args, kwargs):
        new_args = tuple(a.to(torch.float32) if isinstance(a, torch.Tensor) and a.is_floating_point() else a for a in args)
        new_kwargs = {k: v.to(torch.float32) if isinstance(v, torch.Tensor) and v.is_floating_point() else v for k, v in kwargs.items()}
        return new_args, new_kwargs

    def post_hook_base(module, args, output):
        def to_base(x):
            if isinstance(x, torch.Tensor) and x.is_floating_point():
                x.data = x.data.to(base_dtype)
                return x.to(base_dtype)
            elif isinstance(x, tuple):
                return tuple(to_base(i) for i in x)
            elif isinstance(x, list):
                return [to_base(i) for i in x]
            elif hasattr(x, 'key_cache'):
                for i in range(len(getattr(x, 'key_cache', []))):
                    if isinstance(x.key_cache[i], torch.Tensor):
                        x.key_cache[i] = x.key_cache[i].to(base_dtype)
                for i in range(len(getattr(x, 'value_cache', []))):
                    if isinstance(x.value_cache[i], torch.Tensor):
                        x.value_cache[i] = x.value_cache[i].to(base_dtype)
            return x
            
        res = to_base(output)
        return res

    patched_count = 0
    num_layers = getattr(model.config, "num_hidden_layers", 32)
    
    # Determine bounds for layer splits
    min_layer, max_layer = 0, num_layers
    if layer_split == "first_half":
        max_layer = num_layers // 2
    elif layer_split == "last_half":
        min_layer = num_layers // 2
    elif layer_split == "first_quarter":
        max_layer = num_layers // 4
    elif layer_split == "last_quarter":
        min_layer = num_layers - (num_layers // 4)
    elif layer_split == "middle":
        min_layer = num_layers // 4
        max_layer = num_layers - (num_layers // 4)

    for name, module in model.named_modules():
        patch = False
        is_lm_head = False
        
        name_lower = name.lower()
        if ("attention" in variant) and (name.endswith("self_attn") or name.endswith("attn")) and not "post_attention" in name_lower:
            patch = True
            
        if ("norm" in variant) and ("norm" in name_lower) and not ("attn" in name_lower):
            patch = True
            
        if ("lm_head" in variant) and (name == "lm_head"):
            patch = True
            is_lm_head = True
            
        # Check if the module belongs to a specific transformer layer and falls outside the split
        import re
        match = re.search(r'\.([0-9]+)\.', name)
        if patch and match and layer_split != "all":
            layer_idx = int(match.group(1))
            if layer_split in ["first_half", "last_half", "first_quarter", "last_quarter", "middle"]:
                if not (min_layer <= layer_idx < max_layer):
                    patch = False
            else:
                # Treat as comma-separated list of target indices (supports negative indexing)
                target_indices = [int(x.strip()) if int(x.strip()) >= 0 else num_layers + int(x.strip()) for x in layer_split.split(",")]
                if layer_idx not in target_indices:
                    patch = False
                
        if patch:
            print(f"Patching module to FP32 [{layer_split}]: {name}")
            if is_lm_head and hasattr(module, 'weight'):
                # Break weight-tying with input embeddings to prevent poisoning the root residual
                module.weight = torch.nn.Parameter(module.weight.clone().float())
            module.to(torch.float32)
            module.register_forward_pre_hook(pre_hook, with_kwargs=True)
            if not is_lm_head:
                module.register_forward_hook(post_hook_base)
            patched_count += 1
            
    print(f"Total modules patched to FP32: {patched_count}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=str)
    parser.add_argument("--dataset", required=True, type=str)
    parser.add_argument("--variant", required=True, choices=["bf16_baseline", "fp16_baseline", "fp32_reference", "attention", "norm", "lm_head", "attention_lm_head", "attention_norm", "attention_norm_lm_head"])
    parser.add_argument("--layer_split", default="all", type=str, help="Can be a split name or comma separated layer indices like '1' or '1,-1'")
    parser.add_argument("--dtype", default="bfloat16", type=str)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--output", required=True, type=str)
    args = parser.parse_args()

    # Determine load dtype
    load_dtype = torch.float32 if args.variant == "fp32_reference" else getattr(torch, args.dtype)

    # Load Model
    from transformers import AutoTokenizer, AutoModelForCausalLM
    print(f"Loading {args.model} in {load_dtype}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model, 
        torch_dtype=load_dtype, 
        trust_remote_code=True,
        device_map="auto"
    )
    model.eval()

    # Apply Patches if needed
    apply_fp32_mitigation(model, args.variant, load_dtype, layer_split=args.layer_split)

    # Dataset loader
    split = "validation"
    if args.dataset in ["cmmlu", "gsm8k", "humaneval"]: split = "test"
    if args.dataset == "piqa": split = "validation"
    
    loader = DatasetLoader(args.dataset, split=split)
    items = loader.load(limit=args.limit)
    if not items: 
        print("Dataset failed to load!")
        return

    print(f"Starting eval on {len(items)} items...")
    
    correct = 0
    results = []
    
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    
    with open(args.output, "w") as f:
        for item in tqdm(items):
            start_time = time.time()
            prompt = item["prompt"]
            
            probs = get_choice_probs(model, tokenizer, prompt, item.get("choices", []), model.device)
            if probs:
                pred = max(probs, key=probs.get)
                is_correct = (pred == item["label"])
                if is_correct: correct += 1
            else:
                pred = None
                is_correct = False
                
            max_new = 40 if args.dataset in ["gsm8k", "humaneval", "wikitext"] else 1
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                gen_outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new,
                    do_sample=False,
                    temperature=0.0,
                    pad_token_id=tokenizer.pad_token_id
                )
            gen_tokens = gen_outputs[0][inputs.input_ids.shape[1]:].tolist()
            gen_text = tokenizer.decode(gen_tokens, skip_special_tokens=True)
            
            latency = time.time() - start_time
            
            res = {
                "id": item["id"],
                "model": args.model,
                "dataset": args.dataset,
                "variant": args.variant,
                "layer_split": args.layer_split,
                "label": item.get("label", ""),
                "prediction": pred,
                "correct": is_correct,
                "generated_text": gen_text,
                "generated_tokens": gen_tokens,
                "latency_sec": latency
            }
            f.write(json.dumps(res) + "\n")
            f.flush()
            results.append(res)
            
    if torch.cuda.is_available():
        peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 3)
    else:
        peak_mem = 0.0
    print(f"Total Accuracy: {correct/len(items):.2%} ({correct}/{len(items)})")
    print(f"Peak GPU Memory: {peak_mem:.2f} GB")
    
    meta = {
        "model": args.model,
        "dataset": args.dataset,
        "variant": args.variant,
        "layer_split": args.layer_split,
        "accuracy": correct / len(items),
        "peak_memory_gb": peak_mem
    }
    with open(args.output.replace(".jsonl", "_meta.json"), "w") as f:
        json.dump(meta, f)

if __name__ == "__main__":
    main()
