import argparse
import json
import time
import torch
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from datasets import load_dataset
from tqdm import tqdm
import random
import math

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# --- Backend Interfaces ---

class Backend(ABC):
    @abstractmethod
    def get_choice_probs(self, prompt: str, choices: List[str]) -> Dict[str, float]:
        """
        Given a prompt and a list of choice tokens (e.g., ["A", "B", "C", "D"]),
        return a dictionary mapping choice -> prob.
        """
        pass

    @abstractmethod
    def unload(self):
        pass

    def compute_perplexity(self, text_list: List[str]) -> Dict[str, float]:
        """
        Compute perplexity for a list of text strings.
        Returns a dict with 'perplexity', 'avg_loss', etc.
        """
        raise NotImplementedError("Perplexity not implemented for this backend.")

class HFBackend(Backend):
    def __init__(self, model_name: str, device: str = "cuda", dtype=torch.float16, max_memory: Dict[int, str] = None):
        from transformers import AutoTokenizer, AutoModelForCausalLM
        print(f"[HF] Loading {model_name} on {device} ({dtype})...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        # Use device_map="auto" for efficient memory usage (offloading)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            torch_dtype=dtype, 
            trust_remote_code=True,
            device_map="auto",
            max_memory=max_memory
        )
        self.model.eval()
        self.device = self.model.device

    @torch.inference_mode()
    def get_choice_probs(self, prompt: str, choices: List[str]) -> Dict[str, float]:
        # Simple implementation
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        last_logits = outputs.logits[0, -1, :] # [Vocab]
        
        choice_logits = []
        valid_choices = []
        
        for c in choices:
            token_ids = self.tokenizer.encode(c, add_special_tokens=False)
            if not token_ids: continue
            target_id = token_ids[-1]
            valid_choices.append(c)
            choice_logits.append(last_logits[target_id].item())
            
        choice_logits = torch.tensor(choice_logits)
        probs = torch.softmax(choice_logits, dim=0).tolist()
        
        return dict(zip(valid_choices, probs))



    @torch.inference_mode()
    def compute_perplexity(self, text_list: List[str]) -> Dict[str, float]:
        # Sliding window strategy for long texts
        from torch.nn import CrossEntropyLoss
        nlls = []
        token_count = 0
        
        max_length = min(self.model.config.max_position_embeddings, 2048)
        stride = 512
        
        # We treat the list as one long sequence or separate? 
        # Usually PPL is over the entire concatenated text or independent segments.
        # For simplicity/standard wikitext, we process each item (article) independently or concat?
        # Standard: Concat all tokens.
        
        encodings = self.tokenizer("\n\n".join(text_list), return_tensors="pt")
        seq_len = encodings.input_ids.size(1)
        
        print(f"[HF] Evaluating PPL on {seq_len} tokens...")
        
        prev_end_loc = 0
        for begin_loc in tqdm(range(0, seq_len, stride)):
            end_loc = min(begin_loc + max_length, seq_len)
            trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
            
            input_ids = encodings.input_ids[:, begin_loc:end_loc].to(self.device)
            target_ids = input_ids.clone()
            target_ids[:, :-trg_len] = -100 # Ignore context targets
            
            with torch.no_grad():
                outputs = self.model(input_ids, labels=target_ids)
                # loss is calculated on target_ids
                neg_log_likelihood = outputs.loss
            
            nlls.append(neg_log_likelihood * trg_len)
            
            prev_end_loc = end_loc
            if end_loc == seq_len:
                break
                
        total_nll = torch.stack(nlls).sum()
        ppl = torch.exp(total_nll / end_loc)
        
        return {
            "perplexity": ppl.item(),
            "avg_loss": (total_nll / end_loc).item(),
            "total_tokens": end_loc
        }

    def unload(self):
        del self.model
        del self.tokenizer
        torch.cuda.empty_cache()

class VLLMBackend(Backend):
    def __init__(self, model_name: str, dtype: str = "float16", gpu_memory_utilization: float = 0.9):
        from vllm import LLM, SamplingParams
        print(f"[vLLM] Loading {model_name} (dtype={dtype})...")
        # Ensure we don't hog all memory if running other things
        self.llm = LLM(
            model=model_name, 
            dtype=dtype, 
            trust_remote_code=True,
            gpu_memory_utilization=gpu_memory_utilization,
            enforce_eager=True, # Sometimes helps with small batch stability
            max_model_len=4096, # Limit context length to save memory
            max_logprobs=1000 # Allow requesting up to 1000 logprobs
        )
        self.SamplingParams = SamplingParams

    def get_choice_probs(self, prompt: str, choices: List[str]) -> Dict[str, float]:
        # vLLM is batch-optimized, but for this interface we do one by one or need to accept batches.
        # For simplicity in this testbed, we do one by one.
        # We need logprobs for specific tokens.
        # Option 1: Generate 1 token with logprobs=TopK, hope choices are in TopK.
        # Option 2: Provide 'prompt_logprobs' if vLLM supports scoring mode (it does via 'score' mainly in newer versions, or just generate).
        
        # We will use standard generation with max_tokens=1 and logprobs.
        # Note: If choice is NOT in top-K logprobs, vLLM won't return it easily.
        # BUT, we can use `guided_choice` in newer vLLM, or just check top logprobs.
        
        # Let's request high number of logprobs to catch choices.
        # vLLM default max_logprobs is usually 20.
        params = self.SamplingParams(max_tokens=1, logprobs=1000, temperature=0.0)
        outputs = self.llm.generate([prompt], params, use_tqdm=False)
        
        # Extract logprobs from the first generated position
        # output[0].outputs[0].logprobs[0] is dict {token_id: Logprob}
        # Wait, vLLM returns top logprobs for the *generated* token position.
        
        # The prompt is already processed. We want the probability of the NEXT token being A, B, C...
        # vLLM `logprobs` in `outputs[0]` gives the logprobs for the *generated* token.
        # If the model predicts "A", we get "A"'s logprob. But we want "B"'s too.
        # If "B" is not the top token, we need `logprobs` parameter to be high enough.
        
        top_logprobs = outputs[0].outputs[0].logprobs[0] # Dict[int, Logprob]
        
        # We need tokenizer to map choice string -> id
        tokenizer = self.llm.get_tokenizer()
        
        choice_scores = []
        valid_choices = []
        
        for c in choices:
            token_ids = tokenizer.encode(c, add_special_tokens=False)
            if not token_ids: continue
            tid = token_ids[-1]
            
            # Check if tid is in top_logprobs
            if tid in top_logprobs:
                score = top_logprobs[tid].logprob
                choice_scores.append(score)
                valid_choices.append(c)
            else:
                # Missing from top-K. Assign very low logprob?
                # Or fail? For benchmarking, this is a risk with vLLM's standard API.
                # We will assign -9999.0
                choice_scores.append(-9999.0)
                valid_choices.append(c)
                
        # Softmax over collected logits/logprobs
        # Use log-sum-exp trick or just exp
        import math
        scores = np.array(choice_scores)
        exp_scores = np.exp(scores - np.max(scores))
        probs = (exp_scores / np.sum(exp_scores)).tolist()
        
        return dict(zip(valid_choices, probs))

    def compute_perplexity(self, text_list: List[str]) -> Dict[str, float]:
        # vLLM doesn't support sliding window PPL natively easily without orchestration.
        # But we can compute PPL on segments. `prompt_logprobs` gives logprobs for prompt.
        
        # We will iterate over texts.
        # Note: sliding window context management is hard via standard API if we want exact equivalence.
        # Approx: Evaluate each chunk independently or just use max_model_len context.
        
        total_logprob = 0.0
        total_tokens = 0
        
        # For very long text, we split it.
        full_text = "\n\n".join(text_list)
        # Tokenize explicitly to manage window
        tokenizer = self.llm.get_tokenizer()
        tokens = tokenizer.encode(full_text)
        
        print(f"[vLLM] Evaluating PPL on {len(tokens)} tokens...")
        
        # Chunking
        chunk_size = 3000 # Leave room for overhead, max 4096 usually safe-ish or read config
        # Actually Qwen3-8B has 32k context? 
        # We'll use a conservative chunk size.
        
        for i in tqdm(range(0, len(tokens), chunk_size)):
            chunk = tokens[i : i + chunk_size]
            if not chunk: continue
            
            # vLLM `generate` with `prompt_logprobs=1`
            # max_tokens=1 effectively just processes prompt
            
            prompt_token_ids = chunk
            
            # If we want PPL, we need logprob of every token.
            # prompt_logprobs=0 means no, 1 means ??? 
            # In vLLM: prompt_logprobs (int, optional) – If defined, return the log probabilities of the prompt tokens.
            
            params = self.SamplingParams(max_tokens=1, prompt_logprobs=1000, temperature=0.0)
            
            # Generating from token IDs
            # vLLM API: Pass list of dicts for token IDs
            outputs = self.llm.generate(prompts=[{"prompt_token_ids": prompt_token_ids}], sampling_params=params, use_tqdm=False)
            
            # outputs[0].prompt_logprobs is a list of dicts?
            # Or list of None (for first token) + dicts?
            plp = outputs[0].prompt_logprobs
            
            # First token usually has None logprob if it's start?
            # We assume independence for this simplified implementation: PPL of chunk.
            
            if plp:
                for j, token_probs in enumerate(plp):
                    if j == 0 and token_probs is None: continue # Skip first token (no context)
                    if token_probs:
                        # Extract logprob of the actual token
                        # prompt_logprobs[j] is {token_id: logprob, ...}
                        # We need the logprob of the token that IS at this position.
                        target_id = prompt_token_ids[j]
                        if target_id in token_probs:
                            total_logprob += token_probs[target_id].logprob
                            total_tokens += 1
        
        ppl = np.exp(-total_logprob / total_tokens) if total_tokens > 0 else 0.0
        return {
            "perplexity": ppl,
            "avg_loss": -total_logprob / total_tokens if total_tokens > 0 else 0.0,
            "total_tokens": total_tokens
        }

    def unload(self):
        # vLLM is hard to unload completely without killing process usually,
        # but we can try basic cleanup.
        import gc
        del self.llm
        gc.collect()
        torch.cuda.empty_cache()

class LlamaCppBackend(Backend):
    def __init__(self, model_path: str, n_gpu_layers: int = -1):
        try:
            from llama_cpp import Llama
        except ImportError:
            raise ImportError("llama-cpp-python is not installed.")
            
        print(f"[LlamaCpp] Loading {model_path}...")
        self.llm = Llama(
            model_path=model_path,
            n_gpu_layers=n_gpu_layers,
            verbose=False,
            logits_all=True
        )

    def get_choice_probs(self, prompt: str, choices: List[str]) -> Dict[str, float]:
        input_ids = self.llm.tokenize(prompt.encode("utf-8"))
        self.llm.reset()
        self.llm.eval(input_ids)
        logits = np.array(self.llm.scores[-1]) 
        
        choice_logits = []
        valid_choices = []
        for c in choices:
            token_ids = self.llm.tokenize(c.encode("utf-8"), add_bos=False)
            if not token_ids: continue
            target_id = token_ids[-1]
            valid_choices.append(c)
            choice_logits.append(logits[target_id])
            
        choice_logits = np.array(choice_logits)
        exp_logits = np.exp(choice_logits - np.max(choice_logits))
        probs = (exp_logits / np.sum(exp_logits)).tolist()
        
        return dict(zip(valid_choices, probs))

    def compute_perplexity(self, text_list: List[str]) -> Dict[str, float]:
        text = "\n\n".join(text_list)
        tokens = self.llm.tokenize(text.encode("utf-8"))
        n_ctx = self.llm.n_ctx()
        
        print(f"[LlamaCpp] Evaluating PPL on {len(tokens)} tokens (n_ctx={n_ctx})...")
        
        nlls = []
        
        # Process in disjoint chunks for simplicity
        # Note: This loses context between chunks, so PPL might be slightly higher than sliding window.
        for i in tqdm(range(0, len(tokens), n_ctx)):
            chunk = tokens[i : i + n_ctx]
            if len(chunk) < 2: continue
            
            self.llm.reset()
            self.llm.eval(chunk)
            
            # logits: [n_tokens, n_vocab]
            # We need to access the raw logits. 
            # In llama-cpp-python, self.llm.scores is a list of pointers or list of lists?
            # It seems self.llm.scores is a property that returns a generator or list.
            # Let's verify. It returns a list of float arrays.
            
            logits = np.array(self.llm.scores)[:len(chunk)] # shape (len(chunk), n_vocab)
            
            # Targets are chunk[1:]
            # Predictions are logits[:-1]
            
            # We only care about predictions for the tokens we provided (except the last one which predicts the next unseen token)
            # Actually, logits[j] predicts chunk[j+1].
            
            # So we iterate j from 0 to len(chunk)-2
            
            shift_logits = logits[:-1]
            shift_labels = chunk[1:]
            
            # Compute CrossEntropy
            # We can use torch if available or numpy
            # Let's use torch since we imported it
            
            t_logits = torch.tensor(shift_logits)
            t_labels = torch.tensor(shift_labels, dtype=torch.long)
            
            loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
            loss = loss_fct(t_logits, t_labels)
            
            nlls.append(loss)
            
        if not nlls:
            return {"perplexity": float("nan"), "avg_loss": float("nan"), "total_tokens": len(tokens)}
            
        all_nlls = torch.cat(nlls)
        avg_loss = all_nlls.mean().item()
        ppl = math.exp(avg_loss)
        
        return {"perplexity": ppl, "avg_loss": avg_loss, "total_tokens": len(tokens)}

    def unload(self):
        del self.llm


# --- Dataset Loader ---
class DatasetLoader:
    def __init__(self, dataset_name: str, split: str = "validation", subset: str = None):
        self.dataset_name = dataset_name
        self.split = split
        self.subset = subset
        
    def load(self, limit: int = None):
        items = []
        
        if self.dataset_name in ["cqa", "commonsense_qa"]:
            print(f"Loading commonsense_qa ({self.split})...")
            # Load without trust_remote_code if possible, else fallback
            try:
                ds = load_dataset("commonsense_qa", split=self.split, trust_remote_code=True)
            except:
                ds = load_dataset("commonsense_qa", split=self.split)
                
            iterator = ds
            if limit: iterator = iterator.select(range(limit))
            
            for row in iterator:
                q_text = row["question"]
                labels = row["choices"]["label"] # ['A', 'B'...]
                texts = row["choices"]["text"]
                
                options_str = "\n".join([f"{l}. {t}" for l, t in zip(labels, texts)])
                prompt = f"Question: {q_text}\n{options_str}\nAnswer:"
                
                items.append({
                    "id": row["id"],
                    "prompt": prompt,
                    "choices": labels,
                    "label": row["answerKey"],
                    "raw": row
                })
        
        elif self.dataset_name == "wikitext":
             print(f"Loading wikitext-2-raw-v1 ({self.split})...")
             ds = load_dataset("wikitext", "wikitext-2-raw-v1", split=self.split)
             # Filter empty lines
             texts = [x["text"] for x in ds if x["text"].strip()]
             if limit: texts = texts[:limit]
             return texts # List[str]

        elif self.dataset_name == "cmmlu":
             subset = self.subset if self.subset else "agronomy"
             # Map split names: validation -> dev, test -> test
             file_split = "dev" if self.split == "validation" else self.split
             csv_path = f"cmmlu_data/{file_split}/{subset}.csv"
             
             print(f"Loading CMMLU subset: {subset} ({self.split}) from {csv_path}...")
             
             try:
                 import pandas as pd
                 df = pd.read_csv(csv_path)
                 if limit: df = df.head(limit)
                 
                 for i, row in df.iterrows():
                    prompt = f"Question: {row['Question']}\nA. {row['A']}\nB. {row['B']}\nC. {row['C']}\nD. {row['D']}\nAnswer:"
                    items.append({
                        "id": i,
                        "prompt": prompt,
                        "choices": ["A", "B", "C", "D"],
                        "label": row["Answer"]
                    })
             except Exception as e:
                 print(f"Failed to load CMMLU from {csv_path}: {e}")
                 # Fallback to empty items

        elif self.dataset_name == "gsm8k":
            print(f"Loading gsm8k ({self.split})...")
            ds = load_dataset("gsm8k", "main", split=self.split)
            if limit: ds = ds.select(range(limit))
            
            # For PPL, we just return texts
            texts = []
            for row in ds:
                text = f"Question: {row['question']}\nAnswer: {row['answer']}"
                texts.append(text)
            return texts

        elif self.dataset_name == "piqa":
            print(f"Loading piqa ({self.split}) from local files...")
            # Assuming files are in piqa_data/physicaliqa-train-dev/
            base_path = "piqa_data/physicaliqa-train-dev"
            jsonl_path = f"{base_path}/dev.jsonl"
            labels_path = f"{base_path}/dev-labels.lst"
            
            try:
                import json
                with open(jsonl_path, 'r') as f:
                    lines = f.readlines()
                with open(labels_path, 'r') as f:
                    labels = f.readlines()
                
                if limit:
                    lines = lines[:limit]
                    labels = labels[:limit]
                
                for i, (line, label) in enumerate(zip(lines, labels)):
                    data = json.loads(line)
                    # Label is 0 or 1 in file? Check debug output or assume 0/1
                    # Usually labels are integers in .lst
                    label_idx = int(label.strip())
                    
                    prompt = f"Goal: {data['goal']}\nSol1: {data['sol1']}\nSol2: {data['sol2']}\nAnswer:"
                    items.append({
                        "id": i,
                        "prompt": prompt,
                        "choices": ["Sol1", "Sol2"], # We can map 0->Sol1, 1->Sol2
                        "label": "Sol1" if label_idx == 0 else "Sol2"
                    })
            except Exception as e:
                 print(f"Failed to load PIQA: {e}")

        return items

# --- Main ---
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["hf", "vllm", "llama_cpp"], required=True)
    parser.add_argument("--model", required=True, help="HF model ID or path")
    parser.add_argument("--dataset", choices=["cmmlu", "cqa", "commonsense_qa", "wikitext", "gsm8k", "piqa"], required=True)
    parser.add_argument("--dataset_subset", type=str, default=None, help="Subset for datasets like cmmlu")
    parser.add_argument("--task", type=str, choices=["qa", "perplexity"], default="qa")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", default="benchmark_results.jsonl")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--n_gpu_layers", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=123, help="Random seed")
    parser.add_argument("--tf32", action="store_true", help="Enable TF32")
    parser.add_argument("--max_gpu_memory", type=str, default=None, help="Max GPU memory for HF (e.g. '6GiB')")
    args = parser.parse_args()

    # Explicit Cleanup at Start
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    # Set Seed
    set_seed(args.seed)


    # Set TF32
    if args.tf32:
        print("Enabling TF32...")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    else:
        print("Disabling TF32...")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    # Init Backend
    if args.backend == "hf":
        dt = getattr(torch, args.dtype) if hasattr(torch, args.dtype) else torch.float16
        max_mem = {0: args.max_gpu_memory} if args.max_gpu_memory else None
        backend = HFBackend(args.model, device=args.device, dtype=dt, max_memory=max_mem)
    elif args.backend == "vllm":
        backend = VLLMBackend(args.model, dtype=args.dtype, gpu_memory_utilization=args.gpu_memory_utilization)
    elif args.backend == "llama_cpp":
        backend = LlamaCppBackend(args.model, n_gpu_layers=args.n_gpu_layers)
        
    # Load Dataset
    # Determine split based on dataset
    split = "validation"
    if args.dataset in ["cmmlu", "gsm8k"]: split = "test" # GSM8K usually uses test
    if args.dataset == "piqa": split = "validation" # We downloaded dev set for PIQA
    
    loader = DatasetLoader(args.dataset, split=split, subset=args.dataset_subset)
    items = loader.load(limit=args.limit)
    if not items:
        print("No items loaded. Exiting.")
        return
        
    print(f"Loaded {len(items)} examples.")
    
    # Run Eval
    start_time = time.time()
    
    if args.task == "perplexity":
        results = backend.compute_perplexity(items) # items is text list
        print(f"\n--- Perplexity Result ---")
        print(f"PPL: {results['perplexity']:.4f}")
        print(f"Loss: {results['avg_loss']:.4f}")
        print(f"Tokens: {results['total_tokens']}")
        
        # Save simple json
        with open(args.output, "w") as f:
            json.dump(results, f)
            
    else:
        results = []
        correct = 0
        
        # Write as we go
        with open(args.output, "w", encoding="utf-8") as f:
            for item in tqdm(items):
                try:
                    probs = backend.get_choice_probs(item["prompt"], item["choices"])
                    
                    # Predict
                    if probs:
                        pred = max(probs, key=probs.get)
                        is_correct = (pred == item["label"])
                        if is_correct: correct += 1
                    else:
                        pred = None
                        is_correct = False
                    
                    res_record = {
                        "id": item["id"],
                        "label": item["label"],
                        "prediction": pred,
                        "correct": is_correct,
                        "choice_probs": probs,
                        "dataset": args.dataset,
                        "backend": args.backend,
                        "model": args.model
                    }
                    f.write(json.dumps(res_record) + "\n")
                    f.flush()
                    results.append(res_record)
                except Exception as e:
                    print(f"Error processing item {item['id']}: {e}")
                
        acc = correct / len(items) if items else 0.0
        print(f"\n--- Benchmark Complete ---")
        print(f"Accuracy: {acc:.2%} ({correct}/{len(items)})")

    total_time = time.time() - start_time
    print(f"Time: {total_time:.2f}s")
    print(f"Dataset: {args.dataset}")
    print(f"Model: {args.model}")
    print(f"Saved to {args.output}")

    try:
        backend.unload()
    except:
        pass

if __name__ == "__main__":
    main()
