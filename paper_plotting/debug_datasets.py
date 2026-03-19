from datasets import load_dataset

def check_dataset(name, split, subset=None):
    print(f"Checking {name} ({subset}) split {split}...")
    try:
        if subset:
            ds = load_dataset(name, subset, split=split, trust_remote_code=True)
        else:
            ds = load_dataset(name, split=split, trust_remote_code=True)
        print(f"Loaded. Size: {len(ds)}")
        print("First item:", ds[0])
    except Exception as e:
        print(f"Failed: {e}")

check_dataset("gsm8k", "test", "main")
check_dataset("ybisk/piqa", "validation")
