from datasets import load_dataset
try:
    print("Attempting to load lmlmcat/cmmlu (agronomy)...")
    ds = load_dataset("lmlmcat/cmmlu", "agronomy", split="test")
    print(f"Loaded lmlmcat/cmmlu test split. Size: {len(ds)}")
    print("First item:", ds[0])
except Exception as e:
    print(f"Failed to load opencompass/cmmlu: {e}")
