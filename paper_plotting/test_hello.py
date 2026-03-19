print("Starting test_hello.py imports...", flush=True)
try:
    import pandas
    print("pandas loaded", flush=True)
    import seaborn
    print("seaborn loaded", flush=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    print("matplotlib loaded", flush=True)
except Exception as e:
    print(f"Error: {e}")
print("Done!", flush=True)
