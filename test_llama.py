from llama_cpp import Llama
print("Import successful")
try:
    # Just try to initialize with dummy path to see if library loads (it will fail on path but that's fine)
    # Or better, check if we can instantiate if we had a model.
    # We can just print the version or something.
    import llama_cpp
    print(f"llama_cpp path: {llama_cpp.__file__}")
except Exception as e:
    print(f"Error: {e}")
