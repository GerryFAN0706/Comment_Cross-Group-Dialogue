import os, yaml, pandas as pd

def run():
    # Verify data files present
    needed = ["data/users.json","data/posts.json","data/comments.json"]
    missing = [p for p in needed if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(f"Missing input files: {missing}")
    print("All input files present. Proceed to step01.")

if __name__ == "__main__":
    run()
