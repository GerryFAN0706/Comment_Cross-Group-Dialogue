import argparse
import json
import random
from pathlib import Path

from ..utils.io_utils import iter_json_like, ensure_dir


def sample_file(src: Path, dest: Path, ratio: float, seed: int) -> tuple[int, int]:
    rng = random.Random(seed)
    ensure_dir(dest.parent.as_posix())
    total = 0
    kept = 0
    with dest.open("w", encoding="utf-8") as out:
        out.write("[\n")
        first = True
        for obj in iter_json_like(src.as_posix()):
            total += 1
            if rng.random() <= ratio:
                if not first:
                    out.write(",\n")
                json.dump(obj, out, ensure_ascii=False)
                first = False
                kept += 1
        out.write("\n]\n")
    return total, kept


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Sample JSON datasets for quick pipeline runs.")
    parser.add_argument("inputs", nargs="+", help="Paths to JSON or JSONL files to sample.")
    parser.add_argument("--ratio", type=float, default=0.05, help="Sampling probability (0-1).")
    parser.add_argument("--seed", type=int, default=2025, help="Random seed for reproducibility.")
    parser.add_argument("--output-dir", type=Path, default=Path("data/sample"),
                        help="Directory to write sampled files into.")
    args = parser.parse_args(argv)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for inp in args.inputs:
        src = Path(inp)
        dest = args.output_dir / src.name
        total, kept = sample_file(src, dest, args.ratio, args.seed)
        print(f"{src.name}: kept {kept} of {total} (~{(kept/total*100 if total else 0):.2f}%) -> {dest}")


if __name__ == "__main__":
    main()
