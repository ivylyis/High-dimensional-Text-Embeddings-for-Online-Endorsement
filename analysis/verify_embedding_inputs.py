"""Verify the three embedding files against their expected SHA-256 digests."""

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1] / "data" / "embeddings"


def main():
    expected = json.loads((ROOT / "expected_files.json").read_text())
    errors = []
    for name, info in expected.items():
        path = ROOT / name
        if not path.is_file() or path.stat().st_size != info["bytes"]:
            errors.append(f"{name}: missing, incomplete, or still a Git LFS pointer")
            continue
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() != info["sha256"]:
            errors.append(f"{name}: SHA-256 mismatch")
    if errors:
        raise SystemExit("\n".join(errors))
    print("All three embedding files match.")


if __name__ == "__main__":
    main()
