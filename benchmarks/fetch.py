"""Download the benchmark datasets that this repository does not redistribute.

Some of the datasets axiom is checked against carry licences that a proprietary
repository cannot satisfy by vendoring them -- CC BY-NC in one case, GPL-3 in
another. Rather than quietly copy them in, they are fetched on request, by you,
under the upstream terms, into a git-ignored cache.

    python benchmarks/fetch.py              # show what is available and why
    python benchmarks/fetch.py darfur       # fetch one
    python benchmarks/fetch.py --all        # fetch everything fetchable

Nothing else in the repository calls this automatically. Tests over fetched data
skip when it is absent.
"""

from __future__ import annotations

import sys
import urllib.request

from registry import CACHE, DATASETS, Dataset

TIMEOUT = 60


def describe() -> None:
    print("Vendored (already here, no action needed):\n")
    for d in DATASETS.values():
        if d.availability == "vendored":
            print(f"  {d.name:14s} {d.title}")
            print(f"  {'':14s} {d.licence}\n")

    print("Fetched on request:\n")
    for d in DATASETS.values():
        if d.availability != "fetch":
            continue
        state = "present" if d.available else "not fetched"
        print(f"  {d.name:14s} [{state}] {d.title}")
        print(f"  {'':14s} licence: {d.licence}")
        print(f"  {'':14s} from:    {d.url}")
        print(f"  {'':14s} cite:    {d.citation.split('.')[0]}.\n")

    print("Fetching a dataset means accepting its licence. Read the lines above.")


def fetch(spec: Dataset, force: bool = False) -> bool:
    if spec.availability != "fetch":
        print(f"{spec.name}: vendored, nothing to fetch")
        return True
    if spec.available and not force:
        print(f"{spec.name}: already at {spec.path} ({spec.path.stat().st_size:,} bytes)")
        return True

    CACHE.mkdir(parents=True, exist_ok=True)
    print(f"{spec.name}: downloading from {spec.url}")
    print(f"{spec.name}: licence -- {spec.licence}")
    request = urllib.request.Request(spec.url, headers={"User-Agent": "axiom-benchmarks"})
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT) as response:  # noqa: S310
            payload = response.read()
    except Exception as exc:  # noqa: BLE001 - report and carry on to the next one
        print(f"{spec.name}: FAILED -- {exc}")
        print(f"{spec.name}: fetch it by hand from {spec.source} and save to {spec.path}")
        return False

    spec.path.write_bytes(payload)
    print(f"{spec.name}: wrote {spec.path} ({len(payload):,} bytes)")
    print(f"{spec.name}: sha256 {spec.sha256()}")
    return True


def main(argv: list[str]) -> int:
    args = [a for a in argv if not a.startswith("-")]
    flags = {a for a in argv if a.startswith("-")}

    if not args and "--all" not in flags:
        describe()
        return 0

    if "--all" in flags:
        names = [n for n, d in DATASETS.items() if d.availability == "fetch"]
    else:
        names = args
    unknown = [n for n in names if n not in DATASETS]
    if unknown:
        print(f"unknown dataset(s): {unknown}; have {sorted(DATASETS)}")
        return 2

    ok = all(fetch(DATASETS[n], force="--force" in flags) for n in names)
    if ok:
        print("\nAll requested datasets are present. Run them with:")
        print("  python benchmarks/run_all.py")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
