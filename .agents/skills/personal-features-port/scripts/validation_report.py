"""Report a personal-port validation run against the upstream baseline.

Reads the nextest JUnit results of every head shard and of any baseline shards that ran, folds new
baseline results into the per-crate baseline cache, and lists only the failures the branch adds:
tests that fail on the branch but pass, or do not exist, on the upstream tag. Upstream's own
failures in the CI environment are reported as a count only. Exits non-zero when the branch adds
failures or a head shard produced no results.
"""

import argparse
import json
from pathlib import Path
import re
import sys
import tarfile
import xml.etree.ElementTree as ET

SUMMARY_MESSAGE_CHARS = 300
DETAIL_MESSAGE_CHARS = 4000
COMPILE_ERROR_LINES = 200


def parse_junit(path):
    """Returns {test id: (passed, message)} for one nextest JUnit file."""
    results = {}
    for case in ET.parse(path).getroot().iter("testcase"):
        test_id = f"{case.get('classname')} {case.get('name')}"
        failure = next((child for child in case if child.tag in ("failure", "error")), None)
        if failure is None:
            results[test_id] = (True, "")
            continue
        text = "\n".join(
            part
            for part in [failure.get("message") or "", failure.text or ""]
            + [(child.text or "") for child in case if child.tag in ("system-out", "system-err")]
            if part
        )
        results[test_id] = (False, text)
    return results


def package_of(test_id):
    return test_id.split(" ", 1)[0].split("::", 1)[0]


def shard_results(root, prefix, shards):
    """Merges the results of all shard artifacts; returns None when any shard has no JUnit file."""
    results = {}
    for shard in range(1, shards + 1):
        junit = next((root / f"{prefix}-{shard}").glob("**/junit.xml"), None)
        if junit is None:
            return None
        results.update(parse_junit(junit))
    return results


def failure_headline(message):
    match = re.search(r"panicked at ([^\n]+)\n([^\n]*)", message)
    text = f"{match.group(1)}: {match.group(2)}" if match else message.strip().replace("\n", " ")
    return text[:SUMMARY_MESSAGE_CHARS]


def compile_errors(root, prefix, shards):
    lines = []
    for shard in range(1, shards + 1):
        log = root / f"{prefix}-{shard}" / "nextest.log"
        if not log.exists():
            continue
        content = log.read_text(encoding="utf-8", errors="replace").splitlines()
        for index, line in enumerate(content):
            if line.startswith("error"):
                lines.extend(content[index : index + 12])
        if lines:
            break
    return lines[:COMPILE_ERROR_LINES]


def pending_snapshots(root, prefix):
    names = set()
    for archive in root.glob(f"{prefix}-*/snapshots.tar.gz"):
        with tarfile.open(archive) as tar:
            names.update(member.name for member in tar.getmembers() if member.isfile())
    return names


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True, help="directory with head-N and baseline-N")
    parser.add_argument("--shards", type=int, required=True)
    parser.add_argument("--baseline-dir", required=True, help="per-crate upstream results cache")
    parser.add_argument("--baseline-packages", default="", help="crates the baseline ran for")
    parser.add_argument("--packages", default="", help="crates the head run tested")
    parser.add_argument("--out", required=True, help="directory for report.md and details")
    args = parser.parse_args()

    results_root = Path(args.results)
    baseline_dir = Path(args.baseline_dir)
    baseline_dir.mkdir(parents=True, exist_ok=True)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    baseline_packages = args.baseline_packages.split()
    if baseline_packages:
        fresh = shard_results(results_root, "baseline", args.shards)
        if fresh is None:
            print("upstream baseline run is incomplete; its results are not cached", file=sys.stderr)
        else:
            for package in baseline_packages:
                status = {
                    test_id: "pass" if passed else "fail"
                    for test_id, (passed, _) in fresh.items()
                    if package_of(test_id) == package
                }
                (baseline_dir / f"{package}.json").write_text(json.dumps(status), encoding="utf-8")
            snapshot_file = baseline_dir / "snapshots.json"
            known_snapshots = set(json.loads(snapshot_file.read_text(encoding="utf-8"))) if snapshot_file.exists() else set()
            known_snapshots |= pending_snapshots(results_root, "baseline")
            snapshot_file.write_text(json.dumps(sorted(known_snapshots)), encoding="utf-8")

    baseline = {}
    for package in args.packages.split():
        path = baseline_dir / f"{package}.json"
        if path.exists():
            baseline[package] = json.loads(path.read_text(encoding="utf-8"))

    report = ["# Personal port validation", ""]
    head = shard_results(results_root, "head", args.shards)
    if head is None:
        report += [
            "A test shard produced no results; the first compile errors follow.",
            "",
            "```",
            *compile_errors(results_root, "head", args.shards),
            "```",
        ]
        (out / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
        print("\n".join(report))
        return 1

    new_failures, known = [], 0
    for test_id, (passed, message) in sorted(head.items()):
        if passed:
            continue
        package = package_of(test_id)
        upstream = baseline.get(package, {}).get(test_id) if package in baseline else None
        if upstream == "fail":
            known += 1
            continue
        if package not in baseline:
            reason = "crate has no upstream baseline"
        elif upstream is None:
            reason = "test does not exist upstream"
        else:
            reason = "passes upstream"
        new_failures.append((test_id, reason, message))

    snapshot_file = baseline_dir / "snapshots.json"
    upstream_snapshots = set(json.loads(snapshot_file.read_text(encoding="utf-8"))) if snapshot_file.exists() else set()
    snapshots = sorted(pending_snapshots(results_root, "head") - upstream_snapshots)
    report += [
        f"- Tests run: {len(head)} in `{args.packages}`",
        f"- Failures that upstream also has in CI: {known}",
        f"- Failures this branch adds: {len(new_failures)}",
        f"- Pending snapshot updates this branch adds: {len(snapshots)}",
        "",
    ]
    if new_failures:
        report += ["## Failures this branch adds", ""]
        for test_id, reason, message in new_failures:
            report.append(f"- `{test_id}` ({reason}): {failure_headline(message)}")
        report.append("")
    if snapshots:
        report += ["## Pending snapshots", ""] + [f"- `{name}`" for name in snapshots] + [""]

    details = []
    for test_id, reason, message in new_failures:
        details += [f"===== {test_id} ({reason})", message[:DETAIL_MESSAGE_CHARS], ""]
    (out / "new-failures.txt").write_text("\n".join(details), encoding="utf-8")
    (out / "report.md").write_text("\n".join(report), encoding="utf-8")
    print("\n".join(report))
    return 1 if new_failures else 0


if __name__ == "__main__":
    sys.exit(main())
