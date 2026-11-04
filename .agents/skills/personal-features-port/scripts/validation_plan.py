"""Plan a personal-port validation run.

Selects the crates whose tests a validation branch must run and the crates that still lack an
upstream baseline. The selection is the crates the branch changes relative to the upstream tag,
plus the end-to-end crates (core, app-server, TUI, exec, CLI) whose dependency closure reaches a
changed crate, because those suites observe the changed behavior through production paths.
"""

import argparse
import json
from pathlib import Path
import subprocess

END_TO_END = ["codex-core", "codex-app-server", "codex-tui", "codex-exec", "codex-cli"]
# Binaries that integration tests spawn; built before the test run when the workspace has them.
HELPER_BINARIES = ["codex-cli", "codex-exec-server-cli", "codex-code-mode-host", "codex-rmcp-client"]


def metadata(workspace):
    output = subprocess.run(
        ["cargo", "+stable", "metadata", "--no-deps", "--format-version", "1"],
        cwd=workspace,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return json.loads(output)["packages"]


def owning_package(path, manifest_dirs):
    for directory, name in manifest_dirs:
        if directory == path or directory in path.parents:
            return name
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--head", required=True, help="codex-rs directory of the branch")
    parser.add_argument("--upstream", required=True, help="codex-rs directory of the upstream tag")
    parser.add_argument("--upstream-ref", required=True, help="git ref of the upstream tag")
    parser.add_argument("--baseline-dir", required=True, help="cached upstream results per crate")
    parser.add_argument("--github-output", help="file to append step outputs to")
    args = parser.parse_args()

    head_root = Path(args.head).resolve()
    packages = metadata(head_root)
    names = {package["name"] for package in packages}
    upstream_names = {package["name"] for package in metadata(args.upstream)}

    manifest_dirs = sorted(
        ((Path(package["manifest_path"]).resolve().parent, package["name"]) for package in packages),
        key=lambda entry: len(str(entry[0])),
        reverse=True,
    )
    changed = subprocess.run(
        ["git", "diff", "--name-only", "--relative", args.upstream_ref, "HEAD", "--", "."],
        cwd=head_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.split()
    touched = {
        name
        for name in (owning_package(head_root / path, manifest_dirs) for path in changed)
        if name is not None
    }

    dependencies = {
        package["name"]: {dep["name"] for dep in package["dependencies"] if dep["name"] in names}
        for package in packages
    }

    def closure(name):
        seen, stack = set(), [name]
        while stack:
            for dep in dependencies.get(stack.pop(), ()):
                if dep not in seen:
                    seen.add(dep)
                    stack.append(dep)
        return seen

    selected = set(touched)
    for name in END_TO_END:
        if name in names and (name in touched or closure(name) & touched):
            selected.add(name)
    selected = sorted(selected)

    baseline_dir = Path(args.baseline_dir)
    baseline_needed = [
        name
        for name in selected
        if name in upstream_names and not (baseline_dir / f"{name}.json").exists()
    ]

    outputs = {
        "packages": " ".join(selected),
        "package_args": " ".join(f"-p {name}" for name in selected),
        "helper_args": " ".join(f"-p {name}" for name in HELPER_BINARIES if name in names),
        "baseline_packages": " ".join(baseline_needed),
        "baseline_package_args": " ".join(f"-p {name}" for name in baseline_needed),
        "baseline_helper_args": " ".join(
            f"-p {name}" for name in HELPER_BINARIES if name in upstream_names
        ),
    }
    print(f"changed crates: {' '.join(sorted(touched)) or '(none)'}")
    print(f"tested crates: {outputs['packages'] or '(none)'}")
    print(f"crates without an upstream baseline yet: {outputs['baseline_packages'] or '(none)'}")
    if args.github_output:
        with open(args.github_output, "a", encoding="utf-8") as handle:
            for key, value in outputs.items():
                handle.write(f"{key}={value}\n")


if __name__ == "__main__":
    main()
