from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "packages" / "dist"
NATIVE = ROOT / "packages" / "native"
# Seven setuptools packages. The nine release wheels come from three places:
#   1. the 7 packages listed here (core + six feature plugins)
#   2. + sglang_h3_native   (Rust core, built by maturin in main() below)
#   3. + sglang-h3-compat   (repo-root meta bundle, appended to targets in main())
PACKAGES = (
    "packages/core",
    "packages/request_fields",
    "packages/denoise",
    "packages/state_export",
    "packages/low_memory",
    "packages/nvenc",
    "packages/audio",
)


def cargo_environment() -> dict[str, str]:
    environment = os.environ.copy()
    cargo_home = Path.home() / ".cargo" / "bin"
    existing_path = environment.get("PATH", "")
    paths = [str(cargo_home), *existing_path.split(os.pathsep)]
    environment["PATH"] = os.pathsep.join(dict.fromkeys(paths))
    return environment


def run(
    command: list[str], *, cwd: Path, environment: dict[str, str] | None = None
) -> None:
    print(f"run {' '.join(command)} (cwd={cwd})")
    subprocess.run(
        command,
        cwd=cwd,
        env=environment,
        check=True,
    )


def main() -> int:
    environment = cargo_environment()
    if DIST.exists():
        shutil.rmtree(DIST)
    DIST.mkdir(parents=True)

    run(["cargo", "fmt", "--all", "--", "--check"], cwd=NATIVE, environment=environment)
    run(
        ["cargo", "clippy", "--all-targets", "--all-features", "--", "-D", "warnings"],
        cwd=NATIVE,
        environment=environment,
    )
    run(["cargo", "test"], cwd=NATIVE, environment=environment)
    run(
        [
            "uv",
            "tool",
            "run",
            "--from",
            "maturin[zig]",
            "maturin",
            "build",
            "--release",
            "--target",
            "x86_64-unknown-linux-gnu",
            "--compatibility",
            "manylinux_2_34",
            "--out",
            str(DIST),
        ],
        cwd=NATIVE,
        environment=environment,
    )

    targets = [ROOT / package for package in PACKAGES] + [ROOT]
    for target in targets:
        print(f"build {target.relative_to(ROOT)}")
        subprocess.run(
            ["uv", "build", "--wheel", "--out-dir", str(DIST), str(target)],
            cwd=ROOT,
            check=True,
        )

    wheels = sorted(path.name for path in DIST.glob("*.whl"))
    print(f"\nbuilt {len(wheels)} wheels into {DIST} "
          f"(7 setuptools + 1 native/maturin + 1 root meta bundle = 9):")
    for wheel in wheels:
        print(f"  {wheel}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except subprocess.CalledProcessError as exc:
        print(f"build failed with exit code {exc.returncode}", file=sys.stderr)
        raise SystemExit(exc.returncode)
