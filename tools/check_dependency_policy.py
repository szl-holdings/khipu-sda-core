"""Apply the existing forbidden-dependency policy to canonical wheel metadata."""

import argparse
from pathlib import Path
import sys
import tomllib

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name


def forbidden_dependencies(path: Path) -> list[str]:
    project = tomllib.loads(path.read_text())["project"]
    if {"dependencies", "optional-dependencies"} & set(project.get("dynamic", [])):
        raise ValueError("dependency policy requires static project dependencies")
    groups = {"dependencies": project.get("dependencies", [])}
    optional = project.get("optional-dependencies", {})
    if not isinstance(optional, dict):
        raise ValueError("optional-dependencies must be a table")
    groups.update({f"extra {name}": values for name, values in optional.items()})
    forbidden = []
    for group, requirements in groups.items():
        if not isinstance(requirements, list):
            raise ValueError(f"{group} must be a dependency array")
        for text in requirements:
            if not isinstance(text, str):
                raise ValueError(f"{group} contains a non-string requirement")
            if canonicalize_name(Requirement(text).name) == "alibi-detect":
                forbidden.append(f"{group}: {text}")
    return forbidden


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", type=Path, default=Path("pyproject.toml"))
    args = parser.parse_args()
    try:
        forbidden = forbidden_dependencies(args.path)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"dependency policy could not be evaluated: {exc}", file=sys.stderr)
        return 2
    if forbidden:
        print("forbidden alibi-detect dependency: " + "; ".join(forbidden), file=sys.stderr)
        return 1
    print("OK: canonical package dependencies exclude alibi-detect")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
