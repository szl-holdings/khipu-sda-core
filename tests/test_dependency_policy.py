"""The packaging migration must keep the existing license admission control."""

import importlib.util
import json
from pathlib import Path

import pytest

# This is a repository policy tool, not an installed runtime module. Load its
# exact source file without depending on pytest's launch mode or changing sys.path.
_POLICY = Path(__file__).resolve().parents[1] / "tools" / "check_dependency_policy.py"
_SPEC = importlib.util.spec_from_file_location("sda_dependency_policy", _POLICY)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
forbidden_dependencies = _MODULE.forbidden_dependencies


@pytest.mark.parametrize("requirement", [
    "alibi-detect", "alibi_detect>=1", "ALIBI.DETECT[extra]==1",  # Forbidden negative fixtures.
    "Alibi__Detect; python_version < '3.0'",
    "alibi-detect @ https://example.invalid/package.whl",
])
@pytest.mark.parametrize("group", ["base", "optional"])
def test_forbidden_name_is_rejected_in_every_dependency_group(tmp_path, requirement, group):
    values = json.dumps([requirement])
    content = (f"[project]\ndependencies = {values}\n" if group == "base" else
               f"[project]\ndependencies = []\n[project.optional-dependencies]\nfeature = {values}\n")
    path = tmp_path / "pyproject.toml"
    path.write_text(content)
    assert forbidden_dependencies(path)


def test_comments_and_other_project_names_are_not_dependencies(tmp_path):
    path = tmp_path / "pyproject.toml"
    path.write_text('# alibi-detect is forbidden\n[project]\n'
                    'dependencies = ["numpy>=2", "alibi-detect-helper"]\n')
    assert forbidden_dependencies(path) == []


@pytest.mark.parametrize("content", [
    '[project]\ndependencies = "numpy"\n',
    '[project]\ndependencies = [123]\n',
    '[project]\ndependencies = ["not a requirement!"]\n',
    '[project]\ndynamic = ["dependencies"]\n',
    '[project]\ndynamic = ["optional-dependencies"]\n',
])
def test_uninspectable_metadata_does_not_pass(tmp_path, content):
    path = tmp_path / "pyproject.toml"
    path.write_text(content)
    with pytest.raises(ValueError):
        forbidden_dependencies(path)


def test_current_dependency_contract_passes():
    assert forbidden_dependencies(Path(__file__).resolve().parents[1] / "pyproject.toml") == []
