"""Exercise the actual workflow scanner without importing inspected fixtures."""

from pathlib import Path
import subprocess

import pytest


WORKFLOW = Path(__file__).resolve().parents[1] / ".github/workflows/ci.yml"
STEP = "      - name: Doctrine — forbid BSL alibi-detect dependency (license hygiene)"
PACKAGE = "alibi" + "_detect"


def source_gate_script():
    lines = WORKFLOW.read_text().splitlines()
    start = lines.index(STEP) + 1
    assert lines[start] == "        run: |"
    script = []
    for line in lines[start + 1:]:
        if line and not line.startswith("          "):
            break
        script.append(line[10:])
    assert script
    return "\n".join(script)


def run_source_gate(root):
    return subprocess.run(
        ["bash", "-eo", "pipefail", "-c", source_gate_script()],
        cwd=root, capture_output=True, text=True, timeout=10,
    )


@pytest.mark.parametrize("directory", [
    "ordinary", "not used", "excluded", "exclude", "reject", "forbidden",
    "absent", "must not", "BSL", "deliberately", "disclaimer",
])
@pytest.mark.parametrize("filename,content", [
    ("source.py", f"import {PACKAGE}\n"),
    ("requirements.txt", PACKAGE.replace("_", "-") + ">=1\n"),
])
def test_path_words_cannot_exempt_dependency_content(tmp_path, directory, filename, content):
    path = tmp_path / directory / filename
    path.parent.mkdir()
    path.write_text(content)
    result = run_source_gate(tmp_path)
    assert result.returncode == 1, result.stdout + result.stderr
    assert "::error::" in result.stdout
    assert content.strip() in result.stdout


@pytest.mark.parametrize("content", [
    f"# {PACKAGE} is deliberately excluded\n",
    f"# import {PACKAGE} is forbidden\n",
    "import pathlib\n",
    "numpy>=2\n",
])
def test_existing_content_exclusions_and_permitted_sources_remain_allowed(tmp_path, content):
    (tmp_path / "source.py").write_text(content)
    result = run_source_gate(tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "OK:" in result.stdout


def test_source_gate_accepts_empty_tree(tmp_path):
    result = run_source_gate(tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr


def test_filename_cannot_create_dependency_finding(tmp_path):
    (tmp_path / f"import {PACKAGE}.py").write_text("import pathlib\n")
    result = run_source_gate(tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
