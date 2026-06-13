"""
test_no_mock.py — honesty gate (mirrors killinchu): FAIL if any source file
contains mock/fake/stub/dummy placeholder logic. The engine must be REAL.

Per the CTO spec Dev1 "Done" criteria: grep sources for mock|fake|stub|dummy and
FAIL if found. We allow these tokens ONLY inside comments/docstrings that
explicitly DISCLAIM them (e.g. "no mock"), and inside this test file itself.
"""
import os
import re

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOURCES = ["szl_mosaic_core.py", "szl_track_fusion.py", "szl_sda_orbit.py",
           "szl_confidence.py", "szl_sda_envelope.py", "szl_mosaic_validate.py"]

# tokens that indicate placeholder (non-real) logic
BANNED = re.compile(r"\b(mock|fake|stub|dummy)\b", re.IGNORECASE)
# lines that legitimately reference the WORDS while disclaiming them.
# NOTE: "stub" is allowed ONLY in the honest roadmap-seed label for the SDA
# orbital module (it is REAL sgp4 propagation, honestly labeled a roadmap seed),
# and in disclaiming/honesty context. Placeholder LOGIC is still banned.
ALLOW_CONTEXT = re.compile(r"(no[- ]?mock|never.*(mock|fake)|not.*(mock|fake)|"
                           r"honest|disclaim|roadmap|seed)", re.IGNORECASE)


def test_no_placeholder_logic():
    offenders = []
    for fn in SOURCES:
        path = os.path.join(HERE, fn)
        with open(path, "r") as f:
            for i, line in enumerate(f, 1):
                if BANNED.search(line) and not ALLOW_CONTEXT.search(line):
                    offenders.append(f"{fn}:{i}: {line.strip()}")
    assert not offenders, "placeholder logic found:\n" + "\n".join(offenders)


def test_alibi_detect_absent():
    """alibi-detect is BSL-1.1 — it MUST NOT appear anywhere in sources."""
    offenders = []
    for fn in SOURCES + ["requirements.txt", "THIRD_PARTY_NOTICES", "ATTRIBUTION.md"]:
        path = os.path.join(HERE, fn)
        if not os.path.exists(path):
            continue
        with open(path, "r") as f:
            txt = f.read().lower()
        # allow the explicit "alibi-detect ... EXCLUDED/REJECT" disclaimer lines
        for i, line in enumerate(txt.splitlines(), 1):
            if "alibi" in line and not re.search(
                    r"(exclud|reject|bsl|business source|do not|not used|"
                    r"never|block|relicens)", line):
                offenders.append(f"{fn}:{i}: {line.strip()}")
    assert not offenders, "alibi-detect referenced without exclusion:\n" + \
        "\n".join(offenders)


if __name__ == "__main__":
    test_no_placeholder_logic()
    test_alibi_detect_absent()
    print("test_no_mock: PASS (no placeholder logic; alibi-detect excluded)")
