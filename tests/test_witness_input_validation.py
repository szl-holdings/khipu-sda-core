"""Invalid evidence cannot produce an affirmative advisory quorum receipt."""

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from szl_witness import WitnessOpinion, verdict_band, witness_envelope, witness_quorum


INVALID_SCORES = [math.nan, math.inf, -math.inf, -0.01, 1.01, True, "0.2"]


@pytest.mark.parametrize("score", INVALID_SCORES)
def test_invalid_base_score_is_rejected(score):
    with pytest.raises(ValueError, match="base_score"):
        witness_quorum("invalid-base", score, np.array([0.2]))


@pytest.mark.parametrize("score", INVALID_SCORES)
def test_invalid_custom_rescore_is_rejected(score):
    def rescore(evidence, seed, base_score):
        return score

    with pytest.raises(ValueError, match="score"):
        witness_quorum("invalid-rescore", 0.2, np.array([0.2]), rescorer=rescore)


@pytest.mark.parametrize("score", INVALID_SCORES)
def test_invalid_external_scores_cannot_be_witnessed(score):
    opinions = [WitnessOpinion(str(i), i, score, "allow") for i in range(3)]
    with pytest.raises(ValueError, match="score"):
        witness_quorum("invalid-external", 0.2, np.array([0.2]),
                       external_opinions=opinions)


@pytest.mark.parametrize("score,band", [(0.95, "allow"), (0.1, "deny"),
                                       (0.5, "unknown"), (0.35, "allow"),
                                       (0.65, "advisory")])
def test_external_band_must_match_score(score, band):
    opinions = [WitnessOpinion(str(i), i, score, band) for i in range(3)]
    with pytest.raises(ValueError, match="band"):
        witness_quorum("mismatched-band", 0.2, np.array([0.2]),
                       external_opinions=opinions)


@pytest.mark.parametrize("allow,deny", [(math.nan, 0.65), (0.35, math.inf),
                                       (-0.1, 0.65), (0.35, 1.1),
                                       (0.8, 0.2), (0.5, 0.5)])
def test_invalid_thresholds_are_rejected_even_without_witnesses(allow, deny):
    with pytest.raises(ValueError, match="threshold|allow_thr|deny_thr"):
        witness_quorum("invalid-thresholds", 0.2, np.array([0.2]),
                       allow_thr=allow, deny_thr=deny, external_opinions=[])


@pytest.mark.parametrize("score", [math.nan, math.inf, -math.inf])
def test_direct_band_mapping_does_not_turn_nonfinite_score_into_a_verdict(score):
    with pytest.raises(ValueError, match="score"):
        verdict_band(score)


def test_default_rescorer_nonfinite_output_is_rejected():
    with pytest.raises(ValueError, match="score"):
        witness_quorum("invalid-evidence", 0.2, np.array([math.nan]))


@pytest.mark.parametrize("score", INVALID_SCORES)
def test_envelope_with_invalid_score_does_not_gain_a_witness_receipt(score):
    envelope = {"anomaly_score": score, "_signing": {"status": "UNSIGNED"}}
    with pytest.raises(ValueError, match="base_score"):
        witness_envelope(envelope)
    assert "_witness" not in envelope


@pytest.mark.parametrize("score,band", [(0.0, "allow"), (0.2, "advisory"),
                                       (0.8, "deny"), (1.0, "deny")])
def test_valid_custom_thresholds_and_score_boundaries_remain_witnessed(score, band):
    opinions = [WitnessOpinion(str(i), i, np.float64(score), band)
                for i in range(3)]
    receipt = witness_quorum("valid-boundary", score, np.array([score]),
                             allow_thr=0.2, deny_thr=0.8,
                             external_opinions=opinions).to_dict()
    assert receipt["witnessed"] is True
    assert receipt["agreed_band"] == band
    assert receipt["verified"] is False
    assert receipt["advisory"] is True
    json.dumps(receipt, allow_nan=False)


def test_first_seen_duplicate_policy_is_preserved():
    opinions = [WitnessOpinion(str(i), i, 0.2, "allow") for i in range(3)]
    opinions.append(WitnessOpinion("0", 9, math.nan, "invalid"))
    receipt = witness_quorum("duplicates", 0.2, np.array([0.2]),
                             external_opinions=opinions)
    assert receipt.witnessed is True
    assert receipt.n_witnesses == 3
    assert receipt.n_opinions_submitted == 4

