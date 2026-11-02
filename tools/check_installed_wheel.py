"""Check a base wheel from a clean venv: python -I tools/check_installed_wheel.py.

This check deliberately requires torch and szl-receipt to be absent. The existing
source suite exercises the separate environment with those optional extras.
"""

import importlib
import importlib.metadata
import importlib.util
from pathlib import Path
import sys
import unittest


class InstalledWheelTests(unittest.TestCase):
    def test_all_modules_come_from_installed_distribution(self):
        distribution = importlib.metadata.distribution("khipu-sda-core")
        root = Path(distribution.locate_file("")).resolve()
        checkout = Path(__file__).resolve().parents[1]
        self.assertTrue(root.is_relative_to(Path(sys.prefix).resolve()))
        self.assertFalse(root.is_relative_to(checkout))
        for name in ("szl_confidence", "szl_mosaic_core", "szl_mosaic_validate",
                     "szl_sda_envelope", "szl_sda_orbit", "szl_sda_spine",
                     "szl_track_fusion", "szl_witness"):
            with self.subTest(module=name):
                module = importlib.import_module(name)
                self.assertEqual(Path(module.__file__).resolve(), root / (name + ".py"))

    def test_base_autoencoder_without_torch(self):
        self.assertIsNone(importlib.util.find_spec("torch"))
        import numpy as np
        from szl_mosaic_core import AutoencoderDetector

        data = np.random.default_rng(7).normal(size=(24, 4))
        model = AutoencoderDetector().fit(data)
        self.assertEqual(model._backend, "numpy-pca")
        scores = model.score(data[:3])
        self.assertEqual(scores.shape, (3,))
        self.assertTrue(np.isfinite(scores).all())

    def test_base_receipt_stays_unsigned_without_shared_builder(self):
        self.assertIsNone(importlib.util.find_spec("szl_receipt"))
        from szl_sda_envelope import make_intoto_receipt

        digest = "a" * 64
        receipt = make_intoto_receipt("wheel-check", digest, "model", 7, 0.1)
        self.assertEqual(receipt["subject"][0]["digest"]["sha256"], digest)
        self.assertIs(receipt["predicate"]["verified"], False)
        self.assertNotIn("signatures", receipt)

    def test_base_orbit_has_real_propagator(self):
        from sgp4.api import jday
        from szl_sda_orbit import demo_catalog, propagate_tle

        line1, line2 = next(iter(demo_catalog().values()))
        jd, fr = jday(2026, 1, 1, 0, 0, 0)
        states = propagate_tle(line1, line2, jd, fr, n_steps=3)
        self.assertEqual(len(states), 3)
        self.assertTrue(all(state.error_code == 0 for state in states))


if __name__ == "__main__":
    unittest.main(verbosity=2)
