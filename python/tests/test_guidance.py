import unittest
import numpy as np
from woundwatch.guidance import attention_regions, build_guidance


class GuidanceTests(unittest.TestCase):
    def test_shares_sum_to_one(self):
        cam = np.ones((10, 10)); wound = np.zeros((10, 10), bool); wound[:3] = True
        ring = np.zeros((10, 10), bool); ring[3:5] = True
        s = attention_regions(cam, wound, ring)
        self.assertAlmostEqual(s["wound"] + s["ring"] + s["outside"], 1.0)

    def test_retake_gives_no_risk_statement(self):
        g = build_guidance({"status": "retake", "reasons": ["Image may be blurred"]})
        self.assertIn("Retake", g["check"][0]); self.assertNotIn("Elevated", g["summary"])

    def test_low_never_claims_absence(self):
        g = build_guidance({"status": "ok", "risk_band": "low", "probability": .05}, {"wound": .6, "ring": .2, "outside": .2})
        self.assertIn("does not exclude", g["summary"])

    def test_outside_attention_flagged_unreliable(self):
        g = build_guidance({"status": "ok", "risk_band": "elevated", "probability": .6}, {"wound": .1, "ring": .1, "outside": .8})
        self.assertTrue(any("unreliable" in o for o in g["observations"]))

    def test_missing_outline_hides_region_and_colour_claims(self):
        g = build_guidance({"status": "ok", "risk_band": "elevated", "probability": .6}, None, {"erythema_a_mean_difference": 5, "a_ring_mean": 1, "a_wound_mean": 6}, outline_found=False)
        text = " ".join(g["observations"])
        self.assertIn("outline could not be located", text); self.assertNotIn("redder", text)


if __name__ == "__main__":
    unittest.main()
