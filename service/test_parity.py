# service/test_parity.py
"""Compare /predict probabilities with MATLAB predict_ssi outputs in a CSV.

CSV columns: image_path,matlab_probability,matlab_status (status optional).
Only parity images with a finite MATLAB probability should be included. Paths
may be absolute or relative to the CSV file. Run with the service and these
images available to this process:
  MATLAB_PARITY_CSV=path/to/matlab_parity.csv python -m unittest service.test_parity
"""

from __future__ import annotations

import csv
import os
import unittest
from pathlib import Path

import requests


class MatlabServiceParityTest(unittest.TestCase):
    def test_probabilities_match_saved_matlab_outputs(self) -> None:
        csv_value = os.getenv("MATLAB_PARITY_CSV", "")
        self.assertTrue(csv_value, "Set MATLAB_PARITY_CSV to MATLAB output CSV")
        csv_path = Path(csv_value).resolve()
        self.assertTrue(csv_path.is_file(), "MATLAB parity CSV is missing")
        base_url = os.getenv("WOUNDWATCH_URL", "http://127.0.0.1:8000").rstrip("/")
        tolerance = float(os.getenv("PARITY_TOLERANCE", "0.001"))
        with csv_path.open("r", newline="", encoding="utf-8-sig") as handle:
            rows = list(csv.DictReader(handle))
        self.assertGreater(len(rows), 0, "MATLAB parity CSV has no data rows")
        for row_number, row in enumerate(rows, start=2):
            self.assertIn("image_path", row, "CSV needs image_path column")
            self.assertIn("matlab_probability", row, "CSV needs matlab_probability column")
            image_path = Path(row["image_path"])
            if not image_path.is_absolute():
                image_path = (csv_path.parent / image_path).resolve()
            self.assertTrue(image_path.is_file(), f"CSV row {row_number} image is unavailable")
            expected = float(row["matlab_probability"])
            self.assertTrue(0.0 <= expected <= 1.0, f"CSV row {row_number} probability is outside [0,1]")
            with image_path.open("rb") as image_file:
                response = requests.post(
                    f"{base_url}/predict",
                    files={"file": ("parity-image", image_file, "application/octet-stream")},
                    timeout=120,
                )
            self.assertEqual(response.status_code, 200, f"Service failed on CSV row {row_number}")
            actual_result = response.json()
            self.assertIn(actual_result.get("status"), {"ok", "defer"},
                          f"Service gate rejected parity row {row_number}")
            self.assertIsNotNone(actual_result.get("probability"),
                                 f"Service returned no probability for row {row_number}")
            actual = float(actual_result["probability"])
            self.assertLessEqual(abs(actual - expected), tolerance,
                                 f"Probability mismatch at CSV row {row_number}")
            if row.get("matlab_status"):
                self.assertEqual(actual_result["status"], row["matlab_status"],
                                 f"Status mismatch at CSV row {row_number}")


if __name__ == "__main__":
    unittest.main()
