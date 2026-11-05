import copy
import json
import tempfile
import unittest
from pathlib import Path

from scripts.validate_publication import (
    ParsedPage,
    finite_number,
    interval,
    load_json,
    safe_path,
    timestamp,
    validate_runs,
)


class DataValidationTests(unittest.TestCase):
    def setUp(self):
        self.campaign = {
            "startedAt": "2026-11-05T06:26:06.830+00:00",
            "closedAt": "2026-11-05T08:00:00+00:00",
            "runCount": 1,
            "scoredCount": 1,
            "checkCount": 0,
        }
        self.row = {
            "id": "0123456789abcdef",
            "problem": "muse_bf16_proj",
            "mode": "run",
            "submittedAt": "2026-11-05T07:59:00Z",
            "finishedAt": "2026-11-05T08:02:00Z",
            "status": "done",
            "passed": True,
            "speedup": 1.2,
            "ci95": [1.1, 1.3],
            "scoreUs": 100.0,
            "baselineUs": 120.0,
        }

    def document(self, row=None):
        return {
            "schemaVersion": 1,
            "scope": "five-agent-campaign",
            "runs": [self.row if row is None else row],
        }

    def test_accepted_run_can_finish_after_cutoff(self):
        self.assertIn(self.row["id"], validate_runs(self.document(), self.campaign))

    def test_post_cutoff_submission_is_rejected(self):
        self.row["submittedAt"] = self.campaign["closedAt"]
        with self.assertRaisesRegex(ValueError, "campaign window"):
            validate_runs(self.document(), self.campaign)

    def test_failed_run_has_no_invented_speedup(self):
        self.row["passed"] = False
        with self.assertRaisesRegex(ValueError, "performance claim"):
            validate_runs(self.document(), self.campaign)
        self.row.update(speedup=None, ci95=None, scoreUs=None)
        validate_runs(self.document(), self.campaign)

    def test_raw_grader_fields_cannot_leak_into_dataset(self):
        self.row["log_tail"] = "not for publication"
        with self.assertRaisesRegex(ValueError, "allowlist"):
            validate_runs(self.document(), self.campaign)

    def test_duplicate_jobs_are_rejected(self):
        data = self.document()
        data["runs"].append(copy.deepcopy(self.row))
        with self.assertRaisesRegex(ValueError, "Duplicate job"):
            validate_runs(data, self.campaign)

    def test_counts_are_checked(self):
        self.campaign["runCount"] = 2
        with self.assertRaisesRegex(ValueError, "total"):
            validate_runs(self.document(), self.campaign)

    def test_non_numeric_and_non_finite_scores_are_rejected(self):
        for value in (True, "1.2", float("inf"), float("nan")):
            with self.assertRaises(ValueError):
                finite_number(value, "speedup")

    def test_confidence_interval_must_contain_score(self):
        for bounds in ([1.3, 1.1], [1.4, 1.5], [1.0], ["low", 1.3]):
            with self.assertRaises(ValueError):
                interval(bounds, 1.2, "CI")

    def test_timestamp_requires_utc(self):
        for value in ("2026-11-05T08:00:00", "2026-11-05T08:00:00+01:00"):
            with self.assertRaises(ValueError):
                timestamp(value, "time")

    def test_json_non_standard_number_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.json"
            path.write_text('{"score": NaN}')
            with self.assertRaisesRegex(ValueError, "Non-standard JSON"):
                load_json(path)

    def test_relative_file_boundary(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "inside.txt").write_text("public")
            self.assertEqual(safe_path(root, "inside.txt"), root / "inside.txt")
            with self.assertRaisesRegex(ValueError, "escapes"):
                safe_path(root, "../outside.txt")
            with self.assertRaisesRegex(ValueError, "relative"):
                safe_path(root, str(root / "inside.txt"))

    def test_parser_detects_duplicate_ids_and_collects_links(self):
        page = ParsedPage()
        page.feed('<html lang="en"><title>Title</title><meta name="viewport" content="width=device-width"><h1 id="same">Hello</h1><p id="same">Again</p><a href="other.html#section">Link</a></html>')
        self.assertEqual(page.duplicate_ids, {"same"})
        self.assertEqual(page.h1_count, 1)
        self.assertTrue(page.viewport)
        self.assertIn(("a", "other.html#section"), page.links)


if __name__ == "__main__":
    unittest.main()
