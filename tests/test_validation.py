import copy
import json
import tempfile
import unittest
from pathlib import Path

from scripts.validate_publication import (
    ParsedPage,
    finite_number,
    interval,
    kernel_evidence,
    load_json,
    safe_path,
    timestamp,
    validate_publication_policy,
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


class DisclosurePolicyTests(unittest.TestCase):
    def setUp(self):
        self.jobs = {
            job_id: {"problem": "muse_prefill_layer", "mode": "run"}
            for job_id in ("c43f7aba5ae84778", "23b1cc7e2a6040af")
        }
        self.results = {
            "schemaVersion": 2,
            "campaign": {"closedAt": "2026-11-05T08:00:00+00:00"},
            "publication": {
                "workflow": "fully-agentic",
                "originalityAudit": "incomplete",
                "excludedProblemIds": ["muse_prefill_layer"],
                "withdrawnScoredJobIds": list(self.jobs),
                "withdrawalVerifiedAt": "2026-11-07T09:04:37+00:00",
                "correctedAt": "2026-11-07T09:04:37+00:00",
                "disclosurePath": "DISCLOSURE.md",
                "notice": "AI agents assembled the work; copying was confirmed and originality is not fully audited.",
            },
        }
        self.prefill = {
            "id": "muse_prefill_layer",
            "includedInSummary": False,
            "source": {"originalityStatus": "confirmed-verbatim-copy", "reusedPublicSource": True},
            "scored": None,
            "check": None,
            "ranking": None,
            "firstCampaignScoredSpeedup": None,
            "historical": {"scored": {}, "check": {}, "ranking": {}, "firstCampaignScoredSpeedup": 1.7},
        }

    def test_corrected_policy_preserves_historical_evidence(self):
        excluded = validate_publication_policy(self.results, self.jobs)
        self.assertEqual(kernel_evidence(self.prefill, excluded), self.prefill["historical"])

    def test_copied_result_cannot_be_reincluded(self):
        self.prefill["includedInSummary"] = True
        with self.assertRaisesRegex(ValueError, "restored"):
            kernel_evidence(self.prefill, {"muse_prefill_layer"})

    def test_copied_result_cannot_keep_active_score_or_rank(self):
        for field in ("scored", "check", "ranking", "firstCampaignScoredSpeedup"):
            candidate = copy.deepcopy(self.prefill)
            candidate[field] = 1.9
            with self.assertRaisesRegex(ValueError, "active"):
                kernel_evidence(candidate, {"muse_prefill_layer"})

    def test_unrelated_withdrawals_are_rejected(self):
        self.jobs["c43f7aba5ae84778"]["problem"] = "muse_bf16_proj"
        with self.assertRaisesRegex(ValueError, "another problem"):
            validate_publication_policy(self.results, self.jobs)

    def test_originality_certification_is_rejected(self):
        self.results["publication"]["originalityAudit"] = "complete"
        with self.assertRaisesRegex(ValueError, "Originality"):
            validate_publication_policy(self.results, self.jobs)

    def test_unaudited_source_cannot_claim_no_copying(self):
        candidate = {
            "id": "muse_bf16_proj",
            "includedInSummary": True,
            "source": {"originalityStatus": "not-comprehensively-audited", "reusedPublicSource": False},
        }
        with self.assertRaisesRegex(ValueError, "unaudited"):
            kernel_evidence(candidate, {"muse_prefill_layer"})

    def test_legacy_schema_cannot_restore_old_claims(self):
        self.results["schemaVersion"] = 1
        with self.assertRaisesRegex(ValueError, "corrected"):
            validate_publication_policy(self.results, self.jobs)


if __name__ == "__main__":
    unittest.main()
