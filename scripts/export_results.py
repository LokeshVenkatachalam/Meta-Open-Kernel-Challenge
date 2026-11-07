#!/usr/bin/env python3
"""Build the public result datasets without importing or executing a kernel."""

from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
import math
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RESULTS_SOURCE = ROOT / "data" / "results-source.json"
RUNS_SOURCE = ROOT / "data" / "runs-source.json"
RESULTS_OUTPUT = ROOT / "docs" / "data" / "results.json"
RUNS_OUTPUT = ROOT / "docs" / "data" / "runs.json"
WITHDRAWAL_SOURCE = ROOT / "data" / "withdrawal-evidence.json"
WITHDRAWAL_OUTPUT = ROOT / "docs" / "data" / "withdrawal.json"

PROBLEM_ORDER = (
    "muse_bf16_proj",
    "muse_attention_decode",
    "muse_moe_w4a16",
    "muse_decode_layer",
    "muse_prefill_layer",
)
RUN_FIELDS = (
    "id",
    "problem",
    "mode",
    "submittedAt",
    "finishedAt",
    "status",
    "passed",
    "speedup",
    "ci95",
    "scoreUs",
    "baselineUs",
)
ARCHIVE_JOB_FIELDS_USED = (
    "id",
    "board",
    "mode",
    "created_at",
    "finished_at",
    "status",
    "result",
)
ARCHIVE_RESULT_FIELDS_USED = (
    "passed",
    "speedup",
    "speedup_ci",
    "score_us",
    "baseline_us",
)
PRIVATE_MARKERS = ("/home/", "/Users/", "\\Users\\", ".okc/")


class ExportError(RuntimeError):
    """Raised when curated or archived evidence violates the public contract."""


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ExportError(f"cannot read JSON from {path}: {exc}") from exc


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def require_dict(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ExportError(f"{label} must be an object")
    return value


def require_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ExportError(f"{label} must be a non-empty string")
    return value


def optional_number(value: Any, label: str) -> int | float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ExportError(f"{label} must be a finite JSON number or null")
    if not math.isfinite(value):
        raise ExportError(f"{label} must be finite")
    return value


def timestamp_from_epoch(value: Any, label: str) -> str | None:
    number = optional_number(value, label)
    if number is None:
        return None
    try:
        return datetime.fromtimestamp(number, timezone.utc).isoformat()
    except (OverflowError, OSError, ValueError) as exc:
        raise ExportError(f"{label} is not a valid Unix timestamp") from exc


def parse_timestamp(value: Any, label: str) -> datetime:
    text = require_string(value, label)
    normalized = text[:-1] + "+00:00" if text.endswith("Z") else text
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise ExportError(f"{label} must be ISO-8601") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ExportError(f"{label} must include a UTC offset")
    return parsed.astimezone(timezone.utc)


def sanitize_archive_job(raw_value: Any, source: Path) -> dict[str, Any]:
    raw = require_dict(raw_value, str(source))
    result = require_dict(raw.get("result"), f"{source}: result")

    ci_value = result.get("speedup_ci")
    if ci_value is None:
        ci95 = None
    else:
        if not isinstance(ci_value, list) or len(ci_value) != 2:
            raise ExportError(f"{source}: result.speedup_ci must contain two numbers or be null")
        ci95 = [
            optional_number(ci_value[0], f"{source}: result.speedup_ci[0]"),
            optional_number(ci_value[1], f"{source}: result.speedup_ci[1]"),
        ]
        if None in ci95:
            raise ExportError(f"{source}: result.speedup_ci cannot contain null")

    passed = result.get("passed")
    if passed is not None and not isinstance(passed, bool):
        raise ExportError(f"{source}: result.passed must be boolean or null")

    record = {
        "id": require_string(raw.get("id"), f"{source}: id"),
        "problem": require_string(raw.get("board"), f"{source}: board"),
        "mode": require_string(raw.get("mode"), f"{source}: mode"),
        "submittedAt": timestamp_from_epoch(raw.get("created_at"), f"{source}: created_at"),
        "finishedAt": timestamp_from_epoch(raw.get("finished_at"), f"{source}: finished_at"),
        "status": require_string(raw.get("status"), f"{source}: status"),
        "passed": passed,
        "speedup": optional_number(result.get("speedup"), f"{source}: result.speedup"),
        "ci95": ci95,
        "scoreUs": optional_number(result.get("score_us"), f"{source}: result.score_us"),
        "baselineUs": optional_number(result.get("baseline_us"), f"{source}: result.baseline_us"),
    }
    if tuple(record) != RUN_FIELDS:
        raise ExportError("internal run allowlist mismatch")
    return record


def sanitize_archive(archive_root: Path) -> dict[str, Any]:
    if not archive_root.is_dir():
        raise ExportError(f"archive root is not a directory: {archive_root}")
    paths = sorted(archive_root.glob("*.json"))
    if not paths:
        raise ExportError(f"archive root contains no JSON jobs: {archive_root}")
    runs = [sanitize_archive_job(load_json(path), path) for path in paths]
    runs.sort(key=lambda item: parse_timestamp(item["submittedAt"], f"run {item['id']} submittedAt"))
    return {"schemaVersion": 1, "scope": "five-agent-campaign", "runs": runs}


def validate_run_record(value: Any, index: int) -> dict[str, Any]:
    run = require_dict(value, f"runs[{index}]")
    if tuple(run) != RUN_FIELDS:
        extra = sorted(set(run) - set(RUN_FIELDS))
        missing = sorted(set(RUN_FIELDS) - set(run))
        raise ExportError(f"runs[{index}] fields differ from allowlist; extra={extra}, missing={missing}")
    require_string(run["id"], f"runs[{index}].id")
    if run["problem"] not in PROBLEM_ORDER:
        raise ExportError(f"runs[{index}].problem is not a published problem")
    if run["mode"] not in {"check", "run"}:
        raise ExportError(f"runs[{index}].mode must be check or run")
    parse_timestamp(run["submittedAt"], f"runs[{index}].submittedAt")
    if run["finishedAt"] is not None:
        parse_timestamp(run["finishedAt"], f"runs[{index}].finishedAt")
    require_string(run["status"], f"runs[{index}].status")
    if run["passed"] is not None and not isinstance(run["passed"], bool):
        raise ExportError(f"runs[{index}].passed must be boolean or null")
    optional_number(run["speedup"], f"runs[{index}].speedup")
    optional_number(run["scoreUs"], f"runs[{index}].scoreUs")
    optional_number(run["baselineUs"], f"runs[{index}].baselineUs")
    ci95 = run["ci95"]
    if ci95 is not None:
        if not isinstance(ci95, list) or len(ci95) != 2:
            raise ExportError(f"runs[{index}].ci95 must contain two numbers or be null")
        optional_number(ci95[0], f"runs[{index}].ci95[0]")
        optional_number(ci95[1], f"runs[{index}].ci95[1]")
    if run["speedup"] is None:
        if run["ci95"] is not None or run["scoreUs"] is not None or run["baselineUs"] is not None:
            raise ExportError(f"runs[{index}] has timing metrics without a speedup")
    elif run["ci95"] is None or run["scoreUs"] is None or run["baselineUs"] is None:
        raise ExportError(f"runs[{index}] has an incomplete timing result")
    return run


def validate_runs_document(value: Any) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    document = require_dict(value, "runs document")
    if set(document) != {"schemaVersion", "scope", "runs"}:
        raise ExportError("runs document must contain only schemaVersion, scope, and runs")
    if document["schemaVersion"] != 1 or document["scope"] != "five-agent-campaign":
        raise ExportError("unsupported runs document version or scope")
    if not isinstance(document["runs"], list):
        raise ExportError("runs must be an array")
    runs = [validate_run_record(value, index) for index, value in enumerate(document["runs"])]
    if len(runs) != 79:
        raise ExportError(f"expected 79 campaign records, found {len(runs)}")
    ids = [run["id"] for run in runs]
    if len(set(ids)) != len(ids):
        raise ExportError("campaign run IDs must be unique")
    chronological = sorted(
        runs,
        key=lambda item: parse_timestamp(item["submittedAt"], f"run {item['id']} submittedAt"),
    )
    if ids != [run["id"] for run in chronological]:
        raise ExportError("campaign runs must be sorted chronologically")
    modes = Counter(run["mode"] for run in runs)
    if modes != Counter({"check": 61, "run": 18}):
        raise ExportError(f"expected 18 scored runs and 61 checks, found {dict(modes)}")
    return document, runs



def validate_withdrawal_document(value: Any) -> dict[str, Any]:
    document = require_dict(value, "withdrawal evidence")
    if set(document) != {"schemaVersion", "scope", "problemId", "actions", "verification"}:
        raise ExportError("withdrawal evidence has unexpected or missing fields")
    if document["schemaVersion"] != 1:
        raise ExportError("unsupported withdrawal evidence schemaVersion")
    if document["scope"] != "prefill-scored-result-withdrawal":
        raise ExportError("unexpected withdrawal evidence scope")
    if document["problemId"] != "muse_prefill_layer":
        raise ExportError("withdrawal evidence must apply to muse_prefill_layer")

    expected_actions = [
        {
            "jobId": "c43f7aba5ae84778",
            "action": "scored-result-withdrawn",
            "confirmedAt": "2026-11-07T09:04:20+00:00",
        },
        {
            "jobId": "23b1cc7e2a6040af",
            "action": "scored-result-withdrawn",
            "confirmedAt": "2026-11-07T09:04:22+00:00",
        },
    ]
    if document["actions"] != expected_actions:
        raise ExportError("withdrawal evidence actions do not match the verified public facts")
    expected_verification = {
        "verifiedAt": "2026-11-07T09:04:37+00:00",
        "result": "no-active-prefill-entry-remains",
    }
    if document["verification"] != expected_verification:
        raise ExportError("withdrawal verification does not match the verified public facts")
    for index, action in enumerate(document["actions"]):
        parse_timestamp(action["confirmedAt"], f"withdrawal.actions[{index}].confirmedAt")
    parse_timestamp(document["verification"]["verifiedAt"], "withdrawal.verification.verifiedAt")
    validate_public_strings(document, "withdrawal")
    return document

def kernel_digest_and_ast(path_text: Any, expected_digest: Any, kernel_id: str) -> None:
    relative = Path(require_string(path_text, f"{kernel_id}.source.path"))
    if relative.is_absolute() or ".." in relative.parts:
        raise ExportError(f"{kernel_id}.source.path must remain inside the repository")
    path = ROOT / relative
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise ExportError(f"cannot read retained source {relative}: {exc}") from exc
    digest = hashlib.sha256(payload).hexdigest()
    if digest != require_string(expected_digest, f"{kernel_id}.source.sha256"):
        raise ExportError(f"retained source hash mismatch for {kernel_id}: {digest}")
    try:
        ast.parse(payload, filename=str(relative))
    except SyntaxError as exc:
        raise ExportError(f"retained source is not valid Python syntax: {relative}: {exc}") from exc


def metric_record(run: dict[str, Any], include_times: bool) -> dict[str, Any]:
    if run["speedup"] is None or run["ci95"] is None:
        raise ExportError(f"selected result {run['id']} has no measurement")
    metric = {"jobId": run["id"], "speedup": run["speedup"], "ci95": run["ci95"]}
    if include_times:
        metric["scoreUs"] = run["scoreUs"]
        metric["baselineUs"] = run["baselineUs"]
    return metric


def build_results(source_value: Any, runs: list[dict[str, Any]]) -> dict[str, Any]:
    source = require_dict(source_value, "results source")
    required_top = {
        "schemaVersion",
        "project",
        "publication",
        "leaderboardSnapshotAt",
        "campaign",
        "accountTotals",
        "kernels",
    }
    if set(source) != required_top:
        raise ExportError("results source has unexpected or missing top-level fields")
    if source["schemaVersion"] != 2:
        raise ExportError("results source must use schemaVersion 2")
    parse_timestamp(source["leaderboardSnapshotAt"], "leaderboardSnapshotAt")

    publication = require_dict(source["publication"], "publication")
    publication_fields = {
        "workflow",
        "originalityAudit",
        "correctedAt",
        "disclosurePath",
        "excludedProblemIds",
        "withdrawnScoredJobIds",
        "withdrawalVerifiedAt",
        "notice",
    }
    if set(publication) != publication_fields:
        raise ExportError("publication disclosure has unexpected or missing fields")
    if publication["workflow"] != "fully-agentic":
        raise ExportError("publication.workflow must be fully-agentic")
    if publication["originalityAudit"] != "incomplete":
        raise ExportError("publication.originalityAudit must be incomplete")
    if publication["correctedAt"] != "2026-11-07T09:04:37+00:00":
        raise ExportError("publication.correctedAt does not match the correction record")
    if publication["disclosurePath"] != "DISCLOSURE.md":
        raise ExportError("publication.disclosurePath must be DISCLOSURE.md")
    if publication["excludedProblemIds"] != ["muse_prefill_layer"]:
        raise ExportError("publication must exclude exactly muse_prefill_layer")
    if publication["withdrawnScoredJobIds"] != ["c43f7aba5ae84778", "23b1cc7e2a6040af"]:
        raise ExportError("publication withdrawn job IDs do not match the verified record")
    if publication["withdrawalVerifiedAt"] != "2026-11-07T09:04:37+00:00":
        raise ExportError("publication.withdrawalVerifiedAt does not match the verification record")
    parse_timestamp(publication["correctedAt"], "publication.correctedAt")
    parse_timestamp(publication["withdrawalVerifiedAt"], "publication.withdrawalVerifiedAt")
    notice = require_string(publication["notice"], "publication.notice")
    for phrase in ("AI-agent-driven", "verbatim copies", "remaining four", "originality audit"):
        if phrase not in notice:
            raise ExportError(f"publication.notice must include {phrase!r}")

    campaign = require_dict(source["campaign"], "campaign")
    account_totals = require_dict(source["accountTotals"], "accountTotals")
    if (campaign.get("runCount"), campaign.get("scoredCount"), campaign.get("checkCount")) != (79, 18, 61):
        raise ExportError("campaign totals must remain 79 runs, 18 scored, and 61 checks")
    if (account_totals.get("runCount"), account_totals.get("scoredCount"), account_totals.get("checkCount")) != (92, 20, 72):
        raise ExportError("account totals must remain 92 runs, 20 scored, and 72 checks")

    run_by_id = {run["id"]: run for run in runs}
    source_kernels = source.get("kernels")
    if not isinstance(source_kernels, list) or len(source_kernels) != 5:
        raise ExportError("results source must define exactly five kernels")
    source_kernel_objects = [require_dict(kernel, f"kernels[{index}]") for index, kernel in enumerate(source_kernels)]
    if [kernel.get("id") for kernel in source_kernel_objects] != list(PROBLEM_ORDER):
        raise ExportError("kernels must use the contract order")

    output_kernels = []
    account_sums = Counter()
    included_count = 0
    excluded_count = 0
    for kernel in source_kernel_objects:
        kernel_id = require_string(kernel.get("id"), "kernel.id")
        included = kernel.get("includedInSummary")
        if not isinstance(included, bool):
            raise ExportError(f"{kernel_id}.includedInSummary must be boolean")
        expected_included = kernel_id != "muse_prefill_layer"
        if included != expected_included:
            raise ExportError(f"{kernel_id}.includedInSummary conflicts with the correction contract")
        included_count += int(included)
        excluded_count += int(not included)

        problem_runs = [run for run in runs if run["problem"] == kernel_id]
        campaign_modes = Counter(run["mode"] for run in problem_runs)
        campaign_counts = {
            "scored": campaign_modes["run"],
            "checks": campaign_modes["check"],
            "runs": len(problem_runs),
        }
        account_counts = require_dict(kernel.get("accountCounts"), f"{kernel_id}.accountCounts")
        if set(account_counts) != {"scored", "checks", "runs"}:
            raise ExportError(f"{kernel_id}.accountCounts has unexpected fields")
        if account_counts["runs"] != account_counts["scored"] + account_counts["checks"]:
            raise ExportError(f"{kernel_id}.accountCounts does not add up")
        if account_counts["scored"] < campaign_counts["scored"] or account_counts["checks"] < campaign_counts["checks"]:
            raise ExportError(f"{kernel_id}.accountCounts is smaller than the historical campaign")
        account_sums.update(account_counts)

        scored_id = require_string(kernel.get("scoredJobId"), f"{kernel_id}.scoredJobId")
        check_id = require_string(kernel.get("checkJobId"), f"{kernel_id}.checkJobId")
        scored_run = run_by_id.get(scored_id)
        check_run = run_by_id.get(check_id)
        if not scored_run or scored_run["problem"] != kernel_id or scored_run["mode"] != "run" or scored_run["passed"] is not True:
            raise ExportError(f"{kernel_id}.scoredJobId is not a passing historical scored run")
        if not check_run or check_run["problem"] != kernel_id or check_run["mode"] != "check" or check_run["passed"] is not True:
            raise ExportError(f"{kernel_id}.checkJobId is not a passing historical check")
        measured_checks = [
            run for run in problem_runs
            if run["mode"] == "check" and run["passed"] is True and run["speedup"] is not None
        ]
        if not measured_checks or max(measured_checks, key=lambda run: run["speedup"])["id"] != check_id:
            raise ExportError(f"{kernel_id}.checkJobId is not the highest passing historical campaign check")

        ranking = require_dict(kernel.get("ranking"), f"{kernel_id}.ranking")
        if set(ranking) != {"rank", "placed", "bestSpeedup"}:
            raise ExportError(f"{kernel_id}.ranking has unexpected fields")
        if not all(isinstance(ranking[key], int) and not isinstance(ranking[key], bool) for key in ("rank", "placed")):
            raise ExportError(f"{kernel_id}.ranking rank and placed must be integers")
        if not 1 < ranking["rank"] <= ranking["placed"]:
            raise ExportError(f"{kernel_id} historical rank must be non-winning and within placed")
        optional_number(ranking["bestSpeedup"], f"{kernel_id}.ranking.bestSpeedup")
        if ranking["bestSpeedup"] <= scored_run["speedup"]:
            raise ExportError(f"{kernel_id} historical leader speedup must exceed the measured score")

        source_info = require_dict(kernel.get("source"), f"{kernel_id}.source")
        expected_source_fields = {
            "path",
            "sha256",
            "pairedCheckJobId",
            "license",
            "reusedPublicSource",
            "originalityStatus",
            "provenancePath",
        }
        if set(source_info) != expected_source_fields:
            raise ExportError(f"{kernel_id}.source has unexpected fields")
        paired_id = require_string(source_info["pairedCheckJobId"], f"{kernel_id}.source.pairedCheckJobId")
        paired_run = run_by_id.get(paired_id)
        if not paired_run or paired_run["problem"] != kernel_id or paired_run["mode"] != "check" or paired_run["passed"] is not True:
            raise ExportError(f"{kernel_id}.source.pairedCheckJobId is not a passing historical check")
        if included:
            if source_info["reusedPublicSource"] is not None:
                raise ExportError(f"{kernel_id}.source.reusedPublicSource must be null until audited")
            if source_info["originalityStatus"] != "not-comprehensively-audited":
                raise ExportError(f"{kernel_id}.source.originalityStatus must mark the incomplete audit")
        else:
            if source_info["reusedPublicSource"] is not True:
                raise ExportError("prefill source.reusedPublicSource must be true")
            if source_info["originalityStatus"] != "confirmed-verbatim-copy":
                raise ExportError("prefill source.originalityStatus must be confirmed-verbatim-copy")
        kernel_digest_and_ast(source_info["path"], source_info["sha256"], kernel_id)

        first_scored = next(
            (
                run for run in problem_runs
                if run["mode"] == "run" and run["passed"] is True and run["speedup"] is not None
            ),
            None,
        )
        if first_scored is None:
            raise ExportError(f"{kernel_id} has no measured historical campaign score")
        if scored_run["speedup"] <= 1:
            raise ExportError(f"{kernel_id} selected historical measurement did not beat SGLang")

        output_kernel = {
            "id": kernel_id,
            "slug": require_string(kernel.get("slug"), f"{kernel_id}.slug"),
            "title": require_string(kernel.get("title"), f"{kernel_id}.title"),
            "summary": require_string(kernel.get("summary"), f"{kernel_id}.summary"),
            "includedInSummary": included,
            "scored": metric_record(scored_run, include_times=True) if included else None,
            "check": metric_record(check_run, include_times=False) if included else None,
            "ranking": copy.deepcopy(ranking) if included else None,
            "campaignCounts": campaign_counts,
            "accountCounts": copy.deepcopy(account_counts),
            "firstCampaignScoredSpeedup": first_scored["speedup"] if included else None,
            "source": copy.deepcopy(source_info),
            "story": copy.deepcopy(require_dict(kernel.get("story"), f"{kernel_id}.story")),
            "references": copy.deepcopy(kernel.get("references")),
        }
        if not included:
            output_kernel["historical"] = {
                "scored": metric_record(scored_run, include_times=True),
                "check": metric_record(check_run, include_times=False),
                "ranking": copy.deepcopy(ranking),
                "firstCampaignScoredSpeedup": first_scored["speedup"],
            }
        output_kernels.append(output_kernel)

    if included_count != 4 or excluded_count != 1:
        raise ExportError(f"expected four included and one excluded kernel, found {included_count} and {excluded_count}")
    if dict(account_sums) != {"scored": 20, "checks": 72, "runs": 92}:
        raise ExportError(f"per-problem account counts do not match account totals: {dict(account_sums)}")

    output = {
        "schemaVersion": 2,
        "project": copy.deepcopy(require_dict(source["project"], "project")),
        "publication": copy.deepcopy(publication),
        "leaderboardSnapshotAt": source["leaderboardSnapshotAt"],
        "campaign": copy.deepcopy(campaign),
        "accountTotals": copy.deepcopy(account_totals),
        "kernels": output_kernels,
    }
    validate_public_strings(output, "results")
    return output

def validate_public_strings(value: Any, label: str) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            validate_public_strings(child, f"{label}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            validate_public_strings(child, f"{label}[{index}]")
    elif isinstance(value, str):
        for marker in PRIVATE_MARKERS:
            if marker in value:
                raise ExportError(f"private path marker {marker!r} found in {label}")


def compare_json(path: Path, expected: Any) -> None:
    actual = load_json(path)
    if actual != expected:
        raise ExportError(f"generated file is missing or stale: {path.relative_to(ROOT)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--archive-root",
        type=Path,
        help="sanitize archived official jobs into data/runs-source.json before exporting",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="validate committed sources and require generated files to be current",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if args.archive_root:
            runs_document = sanitize_archive(args.archive_root)
            validate_runs_document(runs_document)
            if args.check:
                compare_json(RUNS_SOURCE, runs_document)
            else:
                write_json(RUNS_SOURCE, runs_document)
        else:
            runs_document = load_json(RUNS_SOURCE)

        runs_document, runs = validate_runs_document(runs_document)
        results_document = build_results(load_json(RESULTS_SOURCE), runs)
        withdrawal_document = validate_withdrawal_document(load_json(WITHDRAWAL_SOURCE))
        withdrawn_ids = [action["jobId"] for action in withdrawal_document["actions"]]
        if results_document["publication"]["withdrawnScoredJobIds"] != withdrawn_ids:
            raise ExportError("results publication and withdrawal evidence disagree on job IDs")
        if results_document["publication"]["withdrawalVerifiedAt"] != withdrawal_document["verification"]["verifiedAt"]:
            raise ExportError("results publication and withdrawal evidence disagree on verification time")
        validate_public_strings(runs_document, "runs")

        if args.check:
            compare_json(RESULTS_OUTPUT, results_document)
            compare_json(RUNS_OUTPUT, runs_document)
            compare_json(WITHDRAWAL_OUTPUT, withdrawal_document)
        else:
            write_json(RESULTS_OUTPUT, results_document)
            write_json(RUNS_OUTPUT, runs_document)
            write_json(WITHDRAWAL_OUTPUT, withdrawal_document)

        print(
            "validated 4 included kernels, 1 excluded kernel, and {} historical campaign jobs "
            "({} scored, {} checks)".format(
                len(runs),
                sum(run["mode"] == "run" for run in runs),
                sum(run["mode"] == "check" for run in runs),
            )
        )
        return 0
    except ExportError as exc:
        print(f"export_results.py: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
