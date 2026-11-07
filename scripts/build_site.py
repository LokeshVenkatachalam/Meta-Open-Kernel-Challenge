#!/usr/bin/env python3
"""Build the dependency-free Hopper Field Notes static site."""

from __future__ import annotations

import argparse
import html
import json
import math
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlparse


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = ROOT / "docs" / "data"
DEFAULT_OUTPUT_DIR = ROOT / "docs"
DEFAULT_ASSETS_DIR = ROOT / "web" / "assets"

REPOSITORY_URL = "https://github.com/LokeshVenkatachalam/Meta-Open-Kernel-Challenge"
SITE_URL = "https://lokeshvenkatachalam.github.io/Meta-Open-Kernel-Challenge/"
EXPECTED_KERNELS = (
    ("muse_bf16_proj", "bf16-projections", "BF16 projections", 1.131876),
    ("muse_attention_decode", "attention-decode", "Attention decode", 1.151976),
    ("muse_moe_w4a16", "routed-experts", "Routed experts", 1.340289),
    ("muse_decode_layer", "decoder-layer", "Decoder layer", 1.185588),
    ("muse_prefill_layer", "prefill-layer", "Prefill layer", None),
)
EXPECTED_BY_ID = {item[0]: item for item in EXPECTED_KERNELS}
INCLUDED_PROBLEM_IDS = tuple(item[0] for item in EXPECTED_KERNELS[:-1])
EXCLUDED_PROBLEM_ID = "muse_prefill_layer"
WITHDRAWN_SCORED_JOB_IDS = ("c43f7aba5ae84778", "23b1cc7e2a6040af")
WITHDRAWAL_ACTIONS = (
    ("c43f7aba5ae84778", "2026-11-07T09:04:20+00:00"),
    ("23b1cc7e2a6040af", "2026-11-07T09:04:22+00:00"),
)
CONFIRMED_COPY_EVIDENCE = (
    (
        "23b1cc7e2a6040af",
        "f1807a75ec056b7f3874d4fce02331ca0043ac38a45af7710a8287a252421a92",
        "https://github.com/keell0renz/muse-open-kernel-challenge/blob/890809cee421500b00126d084e81b338ef9181ab/kernels/muse_prefill_layer.py",
        "team keell0renz (Bohdan Agarkov, credited in NOTICE)",
    ),
    (
        "c43f7aba5ae84778",
        "76fb7501371fd3331858795e462ce5d0877e4b6a789a10a7c189fb7a8e5aaaa4",
        "https://github.com/Madhavan113/ParadigmKernelChallenge/blob/721c62729fa0b407799692f311790643706f292d/muse_prefill_layer/submission.py",
        "Madhavan113",
    ),
)
CORRECTION_TIMESTAMP = "2026-11-07T09:04:37+00:00"
HISTORICAL_PREFILL_SPEEDUP = 1.919190
STORY_SECTIONS = ("intro", "approach", "lessons", "failures", "caveats")
ASSET_FILES = ("style.css", "site.js", "favicon.svg")


class BuildError(RuntimeError):
    """An actionable content or input error."""


def fail(path: str, message: str) -> None:
    raise BuildError(f"{path}: {message}")


def require_dict(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        fail(path, "expected an object")
    return value


def require_list(value: Any, path: str) -> list[Any]:
    if not isinstance(value, list):
        fail(path, "expected an array")
    return value


def require_string(value: Any, path: str, *, nonempty: bool = True) -> str:
    if not isinstance(value, str):
        fail(path, "expected a string")
    if nonempty and not value.strip():
        fail(path, "must not be empty")
    return value


def require_bool(value: Any, path: str) -> bool:
    if not isinstance(value, bool):
        fail(path, "expected a boolean")
    return value


def require_int(value: Any, path: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        fail(path, "expected an integer")
    if minimum is not None and value < minimum:
        fail(path, f"must be at least {minimum}")
    return value


def require_number(
    value: Any,
    path: str,
    *,
    nullable: bool = False,
    positive: bool = False,
) -> float | None:
    if value is None and nullable:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        fail(path, "expected a JSON number" + (" or null" if nullable else ""))
    number = float(value)
    if not math.isfinite(number):
        fail(path, "must be finite")
    if positive and number <= 0:
        fail(path, "must be greater than zero")
    return number


def require_key(obj: dict[str, Any], key: str, path: str) -> Any:
    if key not in obj:
        fail(path, f"missing required field {key!r}")
    return obj[key]


def parse_iso(value: Any, path: str) -> datetime:
    text = require_string(value, path)
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise BuildError(f"{path}: invalid ISO-8601 timestamp {text!r}") from exc
    if parsed.tzinfo is None:
        fail(path, "timestamp must include a UTC offset")
    return parsed.astimezone(timezone.utc)


def require_https(value: Any, path: str) -> str:
    url = require_string(value, path)
    parsed = urlparse(url)
    if parsed.scheme != "https" or not parsed.netloc:
        fail(path, "expected a public https URL")
    return url


def validate_ci(value: Any, path: str, *, nullable: bool = False) -> list[Any] | None:
    if value is None and nullable:
        return None
    ci = require_list(value, path)
    if len(ci) != 2:
        fail(path, "expected exactly two confidence bounds")
    low = require_number(ci[0], f"{path}[0]")
    high = require_number(ci[1], f"{path}[1]")
    assert low is not None and high is not None
    if low > high:
        fail(path, "lower confidence bound exceeds upper bound")
    return ci


def validate_metric_block(value: Any, path: str, *, scored: bool) -> dict[str, Any]:
    block = require_dict(value, path)
    require_string(require_key(block, "jobId", path), f"{path}.jobId")
    speedup = require_number(require_key(block, "speedup", path), f"{path}.speedup", positive=True)
    ci = validate_ci(require_key(block, "ci95", path), f"{path}.ci95")
    assert speedup is not None and ci is not None
    if speedup < float(ci[0]) - 1e-9 or speedup > float(ci[1]) + 1e-9:
        fail(path, "speedup must fall within ci95")
    if scored:
        require_number(require_key(block, "scoreUs", path), f"{path}.scoreUs", positive=True)
        require_number(require_key(block, "baselineUs", path), f"{path}.baselineUs", positive=True)
    return block


def validate_ranking_block(value: Any, path: str, scored_speedup: float) -> dict[str, Any]:
    ranking = require_dict(value, path)
    rank = require_int(require_key(ranking, "rank", path), f"{path}.rank", minimum=1)
    placed = require_int(require_key(ranking, "placed", path), f"{path}.placed", minimum=1)
    leader = require_number(require_key(ranking, "bestSpeedup", path), f"{path}.bestSpeedup", positive=True)
    assert leader is not None
    if rank > placed:
        fail(path, "rank cannot exceed placed competitor count")
    if rank == 1:
        fail(f"{path}.rank", "the recorded result did not win this board")
    if leader + 1e-9 < scored_speedup:
        fail(f"{path}.bestSpeedup", "leader speedup cannot trail the recorded result")
    return ranking


def validate_results(raw: Any) -> dict[str, Any]:
    results = require_dict(raw, "results.json")
    if require_key(results, "schemaVersion", "results.json") != 2:
        fail("results.json.schemaVersion", "only corrected schema version 2 is supported")

    project = require_dict(require_key(results, "project", "results.json"), "results.json.project")
    expected_project = {
        "title": "Hopper Field Notes",
        "author": "LokeshVenkatachalam",
        "repositoryUrl": REPOSITORY_URL,
        "branch": "main",
        "siteUrl": SITE_URL,
    }
    for key, expected in expected_project.items():
        actual = require_string(require_key(project, key, "results.json.project"), f"results.json.project.{key}")
        if actual != expected:
            fail(f"results.json.project.{key}", f"expected {expected!r}, found {actual!r}")

    publication = require_dict(require_key(results, "publication", "results.json"), "results.json.publication")
    expected_publication_strings = {
        "workflow": "fully-agentic",
        "originalityAudit": "incomplete",
        "disclosurePath": "DISCLOSURE.md",
    }
    for key, expected in expected_publication_strings.items():
        actual = require_string(require_key(publication, key, "results.json.publication"), f"results.json.publication.{key}")
        if actual != expected:
            fail(f"results.json.publication.{key}", f"expected {expected!r}, found {actual!r}")
    if parse_iso(require_key(publication, "correctedAt", "results.json.publication"), "results.json.publication.correctedAt") != parse_iso(CORRECTION_TIMESTAMP, "contract.correctedAt"):
        fail("results.json.publication.correctedAt", f"expected {CORRECTION_TIMESTAMP!r}")
    if parse_iso(require_key(publication, "withdrawalVerifiedAt", "results.json.publication"), "results.json.publication.withdrawalVerifiedAt") != parse_iso(CORRECTION_TIMESTAMP, "contract.withdrawalVerifiedAt"):
        fail("results.json.publication.withdrawalVerifiedAt", f"expected {CORRECTION_TIMESTAMP!r}")
    excluded_ids = require_list(require_key(publication, "excludedProblemIds", "results.json.publication"), "results.json.publication.excludedProblemIds")
    if excluded_ids != [EXCLUDED_PROBLEM_ID]:
        fail("results.json.publication.excludedProblemIds", f"expected only {EXCLUDED_PROBLEM_ID!r}")
    withdrawn_ids = require_list(require_key(publication, "withdrawnScoredJobIds", "results.json.publication"), "results.json.publication.withdrawnScoredJobIds")
    if withdrawn_ids != list(WITHDRAWN_SCORED_JOB_IDS):
        fail("results.json.publication.withdrawnScoredJobIds", "withdrawn scored job IDs do not match the correction record")
    require_string(require_key(publication, "notice", "results.json.publication"), "results.json.publication.notice")

    parse_iso(require_key(results, "leaderboardSnapshotAt", "results.json"), "results.json.leaderboardSnapshotAt")
    campaign = require_dict(require_key(results, "campaign", "results.json"), "results.json.campaign")
    started = parse_iso(require_key(campaign, "startedAt", "results.json.campaign"), "results.json.campaign.startedAt")
    closed = parse_iso(require_key(campaign, "closedAt", "results.json.campaign"), "results.json.campaign.closedAt")
    if started >= closed:
        fail("results.json.campaign", "startedAt must precede closedAt")
    if closed != datetime(2026, 11, 5, 8, 0, tzinfo=timezone.utc):
        fail("results.json.campaign.closedAt", "expected the official 2026-11-05 08:00 UTC closure")
    if require_string(require_key(campaign, "status", "results.json.campaign"), "results.json.campaign.status") != "closed":
        fail("results.json.campaign.status", "expected 'closed'")
    if require_string(require_key(campaign, "gpu", "results.json.campaign"), "results.json.campaign.gpu") != "NVIDIA H200":
        fail("results.json.campaign.gpu", "expected 'NVIDIA H200'")
    if require_bool(require_key(campaign, "localGpuUsed", "results.json.campaign"), "results.json.campaign.localGpuUsed"):
        fail("results.json.campaign.localGpuUsed", "must be false for this CPU-only workflow")
    expected_campaign = {"runCount": 79, "scoredCount": 18, "checkCount": 61}
    for key, expected in expected_campaign.items():
        actual = require_int(require_key(campaign, key, "results.json.campaign"), f"results.json.campaign.{key}", minimum=0)
        if actual != expected:
            fail(f"results.json.campaign.{key}", f"expected {expected}, found {actual}")

    account = require_dict(require_key(results, "accountTotals", "results.json"), "results.json.accountTotals")
    if require_string(require_key(account, "date", "results.json.accountTotals"), "results.json.accountTotals.date") != "2026-11-05":
        fail("results.json.accountTotals.date", "expected '2026-11-05'")
    expected_account = {"runCount": 92, "scoredCount": 20, "checkCount": 72}
    for key, expected in expected_account.items():
        actual = require_int(require_key(account, key, "results.json.accountTotals"), f"results.json.accountTotals.{key}", minimum=0)
        if actual != expected:
            fail(f"results.json.accountTotals.{key}", f"expected {expected}, found {actual}")

    kernels = require_list(require_key(results, "kernels", "results.json"), "results.json.kernels")
    if len(kernels) != len(EXPECTED_KERNELS):
        fail("results.json.kernels", f"expected exactly {len(EXPECTED_KERNELS)} kernels")

    campaign_sums = {"scored": 0, "checks": 0, "runs": 0}
    account_sums = {"scored": 0, "checks": 0, "runs": 0}
    included_ids: list[str] = []
    for index, expected in enumerate(EXPECTED_KERNELS):
        kernel = require_dict(kernels[index], f"results.json.kernels[{index}]")
        path = f"results.json.kernels[{index}]"
        expected_id, expected_slug, expected_title, expected_speedup = expected
        for key, wanted in (("id", expected_id), ("slug", expected_slug), ("title", expected_title)):
            actual = require_string(require_key(kernel, key, path), f"{path}.{key}")
            if actual != wanted:
                fail(f"{path}.{key}", f"expected {wanted!r}, found {actual!r}")
        require_string(require_key(kernel, "summary", path), f"{path}.summary")
        included = require_bool(require_key(kernel, "includedInSummary", path), f"{path}.includedInSummary")
        expected_included = expected_id in INCLUDED_PROBLEM_IDS
        if included != expected_included:
            fail(f"{path}.includedInSummary", f"expected {expected_included}")
        if included:
            included_ids.append(expected_id)
            scored = validate_metric_block(require_key(kernel, "scored", path), f"{path}.scored", scored=True)
            validate_metric_block(require_key(kernel, "check", path), f"{path}.check", scored=False)
            scored_speedup = float(scored["speedup"])
            assert expected_speedup is not None
            if round(scored_speedup, 6) != expected_speedup:
                fail(f"{path}.scored.speedup", f"expected verified six-decimal value {expected_speedup:.6f}")
            if scored_speedup <= 1:
                fail(f"{path}.scored.speedup", "included scored result must beat baseline")
            validate_ranking_block(require_key(kernel, "ranking", path), f"{path}.ranking", scored_speedup)
            require_number(require_key(kernel, "firstCampaignScoredSpeedup", path), f"{path}.firstCampaignScoredSpeedup", positive=True)
        else:
            for field in ("scored", "check", "ranking", "firstCampaignScoredSpeedup"):
                if require_key(kernel, field, path) is not None:
                    fail(f"{path}.{field}", "excluded prefill active result fields must be null")
            historical = require_dict(require_key(kernel, "historical", path), f"{path}.historical")
            historical_scored = validate_metric_block(require_key(historical, "scored", f"{path}.historical"), f"{path}.historical.scored", scored=True)
            validate_metric_block(require_key(historical, "check", f"{path}.historical"), f"{path}.historical.check", scored=False)
            historical_speedup = float(historical_scored["speedup"])
            if round(historical_speedup, 6) != HISTORICAL_PREFILL_SPEEDUP:
                fail(f"{path}.historical.scored.speedup", f"expected historical value {HISTORICAL_PREFILL_SPEEDUP:.6f}")
            validate_ranking_block(require_key(historical, "ranking", f"{path}.historical"), f"{path}.historical.ranking", historical_speedup)
            require_number(require_key(historical, "firstCampaignScoredSpeedup", f"{path}.historical"), f"{path}.historical.firstCampaignScoredSpeedup", positive=True)

        for counts_key, totals, destination in (
            ("campaignCounts", expected_campaign, campaign_sums),
            ("accountCounts", expected_account, account_sums),
        ):
            counts = require_dict(require_key(kernel, counts_key, path), f"{path}.{counts_key}")
            scored_count = require_int(require_key(counts, "scored", f"{path}.{counts_key}"), f"{path}.{counts_key}.scored", minimum=0)
            check_count = require_int(require_key(counts, "checks", f"{path}.{counts_key}"), f"{path}.{counts_key}.checks", minimum=0)
            run_count = require_int(require_key(counts, "runs", f"{path}.{counts_key}"), f"{path}.{counts_key}.runs", minimum=0)
            if scored_count + check_count != run_count:
                fail(f"{path}.{counts_key}", "scored + checks must equal runs")
            destination["scored"] += scored_count
            destination["checks"] += check_count
            destination["runs"] += run_count
            if run_count > totals["runCount"]:
                fail(f"{path}.{counts_key}.runs", "per-kernel count exceeds scope total")

        source = require_dict(require_key(kernel, "source", path), f"{path}.source")
        expected_path = f"kernels/{expected_id}/submission.py"
        if require_string(require_key(source, "path", f"{path}.source"), f"{path}.source.path") != expected_path:
            fail(f"{path}.source.path", f"expected {expected_path!r}")
        digest = require_string(require_key(source, "sha256", f"{path}.source"), f"{path}.source.sha256")
        if not re.fullmatch(r"[0-9a-f]{64}", digest):
            fail(f"{path}.source.sha256", "expected a lowercase 64-character SHA-256")
        require_string(require_key(source, "pairedCheckJobId", f"{path}.source"), f"{path}.source.pairedCheckJobId")
        license_text = require_string(require_key(source, "license", f"{path}.source"), f"{path}.source.license")
        originality = require_string(require_key(source, "originalityStatus", f"{path}.source"), f"{path}.source.originalityStatus")
        reused = require_key(source, "reusedPublicSource", f"{path}.source")
        if included:
            if originality != "not-comprehensively-audited":
                fail(f"{path}.source.originalityStatus", "included results must remain explicitly unaudited")
            if reused is not None:
                fail(f"{path}.source.reusedPublicSource", "unaudited sources must use null, not a false originality guarantee")
        else:
            if originality != "confirmed-verbatim-copy":
                fail(f"{path}.source.originalityStatus", "prefill must be marked confirmed-verbatim-copy")
            if require_bool(reused, f"{path}.source.reusedPublicSource") is not True:
                fail(f"{path}.source.reusedPublicSource", "prefill copied-source reuse must be true")
            if "apache" not in license_text.lower():
                fail(f"{path}.source.license", "prefill license record must retain Apache attribution")
        if require_string(require_key(source, "provenancePath", f"{path}.source"), f"{path}.source.provenancePath") != "PROVENANCE.md":
            fail(f"{path}.source.provenancePath", "expected 'PROVENANCE.md'")

        story = require_dict(require_key(kernel, "story", path), f"{path}.story")
        for section in STORY_SECTIONS:
            paragraphs = require_list(require_key(story, section, f"{path}.story"), f"{path}.story.{section}")
            if not paragraphs:
                fail(f"{path}.story.{section}", "must include at least one narrative paragraph")
            for paragraph_index, paragraph in enumerate(paragraphs):
                require_string(paragraph, f"{path}.story.{section}[{paragraph_index}]")

        references = require_list(require_key(kernel, "references", path), f"{path}.references")
        if not references:
            fail(f"{path}.references", "must include at least one research or source link")
        for reference_index, reference_value in enumerate(references):
            reference = require_dict(reference_value, f"{path}.references[{reference_index}]")
            require_string(require_key(reference, "label", f"{path}.references[{reference_index}]"), f"{path}.references[{reference_index}].label")
            require_https(require_key(reference, "url", f"{path}.references[{reference_index}]"), f"{path}.references[{reference_index}].url")

    if tuple(included_ids) != INCLUDED_PROBLEM_IDS:
        fail("results.json.kernels[*].includedInSummary", "exactly the four non-prefill problems must be included")
    if campaign_sums != {"scored": 18, "checks": 61, "runs": 79}:
        fail("results.json.kernels[*].campaignCounts", f"aggregate must remain 18 scored, 61 checks, 79 historical runs; found {campaign_sums}")
    if account_sums != {"scored": 20, "checks": 72, "runs": 92}:
        fail("results.json.kernels[*].accountCounts", f"aggregate must remain 20 scored, 72 checks, 92 account runs; found {account_sums}")
    return results


def validate_runs(raw: Any, results: dict[str, Any]) -> dict[str, Any]:
    payload = require_dict(raw, "runs.json")
    if require_key(payload, "schemaVersion", "runs.json") != 1:
        fail("runs.json.schemaVersion", "only schema version 1 is supported")
    if require_string(require_key(payload, "scope", "runs.json"), "runs.json.scope") != "five-agent-campaign":
        fail("runs.json.scope", "expected 'five-agent-campaign'")
    runs = require_list(require_key(payload, "runs", "runs.json"), "runs.json.runs")
    if len(runs) != 79:
        fail("runs.json.runs", f"expected exactly 79 historical campaign records, found {len(runs)}")

    ids: set[str] = set()
    mode_counts = {"run": 0, "check": 0}
    problem_counts = {kernel_id: {"scored": 0, "checks": 0, "runs": 0} for kernel_id in EXPECTED_BY_ID}
    run_by_id: dict[str, dict[str, Any]] = {}
    previous_submitted: datetime | None = None
    for index, value in enumerate(runs):
        path = f"runs.json.runs[{index}]"
        run = require_dict(value, path)
        run_id = require_string(require_key(run, "id", path), f"{path}.id")
        if run_id in ids:
            fail(f"{path}.id", f"duplicate job ID {run_id!r}")
        ids.add(run_id)
        run_by_id[run_id] = run
        problem = require_string(require_key(run, "problem", path), f"{path}.problem")
        if problem not in EXPECTED_BY_ID:
            fail(f"{path}.problem", f"unexpected problem ID {problem!r}")
        mode = require_string(require_key(run, "mode", path), f"{path}.mode")
        if mode not in mode_counts:
            fail(f"{path}.mode", "expected 'check' or 'run'")
        mode_counts[mode] += 1
        problem_counts[problem]["runs"] += 1
        problem_counts[problem]["scored" if mode == "run" else "checks"] += 1
        submitted = parse_iso(require_key(run, "submittedAt", path), f"{path}.submittedAt")
        if previous_submitted is not None and submitted < previous_submitted:
            fail(path, "records must be sorted chronologically by submittedAt")
        previous_submitted = submitted
        finished_value = require_key(run, "finishedAt", path)
        if finished_value is not None:
            finished = parse_iso(finished_value, f"{path}.finishedAt")
            if finished < submitted:
                fail(f"{path}.finishedAt", "cannot precede submittedAt")
        require_string(require_key(run, "status", path), f"{path}.status")
        passed = require_key(run, "passed", path)
        if passed is not None:
            require_bool(passed, f"{path}.passed")
        speedup = require_number(require_key(run, "speedup", path), f"{path}.speedup", nullable=True, positive=True)
        ci = validate_ci(require_key(run, "ci95", path), f"{path}.ci95", nullable=True)
        if speedup is not None and ci is not None and (speedup < float(ci[0]) - 1e-9 or speedup > float(ci[1]) + 1e-9):
            fail(path, "reported speedup must fall within ci95")
        require_number(require_key(run, "scoreUs", path), f"{path}.scoreUs", nullable=True, positive=True)
        require_number(require_key(run, "baselineUs", path), f"{path}.baselineUs", nullable=True, positive=True)

    if mode_counts != {"run": 18, "check": 61}:
        fail("runs.json.runs", f"expected unchanged history of 18 scored runs and 61 checks, found {mode_counts}")

    for kernel in results["kernels"]:
        kernel_id = kernel["id"]
        if problem_counts[kernel_id] != kernel["campaignCounts"]:
            fail(
                f"runs.json.runs[{kernel_id}]",
                f"counts {problem_counts[kernel_id]} do not match results.json campaignCounts {kernel['campaignCounts']}",
            )
        evidence = kernel if kernel["includedInSummary"] else kernel["historical"]
        for field, mode in (("scored", "run"), ("check", "check")):
            job_id = evidence[field]["jobId"]
            if job_id not in run_by_id:
                fail(f"results.json.kernels[{kernel_id}].{field}.jobId", f"job {job_id!r} is missing from runs.json")
            job = run_by_id[job_id]
            if job["problem"] != kernel_id or job["mode"] != mode:
                fail(f"results.json.kernels[{kernel_id}].{field}.jobId", "job problem or mode does not match")
            if job["passed"] is not True:
                fail(f"results.json.kernels[{kernel_id}].{field}.jobId", "referenced evidence job must be passing")
            if job["speedup"] is None:
                fail(f"results.json.kernels[{kernel_id}].{field}.jobId", "referenced evidence job must report a speedup")
            if not math.isclose(float(job["speedup"]), float(evidence[field]["speedup"]), rel_tol=0, abs_tol=1e-9):
                fail(f"results.json.kernels[{kernel_id}].{field}.speedup", "does not match the referenced historical run")
        paired_id = kernel["source"]["pairedCheckJobId"]
        if paired_id not in run_by_id:
            fail(f"results.json.kernels[{kernel_id}].source.pairedCheckJobId", f"job {paired_id!r} is missing from runs.json")
        paired = run_by_id[paired_id]
        if paired["problem"] != kernel_id or paired["mode"] != "check" or paired["passed"] is not True:
            fail(f"results.json.kernels[{kernel_id}].source.pairedCheckJobId", "must identify a passing check for this problem")
        for field, mode in (("scored", "run"), ("check", "check")):
            passing = [
                float(run["speedup"])
                for run in runs
                if run["problem"] == kernel_id
                and run["mode"] == mode
                and run["passed"] is True
                and run["speedup"] is not None
            ]
            if not passing:
                fail(f"runs.json.runs[{kernel_id}]", f"no passing {mode} with a reported speedup")
            if not math.isclose(max(passing), float(evidence[field]["speedup"]), rel_tol=0, abs_tol=1e-9):
                fail(f"results.json.kernels[{kernel_id}].{field}.speedup", f"is not the highest passing historical {mode} speedup")

    for job_id in WITHDRAWN_SCORED_JOB_IDS:
        if job_id not in run_by_id:
            fail("results.json.publication.withdrawnScoredJobIds", f"withdrawn job {job_id!r} is missing from unchanged run history")
        run = run_by_id[job_id]
        if run["problem"] != EXCLUDED_PROBLEM_ID or run["mode"] != "run" or run["passed"] is not True:
            fail("results.json.publication.withdrawnScoredJobIds", f"withdrawn job {job_id!r} is not a passing prefill scored record")
    prefill_scored_ids = {run["id"] for run in runs if run["problem"] == EXCLUDED_PROBLEM_ID and run["mode"] == "run"}
    if prefill_scored_ids != set(WITHDRAWN_SCORED_JOB_IDS):
        fail("runs.json.runs", "the two historical prefill scored records must exactly match the withdrawn IDs")
    return payload


def validate_withdrawal(raw: Any, results: dict[str, Any]) -> dict[str, Any]:
    evidence = require_dict(raw, "withdrawal.json")
    if require_key(evidence, "schemaVersion", "withdrawal.json") != 1:
        fail("withdrawal.json.schemaVersion", "only schema version 1 is supported")
    if require_string(require_key(evidence, "scope", "withdrawal.json"), "withdrawal.json.scope") != "prefill-scored-result-withdrawal":
        fail("withdrawal.json.scope", "unexpected withdrawal evidence scope")
    if require_string(require_key(evidence, "problemId", "withdrawal.json"), "withdrawal.json.problemId") != EXCLUDED_PROBLEM_ID:
        fail("withdrawal.json.problemId", f"expected {EXCLUDED_PROBLEM_ID!r}")
    actions = require_list(require_key(evidence, "actions", "withdrawal.json"), "withdrawal.json.actions")
    if len(actions) != len(WITHDRAWAL_ACTIONS):
        fail("withdrawal.json.actions", "expected exactly two withdrawal actions")
    for index, (expected_id, expected_time) in enumerate(WITHDRAWAL_ACTIONS):
        action = require_dict(actions[index], f"withdrawal.json.actions[{index}]")
        if require_string(require_key(action, "jobId", f"withdrawal.json.actions[{index}]"), f"withdrawal.json.actions[{index}].jobId") != expected_id:
            fail(f"withdrawal.json.actions[{index}].jobId", f"expected {expected_id!r}")
        if require_string(require_key(action, "action", f"withdrawal.json.actions[{index}]"), f"withdrawal.json.actions[{index}].action") != "scored-result-withdrawn":
            fail(f"withdrawal.json.actions[{index}].action", "expected 'scored-result-withdrawn'")
        if parse_iso(require_key(action, "confirmedAt", f"withdrawal.json.actions[{index}]"), f"withdrawal.json.actions[{index}].confirmedAt") != parse_iso(expected_time, f"contract.withdrawalActions[{index}]"):
            fail(f"withdrawal.json.actions[{index}].confirmedAt", f"expected {expected_time!r}")
    verification = require_dict(require_key(evidence, "verification", "withdrawal.json"), "withdrawal.json.verification")
    if parse_iso(require_key(verification, "verifiedAt", "withdrawal.json.verification"), "withdrawal.json.verification.verifiedAt") != parse_iso(results["publication"]["withdrawalVerifiedAt"], "results.json.publication.withdrawalVerifiedAt"):
        fail("withdrawal.json.verification.verifiedAt", "does not match results publication metadata")
    if require_string(require_key(verification, "result", "withdrawal.json.verification"), "withdrawal.json.verification.result") != "no-active-prefill-entry-remains":
        fail("withdrawal.json.verification.result", "expected 'no-active-prefill-entry-remains'")
    return evidence


def load_json(path: Path) -> Any:
    if not path.is_file():
        raise BuildError(f"missing required input: {path}")
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except json.JSONDecodeError as exc:
        raise BuildError(f"{path}: invalid JSON at line {exc.lineno}, column {exc.colno}: {exc.msg}") from exc


def e(value: Any) -> str:
    return html.escape(str(value), quote=True)


def fmt_number(value: float | int, places: int = 6) -> str:
    rendered = f"{float(value):.{places}f}".rstrip("0").rstrip(".")
    return rendered if rendered != "-0" else "0"


def fmt_speedup(value: float | int | None) -> str:
    return "—" if value is None else f"{float(value):.6f}×"


def fmt_us(value: float | int | None) -> str:
    return "—" if value is None else f"{fmt_number(value, 3)} µs"


def fmt_ci(value: list[Any] | None) -> str:
    if value is None:
        return "Not reported"
    return f"{float(value[0]):.6f}–{float(value[1]):.6f}×"


def fmt_date(value: str, *, include_seconds: bool = False) -> str:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)
    pattern = "%d %b %Y, %H:%M:%S UTC" if include_seconds else "%d %b %Y, %H:%M UTC"
    return parsed.strftime(pattern).lstrip("0")


def fmt_duration(start: str, end: str) -> str:
    start_dt = datetime.fromisoformat(start.replace("Z", "+00:00"))
    end_dt = datetime.fromisoformat(end.replace("Z", "+00:00"))
    seconds = int((end_dt - start_dt).total_seconds())
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    parts = []
    if hours:
        parts.append(f"{hours}h")
    if minutes:
        parts.append(f"{minutes}m")
    if seconds:
        parts.append(f"{seconds}s")
    return " ".join(parts)


def local(depth: int, route: str) -> str:
    return "../" * depth + route


def canonical(project: dict[str, Any], route: str) -> str:
    return project["siteUrl"].rstrip("/") + "/" + route.lstrip("/")


def github_file(project: dict[str, Any], path: str) -> str:
    return f"{project['repositoryUrl']}/blob/{project['branch']}/{quote(path, safe='/')}"


def json_ld(data: dict[str, Any]) -> str:
    return json.dumps(data, ensure_ascii=True, separators=(",", ":")).replace("</", "<\\/")


def included_kernels(results: dict[str, Any]) -> list[dict[str, Any]]:
    return [kernel for kernel in results["kernels"] if kernel["includedInSummary"]]


def excluded_kernel(results: dict[str, Any]) -> dict[str, Any]:
    matches = [kernel for kernel in results["kernels"] if not kernel["includedInSummary"]]
    if len(matches) != 1:
        raise BuildError("results.json must contain exactly one excluded kernel")
    return matches[0]


def layout(
    results: dict[str, Any],
    *,
    route: str,
    title: str,
    description: str,
    active: str,
    depth: int,
    content: str,
    body_class: str = "",
    noindex: bool = False,
) -> str:
    project = results["project"]
    publication = results["publication"]
    page_title = title if title == project["title"] else f"{title} · {project['title']}"
    canonical_url = canonical(project, route)
    prefix = "../" * depth
    robots = '<meta name="robots" content="noindex">' if noindex else '<meta name="robots" content="index,follow">'
    nav_items = (
        ("notes", "index.html", "Field notes"),
        ("methodology", "methodology.html", "Methodology"),
        ("experiments", "experiments.html", "Experiments"),
        ("disclosure", "disclosure.html", "Disclosure"),
    )
    nav_parts = []
    for key, href, label in nav_items:
        current = ' aria-current="page"' if key == active else ""
        nav_parts.append(f'<a href="{e(prefix + href)}"{current}>{e(label)}</a>')
    nav = "".join(nav_parts)
    closure = fmt_date(results["campaign"]["closedAt"])
    disclosure_href = "#disclosure-details" if active == "disclosure" else prefix + "disclosure.html"
    structured = {
        "@context": "https://schema.org",
        "@type": "WebPage",
        "name": page_title,
        "description": description,
        "url": canonical_url,
        "isPartOf": {
            "@type": "WebSite",
            "name": project["title"],
            "url": project["siteUrl"],
        },
        "creator": {"@type": "SoftwareApplication", "name": "AI-agent workflow"},
        "publisher": {"@type": "Person", "name": project["author"]},
    }
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{e(page_title)}</title>
  <meta name="description" content="{e(description)}">
  {robots}
  <meta name="theme-color" content="#102a33">
  <meta property="og:type" content="website">
  <meta property="og:title" content="{e(page_title)}">
  <meta property="og:description" content="{e(description)}">
  <meta property="og:url" content="{e(canonical_url)}">
  <meta property="og:site_name" content="{e(project['title'])}">
  <meta name="twitter:card" content="summary">
  <link rel="canonical" href="{e(canonical_url)}">
  <link rel="icon" href="{e(prefix + 'assets/favicon.svg')}" type="image/svg+xml">
  <link rel="stylesheet" href="{e(prefix + 'assets/style.css')}">
  <script type="application/ld+json">{json_ld(structured)}</script>
  <script defer src="{e(prefix + 'assets/site.js')}"></script>
</head>
<body class="{e(body_class)}">
  <a class="skip-link" href="#main">Skip to content</a>
  <header class="site-header">
    <div class="masthead shell">
      <a class="brand" href="{e(prefix + 'index.html')}" aria-label="Hopper Field Notes home">
        <span class="brand-mark" aria-hidden="true">HFN</span>
        <span><strong>Hopper Field Notes</strong><small>Corrected agentic research log</small></span>
      </a>
      <nav class="primary-nav" aria-label="Primary navigation">{nav}</nav>
      <a class="repo-link" href="{e(project['repositoryUrl'])}">Repository <span aria-hidden="true">↗</span></a>
    </div>
    <div class="closure-strip">
      <div class="shell"><strong>Track closed</strong><span>{e(closure)}. No new challenge runs are being started.</span><span>Four included rankings remain provisional.</span></div>
    </div>
  </header>
  <aside class="agentic-notice" data-agentic-disclosure aria-labelledby="agentic-disclosure-title">
    <div class="shell agentic-notice-grid">
      <span class="agentic-notice-mark" aria-hidden="true">!</span>
      <div>
        <p class="eyebrow light">Agentic authorship and originality correction</p>
        <h2 id="agentic-disclosure-title">This entire project was AI-agent-driven; one result was copied verbatim and withdrawn.</h2>
        <p>AI agents assembled or adapted kernel code, directed experiments and submissions, and wrote the documentation and website. The work is not independently human-authored or comprehensively originality-audited. Prefill submissions copied complete public kernels unchanged; the other four problems may contain additional copied or adapted code and are not certified original.</p>
      </div>
      <a class="button disclosure-button" href="{e(disclosure_href)}">Read full disclosure</a>
    </div>
  </aside>
  <main id="main" tabindex="-1">{content}</main>
  <footer class="site-footer">
    <div class="shell footer-grid">
      <div><p class="footer-title">Hopper Field Notes</p><p>Four included measured results, one copied-and-withdrawn archive, and the unchanged 79-attempt historical record.</p></div>
      <div><p class="footer-label">Evidence</p><ul class="footer-links">
        <li><a href="{e(prefix + 'disclosure.html')}">Website disclosure</a></li>
        <li><a href="{e(github_file(project, publication['disclosurePath']))}">Disclosure source</a></li>
        <li><a href="{e(github_file(project, 'PROVENANCE.md'))}">Provenance</a></li>
        <li><a href="{e(github_file(project, 'EXPERIMENTS.md'))}">Experiment log</a></li>
      </ul></div>
      <div><p class="footer-label">Publication status</p><p>Fully agentic; originality audit incomplete. Source hashes verify bytes, not independent authorship.</p><p class="footer-small">Corrected: <time datetime="{e(publication['correctedAt'])}">{e(fmt_date(publication['correctedAt'], include_seconds=True))}</time></p></div>
    </div>
  </footer>
</body>
</html>
"""


def paragraph_list(paragraphs: list[str]) -> str:
    return "".join(f"<p>{e(paragraph)}</p>" for paragraph in paragraphs)


def result_card(kernel: dict[str, Any], *, depth: int = 0) -> str:
    if not kernel["includedInSummary"]:
        raise BuildError(f"excluded kernel {kernel['id']} cannot render as an active result card")
    scored = kernel["scored"]
    ranking = kernel["ranking"]
    leader = float(ranking["bestSpeedup"])
    ours = float(scored["speedup"])
    ratio = min(100.0, max(0.0, ours / leader * 100.0))
    gap = leader - ours
    time_reduction = (1 - float(scored["scoreUs"]) / float(scored["baselineUs"])) * 100
    source_project = {"repositoryUrl": REPOSITORY_URL, "branch": "main"}
    return f"""
<article class="result-card" data-claimed-result="{e(kernel['id'])}">
  <div class="card-topline"><span class="kernel-index">{e(kernel['id'])}</span><span class="status-badge measured">Included measurement</span></div>
  <h3><a href="{e(local(depth, 'kernels/' + kernel['slug'] + '.html'))}">{e(kernel['title'])}</a></h3>
  <p>{e(kernel['summary'])}</p>
  <p class="originality-note"><strong>Originality status:</strong> not comprehensively audited; this source is not certified original.</p>
  <div class="headline-metric"><strong>{e(fmt_speedup(scored['speedup']))}</strong><span>best included scored measurement</span></div>
  <dl class="metric-pairs">
    <div><dt>Measured</dt><dd>{e(fmt_us(scored['scoreUs']))}</dd></div>
    <div><dt>Baseline</dt><dd>{e(fmt_us(scored['baselineUs']))}</dd></div>
    <div><dt>95% CI</dt><dd>{e(fmt_ci(scored['ci95']))}</dd></div>
    <div><dt>Provisional rank</dt><dd>#{e(ranking['rank'])} of {e(ranking['placed'])}</dd></div>
  </dl>
  <div class="comparison" role="img" aria-label="Included measured speedup {e(fmt_speedup(ours))}; provisional leader {e(fmt_speedup(leader))}">
    <div class="comparison-row"><span>Leader</span><span class="bar-track"><span class="bar-fill leader" style="width:100%"></span></span><strong>{e(fmt_speedup(leader))}</strong></div>
    <div class="comparison-row"><span>Measured</span><span class="bar-track"><span class="bar-fill ours" style="width:{ratio:.3f}%"></span></span><strong>{e(fmt_speedup(ours))}</strong></div>
  </div>
  <p class="card-note">{e(fmt_number(time_reduction, 2))}% lower measured time than baseline. Gap to the provisional leader: {e(fmt_speedup(gap))}. This performance record is separate from originality.</p>
  <div class="card-actions"><a class="text-link" href="{e(local(depth, 'kernels/' + kernel['slug'] + '.html'))}">Read the field note <span aria-hidden="true">→</span></a><a class="text-link quiet" href="{e(github_file(source_project, kernel['source']['path']))}">View published source <span aria-hidden="true">↗</span></a></div>
</article>"""


def build_index(results: dict[str, Any]) -> str:
    campaign = results["campaign"]
    account = results["accountTotals"]
    publication = results["publication"]
    included = included_kernels(results)
    excluded = excluded_kernel(results)
    cards = "".join(result_card(kernel) for kernel in included)
    notebook_parts = []
    for index, kernel in enumerate(results["kernels"], 1):
        if kernel["includedInSummary"]:
            notebook_parts.append(f"""<article class="notebook-card">
  <p class="eyebrow">Kernel {index:02d} · included measurement</p>
  <h3><a href="kernels/{e(kernel['slug'])}.html">{e(kernel['title'])}</a></h3>
  <p>{e(kernel['story']['intro'][0])}</p>
  <p class="lesson"><strong>Originality remains unaudited.</strong> Measurement and source hashes do not certify independent authorship.</p>
  <a class="text-link" href="kernels/{e(kernel['slug'])}.html">Open field note <span aria-hidden="true">→</span></a>
</article>""")
        else:
            notebook_parts.append(f"""<article class="notebook-card excluded-notebook">
  <p class="eyebrow">Kernel {index:02d} · copied / withdrawn archive</p>
  <h3><a href="kernels/{e(kernel['slug'])}.html">{e(kernel['title'])}</a></h3>
  <p>AI agents submitted complete public prefill kernels unchanged. Both scored entries were withdrawn, no active leaderboard row remains, and this problem is excluded from result summaries.</p>
  <p class="lesson"><strong>Historical evidence only.</strong> Measurements remain in the unchanged 79-run ledger; they are not presented as an authored optimization or achievement.</p>
  <a class="text-link" href="kernels/{e(kernel['slug'])}.html">Open correction archive <span aria-hidden="true">→</span></a>
</article>""")
    notebook_cards = "".join(notebook_parts)
    duration = fmt_duration(campaign["startedAt"], campaign["closedAt"])
    snapshot = fmt_date(results["leaderboardSnapshotAt"], include_seconds=True)
    content = f"""
<section class="hero shell" aria-labelledby="hero-title">
  <div class="hero-copy">
    <p class="eyebrow light">Muse Spark Open Kernel Challenge · Corrected agentic record</p>
    <h1 id="hero-title">Measured remotely.<br><em>Corrected in public.</em></h1>
    <p class="hero-lede">This AI-agent-driven campaign assembled or adapted kernel code on a CPU-only Linux host and benchmarked it remotely on an NVIDIA H200. Four measured results remain in the summary; copied prefill submissions were withdrawn and are preserved only as historical evidence.</p>
    <div class="hero-actions"><a class="button primary" href="#results">View included measurements</a><a class="button ghost" href="disclosure.html">Read the correction</a></div>
    <p class="hero-disclosure"><strong>Current summary:</strong> four measured baseline improvements, none certified original and none a leaderboard win. Prefill is excluded. Included rankings are provisional at the {e(snapshot)} snapshot.</p>
  </div>
  <aside class="hero-panel" aria-label="Corrected campaign at a glance">
    <p class="panel-kicker">Historical campaign window</p>
    <p class="window-time">{e(duration)}</p>
    <p>{e(fmt_date(campaign['startedAt']))}<br>to {e(fmt_date(campaign['closedAt']))}</p>
    <div class="hero-stat-grid">
      <div><strong>{len(included)}</strong><span>included results</span></div>
      <div><strong>{e(campaign['runCount'])}</strong><span>historical attempts</span></div>
      <div><strong>2</strong><span>withdrawn scored IDs</span></div>
      <div><strong>1</strong><span>excluded problem</span></div>
    </div>
    <p class="panel-foot"><span class="signal-dot" aria-hidden="true"></span> Withdrawal absence verified {e(fmt_date(publication['withdrawalVerifiedAt']))}</p>
  </aside>
</section>

<section class="context-band" aria-label="Publication context">
  <div class="shell context-grid">
    <div><span class="context-number">01</span><p><strong>Workflow:</strong> agents assembled or adapted code, ran experiments and made submissions.</p></div>
    <div><span class="context-number">02</span><p><strong>Confirmed:</strong> prefill submissions were complete verbatim copies and are withdrawn.</p></div>
    <div><span class="context-number">03</span><p><strong>Unknown:</strong> the other four problems have not had a comprehensive originality audit.</p></div>
  </div>
</section>

<section class="section shell" id="results" aria-labelledby="results-title">
  <div class="section-heading split-heading">
    <div><p class="eyebrow">Included measurement summary</p><h2 id="results-title">Four measured results. Zero originality certifications.</h2></div>
    <p>These cards report archived judge measurements for four problems. They do not claim independent authorship. Each source may contain copied or adapted code because originality review remains incomplete.</p>
  </div>
  <div class="results-grid">{cards}</div>
</section>

<section class="section shell excluded-result" data-excluded-result="{e(excluded['id'])}" aria-labelledby="excluded-result-title">
  <div class="excluded-result-copy">
    <p class="eyebrow">Excluded result · historical evidence only</p>
    <h2 id="excluded-result-title">Prefill was copied verbatim, not an optimization authored by this project.</h2>
    <p>AI agents submitted complete public kernels unchanged under scored job IDs <code>{e(publication['withdrawnScoredJobIds'][0])}</code> and <code>{e(publication['withdrawnScoredJobIds'][1])}</code>. Both entries were withdrawn, and no active prefill leaderboard row remained when verified at <time datetime="{e(publication['withdrawalVerifiedAt'])}">{e(fmt_date(publication['withdrawalVerifiedAt'], include_seconds=True))}</time>.</p>
    <p>Its source, licenses and historical measurements remain available as correction evidence. They are not included in result cards, rank comparisons, baseline-win counts, or achievement language.</p>
    <div class="inline-links"><a class="button dark" href="kernels/{e(excluded['slug'])}.html">Open prefill archive</a><a class="button outline" href="disclosure.html#confirmed-copy">Read copying evidence</a></div>
  </div>
  <dl class="excluded-result-facts"><div><dt>Publication disposition</dt><dd>Excluded</dd></div><div><dt>Scored entries</dt><dd>2 withdrawn</dd></div><div><dt>Active row</dt><dd>None verified</dd></div><div><dt>Originality status</dt><dd>Confirmed verbatim copy</dd></div></dl>
</section>

<section class="section ink-section" aria-labelledby="scope-title">
  <div class="shell scope-layout">
    <div>
      <p class="eyebrow light">History is not a current claim</p>
      <h2 id="scope-title">79 attempts remain in the ledger.<br>Only four results remain in summary.</h2>
      <p>The historical campaign still contains {e(campaign['scoredCount'])} scored runs and {e(campaign['checkCount'])} checks. Withdrawal and exclusion change publication disposition, not the underlying archived facts. Account totals remain separately disclosed.</p>
      <a class="button ghost" href="experiments.html">Inspect dispositions</a>
    </div>
    <div class="scope-comparison" aria-label="Historical campaign and account run totals">
      <article><span>Historical campaign</span><strong>{e(campaign['runCount'])}</strong><p>{e(campaign['scoredCount'])} scored + {e(campaign['checkCount'])} checks</p></article>
      <article><span>Account snapshot</span><strong>{e(account['runCount'])}</strong><p>{e(account['scoredCount'])} scored + {e(account['checkCount'])} checks</p></article>
      <p class="scope-delta">The prefill records remain in history but are excluded from achievements.</p>
    </div>
  </div>
</section>

<section class="section shell" aria-labelledby="notebooks-title">
  <div class="section-heading"><p class="eyebrow">Five problem records</p><h2 id="notebooks-title">Four measurement notes and one copied-result archive.</h2><p>Included articles retain measured evidence while stating that originality is unaudited. The prefill article is a correction archive without an active score ticket, rank, or baseline-win claim.</p></div>
  <div class="notebook-grid">{notebook_cards}</div>
</section>

<section class="section shell methodology-teaser" aria-labelledby="evidence-title">
  <div class="evidence-mark" aria-hidden="true">≠</div>
  <div><p class="eyebrow">Measurement is not originality</p><h2 id="evidence-title">A passing benchmark and matching hash do not establish authorship.</h2><p>Judge results support performance and correctness claims. Source hashes support byte identity. Neither proves code was independently created. The confirmed prefill copies and incomplete audit of the other four problems are therefore presented separately from measurement.</p><div class="inline-links"><a class="text-link" href="methodology.html">Read methodology <span aria-hidden="true">→</span></a><a class="text-link" href="disclosure.html">Read full disclosure <span aria-hidden="true">→</span></a></div></div>
</section>

<aside class="section shell disclosure-card correction-card" aria-labelledby="unknown-title">
  <div><p class="eyebrow">What remains unknown</p><h2 id="unknown-title">The other four problem sources may also contain copied or adapted code.</h2></div>
  <p>No comprehensive originality audit has certified them. This correction does not claim an organizer ruling, ban, disqualification, or direct private message to organizers.</p>
  <a class="button dark" href="disclosure.html#limits">Review disclosure limits</a>
</aside>
"""
    return layout(
        results,
        route="index.html",
        title=results["project"]["title"],
        description="Corrected fully agentic Open Kernel Challenge record: four unaudited measured results, one confirmed copied and withdrawn prefill archive, and all 79 historical attempts.",
        active="notes",
        depth=0,
        content=content,
        body_class="home-page",
    )


def build_methodology(results: dict[str, Any]) -> str:
    campaign = results["campaign"]
    account = results["accountTotals"]
    publication = results["publication"]
    source_rows_parts = []
    for kernel in results["kernels"]:
        source = kernel["source"]
        source_url = github_file(results["project"], source["path"])
        if kernel["includedInSummary"]:
            disposition = '<span class="disposition-badge retained-history">Included measurement</span><small>Originality not comprehensively audited</small>'
            jobs = f'<code>{e(kernel["scored"]["jobId"])}</code><small>scored evidence</small>'
        else:
            disposition = '<span class="disposition-badge withdrawn">Copied / withdrawn</span><small>Excluded from summaries</small>'
            jobs = "<br>".join(f'<code>{e(job_id)}</code>' for job_id in publication["withdrawnScoredJobIds"])
        source_rows_parts.append(f"""<tr>
  <th scope="row"><a href="kernels/{e(kernel['slug'])}.html">{e(kernel['title'])}</a></th>
  <td class="disposition-cell">{disposition}</td>
  <td>{jobs}</td>
  <td><a href="{e(source_url)}"><code>{e(source['sha256'])}</code></a><small>{e(source['originalityStatus'])}</small></td>
</tr>""")
    source_rows = "".join(source_rows_parts)
    references: dict[str, str] = {}
    for kernel in results["kernels"]:
        for reference in kernel["references"]:
            references.setdefault(reference["url"], reference["label"])
    reference_items = "".join(
        f'<li><a href="{e(url)}">{e(label)} <span aria-hidden="true">↗</span></a></li>'
        for url, label in references.items()
    )
    headline_values = ", ".join(e(fmt_speedup(kernel["scored"]["speedup"])) for kernel in included_kernels(results))
    content = f"""
<header class="page-hero shell">
  <p class="eyebrow">Measurement and originality methodology</p>
  <h1>What the evidence proves—and what it does not.</h1>
  <p>The campaign was fully AI-agent-driven. Agents assembled or adapted code, selected experiments, made hosted submissions, and produced this publication. H200 measurements can support performance claims; they cannot establish independent authorship.</p>
  <div class="page-meta"><span>Hardware: {e(campaign['gpu'])}</span><span>Historical attempts: {e(campaign['runCount'])}</span><span>Included results: 4</span><span>Originality audit: incomplete</span></div>
</header>

<section class="section shell" aria-labelledby="workflow-title">
  <div class="section-heading"><p class="eyebrow">Fully agentic workflow</p><h2 id="workflow-title">Agents performed the assembly, testing, submission, and publication workflow.</h2><p>This record is not presented as independently human-authored work.</p></div>
  <ol class="workflow">
    <li><span>1</span><div><h3>Assemble or adapt</h3><p>AI agents prepared kernel sources on a CPU-only Linux host. The process could include copied or adapted public code; only prefill has been confirmed through exact byte comparison.</p></div></li>
    <li><span>2</span><div><h3>Check remotely</h3><p>Agents sent candidates through hosted checks on NVIDIA H200. Correctness and timing evidence did not audit provenance or originality.</p></div></li>
    <li><span>3</span><div><h3>Submit scored runs</h3><p>Agents made the scored submissions. The historical ledger preserves all 18 scored attempts and 61 checks, including the two later-withdrawn prefill entries.</p></div></li>
    <li><span>4</span><div><h3>Correct the publication</h3><p>After exact copying was confirmed for prefill, both scored IDs were withdrawn and the website was changed to exclude that problem from result summaries.</p></div></li>
  </ol>
</section>

<section class="section paper-section" aria-labelledby="modes-title">
  <div class="shell two-column-prose">
    <div><p class="eyebrow">Measurement modes</p><h2 id="modes-title">Checks and scored runs remain historical facts.</h2></div>
    <div class="definition-stack">
      <article><h3><span class="mode-dot check" aria-hidden="true"></span>Check</h3><p>A hosted correctness and timing attempt. Checks remain off-board evidence. For the excluded prefill problem they are historical only, not support for a current result claim.</p></article>
      <article><h3><span class="mode-dot scored" aria-hidden="true"></span>Scored run</h3><p>The four included six-decimal measurements are {headline_values}. Prefill's two scored records remain in history but are withdrawn and excluded from every current summary.</p></article>
    </div>
  </div>
</section>

<section class="section shell originality-method" aria-labelledby="originality-title">
  <div class="section-heading split-heading"><div><p class="eyebrow">Originality status</p><h2 id="originality-title">Hash equality can prove copying; hash uniqueness cannot prove originality.</h2></div><p>The prefill sources matched pinned public kernels byte-for-byte, confirming complete copying. The other four published files have not received a comprehensive comparison against public code and may contain additional copied or adapted material.</p></div>
  <div class="principle-grid">
    <article><strong>Confirmed for prefill</strong><p>Complete public kernels were submitted unchanged. This is described as verbatim copying, not as an optimization authored by the project.</p></article>
    <article><strong>Unknown for four</strong><p>Source hashes, correctness checks and performance measurements do not certify those files as independently created.</p></article>
    <article><strong>License is separate</strong><p>Permission under a software license is distinct from challenge eligibility, originality expectations, or any organizer decision.</p></article>
  </div>
</section>

<section class="section ink-section" id="scope" aria-labelledby="scope-method-title">
  <div class="shell">
    <div class="section-heading split-heading light"><div><p class="eyebrow light">Publication scopes</p><h2 id="scope-method-title">History stays complete while the result summary shrinks.</h2></div><p>The campaign began {e(fmt_date(campaign['startedAt'], include_seconds=True))} and closed {e(fmt_date(campaign['closedAt'], include_seconds=True))}. Withdrawal changes current publication status, not the archived run facts.</p></div>
    <div class="table-scroll light-table" role="region" aria-label="Historical and corrected publication scope comparison" tabindex="0">
      <table><thead><tr><th scope="col">Scope</th><th scope="col">Scored</th><th scope="col">Checks</th><th scope="col">Total / count</th><th scope="col">Publication meaning</th></tr></thead>
      <tbody><tr><th scope="row">Five-problem campaign history</th><td>{e(campaign['scoredCount'])}</td><td>{e(campaign['checkCount'])}</td><td>{e(campaign['runCount'])}</td><td>All attempts preserved</td></tr>
      <tr><th scope="row">5 Nov account snapshot</th><td>{e(account['scoredCount'])}</td><td>{e(account['checkCount'])}</td><td>{e(account['runCount'])}</td><td>Aggregate disclosure only</td></tr>
      <tr><th scope="row">Corrected result summary</th><td colspan="2">Four included problems</td><td>1 excluded</td><td>Prefill copied and withdrawn</td></tr></tbody></table>
    </div>
  </div>
</section>

<section class="section shell" id="provenance" aria-labelledby="provenance-title">
  <div class="section-heading"><p class="eyebrow">Source chain and publication status</p><h2 id="provenance-title">Every file stays available; not every file supports a result claim.</h2><p>SHA-256 values preserve byte identity. They do not certify originality. The table therefore presents source evidence alongside correction disposition.</p></div>
  <div class="table-scroll" role="region" aria-label="Kernel publication and source evidence matrix" tabindex="0">
    <table class="source-table corrected-source-table"><thead><tr><th scope="col">Kernel</th><th scope="col">Publication disposition</th><th scope="col">Scored evidence</th><th scope="col">Published source SHA-256 / originality</th></tr></thead><tbody>{source_rows}</tbody></table>
  </div>
  <aside class="attribution-note correction-attribution"><h3>Confirmed prefill copying and withdrawal</h3><p>The prefill scored submissions copied complete public kernels unchanged. Both scored IDs were withdrawn, and no active row remained when verified at <time datetime="{e(publication['withdrawalVerifiedAt'])}">{e(fmt_date(publication['withdrawalVerifiedAt'], include_seconds=True))}</time>. This publication does not claim an organizer ruling, ban, disqualification, or direct private organizer contact.</p><a class="text-link" href="disclosure.html#confirmed-copy">Read the complete correction <span aria-hidden="true">→</span></a></aside>
</section>

<section class="section shell" aria-labelledby="limits-title" id="limits">
  <div class="section-heading"><p class="eyebrow">Limitations</p><h2 id="limits-title">Boundaries on every remaining claim.</h2></div>
  <ul class="limitations-list">
    <li><strong>Originality incomplete.</strong> The four included sources are not certified original and may include copied or adapted public material.</li>
    <li><strong>Hardware specificity.</strong> Measurements come from the hosted NVIDIA H200 path; they are not local benchmarks or guarantees for other accelerators.</li>
    <li><strong>Workload specificity.</strong> Results apply to judged shapes and runtime conditions, not every workload or deployment.</li>
    <li><strong>Leaderboard timing.</strong> The four included ranks are provisional values captured at <time datetime="{e(results['leaderboardSnapshotAt'])}">{e(fmt_date(results['leaderboardSnapshotAt'], include_seconds=True))}</time>.</li>
    <li><strong>No organizer conclusion.</strong> Withdrawal and disclosure are actions recorded by this project, not a published organizer eligibility decision.</li>
    <li><strong>Historical selection.</strong> All 79 campaign attempts remain, including excluded prefill history and the two withdrawn scored IDs.</li>
  </ul>
</section>

<section class="section shell references-section" aria-labelledby="references-title">
  <div><p class="eyebrow">Research and source links</p><h2 id="references-title">Public material used or cited by the agentic workflow.</h2></div>
  <ul class="reference-list">{reference_items}</ul>
</section>
"""
    return layout(
        results,
        route="methodology.html",
        title="Methodology and originality limits",
        description="How the fully agentic workflow produced H200 measurements, why measurement does not prove originality, and how the copied prefill result was withdrawn and excluded.",
        active="methodology",
        depth=0,
        content=content,
        body_class="methodology-page",
    )


def run_outcome(run: dict[str, Any]) -> tuple[str, str]:
    if run["passed"] is True:
        return "pass", "Pass"
    if run["passed"] is False:
        return "fail", "Fail"
    return "unknown", "Not reported"


def run_disposition(run: dict[str, Any], publication: dict[str, Any]) -> tuple[str, str, str]:
    if run["id"] in publication["withdrawnScoredJobIds"]:
        return "withdrawn", "Withdrawn scored entry", "Copied prefill submission; historical measurement only"
    if run["problem"] in publication["excludedProblemIds"]:
        return "excluded-history", "Excluded history", "Prefill record excluded from result summaries"
    return "retained-history", "Included-problem history", "Measurement retained; originality not comprehensively audited"


def build_experiments(results: dict[str, Any], runs_payload: dict[str, Any]) -> str:
    runs = runs_payload["runs"]
    publication = results["publication"]
    title_by_id = {kernel["id"]: kernel["title"] for kernel in results["kernels"]}
    slug_by_id = {kernel["id"]: kernel["slug"] for kernel in results["kernels"]}
    options = "".join(f'<option value="{e(kernel_id)}">{e(title)}</option>' for kernel_id, title in title_by_id.items())
    rows = []
    for index, run in enumerate(runs, 1):
        outcome_key, outcome_label = run_outcome(run)
        disposition_key, disposition_label, disposition_note = run_disposition(run, publication)
        mode_label = "Scored" if run["mode"] == "run" else "Check"
        finished = (
            f'<time datetime="{e(run["finishedAt"])}">{e(fmt_date(run["finishedAt"], include_seconds=True))}</time>'
            if run["finishedAt"]
            else "—"
        )
        rows.append(
            f"""<tr data-run-row data-problem="{e(run['problem'])}" data-mode="{e(run['mode'])}" data-outcome="{e(outcome_key)}" data-result-disposition="{e(disposition_key)}">
  <td class="row-number">{index}</td>
  <td><time datetime="{e(run['submittedAt'])}">{e(fmt_date(run['submittedAt'], include_seconds=True))}</time></td>
  <th scope="row"><a href="kernels/{e(slug_by_id[run['problem']])}.html">{e(title_by_id[run['problem']])}</a><code>{e(run['problem'])}</code></th>
  <td><span class="mode-badge {e(run['mode'])}">{e(mode_label)}</span></td>
  <td><span class="outcome-badge {e(outcome_key)}">{e(outcome_label)}</span><small>{e(run['status'])}</small></td>
  <td class="disposition-cell"><span class="disposition-badge {e(disposition_key)}">{e(disposition_label)}</span><small>{e(disposition_note)}</small></td>
  <td class="numeric">{e(fmt_speedup(run['speedup']))}</td>
  <td class="numeric">{e(fmt_ci(run['ci95']))}</td>
  <td class="numeric">{e(fmt_us(run['scoreUs']))}</td>
  <td class="numeric">{e(fmt_us(run['baselineUs']))}</td>
  <td><code>{e(run['id'])}</code><details><summary>Finished</summary>{finished}</details></td>
</tr>"""
        )
    campaign = results["campaign"]
    account = results["accountTotals"]
    content = f"""
<header class="page-hero shell experiments-hero">
  <p class="eyebrow">Unchanged historical attempt ledger</p>
  <h1>All 79 attempts remain.<br>Their publication status changed.</h1>
  <p>The raw run dataset is unchanged: 18 scored runs and 61 checks. Rows now distinguish included-problem history, excluded prefill history, and the two withdrawn scored entries. Historical measurements are evidence, not current achievement claims or proof of originality.</p>
  <div class="page-meta"><span>{e(campaign['runCount'])} historical attempts</span><span>2 withdrawn scored entries</span><span>1 excluded problem</span><span>4 included problems</span></div>
</header>

<section class="section shell compact-top" aria-labelledby="ledger-title" data-experiment-ledger>
  <div class="ledger-intro"><div><p class="eyebrow">Filter the record</p><h2 id="ledger-title">Experiment history and disposition</h2></div><p>This is the five-problem campaign history, not the {e(account['runCount'])}-run account total. Filters run locally and never alter the underlying 79 rows.</p></div>
  <form class="filter-panel" data-filter-form aria-label="Filter experiment history">
    <div class="filter-field"><label for="problem-filter">Problem</label><select id="problem-filter" name="problem" data-filter="problem"><option value="all">All five problem histories</option>{options}</select></div>
    <div class="filter-field"><label for="mode-filter">Mode</label><select id="mode-filter" name="mode" data-filter="mode"><option value="all">Scored and checks</option><option value="run">Scored only</option><option value="check">Checks only</option></select></div>
    <div class="filter-field"><label for="outcome-filter">Pass / fail</label><select id="outcome-filter" name="outcome" data-filter="outcome"><option value="all">Every outcome</option><option value="pass">Pass</option><option value="fail">Fail</option><option value="unknown">Not reported</option></select></div>
    <button class="button filter-reset" type="reset">Reset filters</button>
  </form>
  <div class="ledger-status"><p data-filter-summary aria-live="polite" aria-atomic="true">Showing all {len(runs)} attempts.</p><p><span class="legend-key"><span class="mode-dot scored" aria-hidden="true"></span>Scored</span><span class="legend-key"><span class="mode-dot check" aria-hidden="true"></span>Check</span></p></div>
  <noscript><p class="noscript-note">JavaScript is off, so filters are unavailable. The complete 79-row ledger and all disposition labels remain visible below.</p></noscript>
  <div class="no-match" data-no-match hidden><h3>No attempts match those filters.</h3><p>Reset one or more controls to return to the complete historical record.</p></div>
  <div class="table-scroll ledger-scroll" role="region" aria-label="All historical campaign attempts and publication dispositions; scroll horizontally for more columns" tabindex="0">
    <table class="ledger-table corrected-ledger">
      <caption>Chronological five-agent campaign history: {e(campaign['runCount'])} unchanged attempts from {e(fmt_date(campaign['startedAt'], include_seconds=True))} through track closure, annotated with current publication disposition.</caption>
      <thead><tr><th scope="col">#</th><th scope="col">Submitted</th><th scope="col">Problem</th><th scope="col">Mode</th><th scope="col">Outcome</th><th scope="col">Publication disposition</th><th scope="col">Speedup</th><th scope="col">95% CI</th><th scope="col">Score</th><th scope="col">Baseline</th><th scope="col">Job</th></tr></thead>
      <tbody>{''.join(rows)}</tbody>
    </table>
  </div>
</section>

<aside class="section shell ledger-note corrected-ledger-note" aria-labelledby="ledger-note-title">
  <div><p class="eyebrow">Reading corrected history</p><h2 id="ledger-note-title">Preserved measurements do not revive a withdrawn result.</h2></div>
  <p>Rows marked <strong>withdrawn</strong> are the two copied prefill scored submissions. Other prefill rows are excluded history. Rows for the other four problems remain measurement history, but those sources are not comprehensively originality-audited.</p>
  <a class="text-link" href="disclosure.html#historical-record">Read the preservation policy <span aria-hidden="true">→</span></a>
</aside>
"""
    return layout(
        results,
        route="experiments.html",
        title="Corrected experiment history",
        description="All 79 historical attempts with visible withdrawn, excluded-history, and retained-history publication dispositions; raw measurements remain unchanged.",
        active="experiments",
        depth=0,
        content=content,
        body_class="experiments-page",
    )


def article_display_title(kernel: dict[str, Any]) -> str:
    return kernel["title"] if kernel["includedInSummary"] else f"{kernel['title']} archive"


def article_pagination(results: dict[str, Any], index: int) -> str:
    previous_kernel = results["kernels"][index - 1] if index > 0 else None
    next_kernel = results["kernels"][index + 1] if index + 1 < len(results["kernels"]) else None
    previous_link = (
        f'<a rel="prev" href="{e(previous_kernel["slug"])}.html"><span>Previous note</span><strong>← {e(article_display_title(previous_kernel))}</strong></a>'
        if previous_kernel
        else '<span aria-hidden="true"></span>'
    )
    next_link = (
        f'<a rel="next" href="{e(next_kernel["slug"])}.html"><span>Next note</span><strong>{e(article_display_title(next_kernel))} →</strong></a>'
        if next_kernel
        else '<a href="../index.html#results"><span>Return</span><strong>Included results →</strong></a>'
    )
    return f'<nav class="article-pagination shell" aria-label="Kernel article pagination">{previous_link}{next_link}</nav>'


def build_excluded_kernel_article(results: dict[str, Any], kernel: dict[str, Any], index: int) -> str:
    publication = results["publication"]
    source = kernel["source"]
    project = results["project"]
    source_url = github_file(project, source["path"])
    provenance_url = github_file(project, source["provenancePath"])
    disclosure_url = github_file(project, publication["disclosurePath"])
    references = "".join(
        f'<li><a href="{e(reference["url"])}">{e(reference["label"])} <span aria-hidden="true">↗</span></a></li>'
        for reference in kernel["references"]
    )
    withdrawal_times = {action["jobId"]: action["confirmedAt"] for action in results["_withdrawalEvidence"]["actions"]}
    evidence_rows = "".join(
        f"""<tr>
  <th scope="row"><code>{e(job_id)}</code></th>
  <td><strong>{e(publisher)}</strong><br><a href="{e(upstream_url)}">Pinned public source <span aria-hidden="true">↗</span></a></td>
  <td><code>{e(digest)}</code></td>
  <td>Withdrawn <time datetime="{e(withdrawal_times[job_id])}">{e(fmt_date(withdrawal_times[job_id], include_seconds=True))}</time></td>
</tr>"""
        for job_id, digest, upstream_url, publisher in CONFIRMED_COPY_EVIDENCE
    )
    content = f"""
<nav class="breadcrumbs shell" aria-label="Breadcrumb"><ol><li><a href="../index.html">Field notes</a></li><li><span aria-current="page">Prefill copied / withdrawn archive</span></li></ol></nav>
<article class="kernel-article excluded-kernel-article">
  <header class="kernel-hero shell excluded-kernel-hero">
    <div class="kernel-heading">
      <p class="eyebrow">Correction archive · {e(kernel['id'])}</p>
      <h1>{e(kernel['title'])}: copied and withdrawn.</h1>
      <p class="article-deck">This was not an optimization authored by the project. AI agents submitted complete public kernels unchanged. Both scored entries were withdrawn and the problem is excluded from all active result summaries.</p>
      <div class="article-actions"><a class="button dark" href="../disclosure.html#confirmed-copy">Read full disclosure</a><a class="button outline" href="../experiments.html?problem={e(kernel['id'])}">Inspect historical rows</a></div>
    </div>
    <aside class="archive-ticket" aria-label="Prefill correction status">
      <p class="ticket-label">Publication disposition</p>
      <strong>Excluded</strong>
      <p>Two copied scored entries withdrawn</p>
      <dl><div><dt>Active leaderboard row</dt><dd>None verified</dd></div><div><dt>Originality</dt><dd>Confirmed verbatim copy</dd></div><div><dt>History retained</dt><dd>{e(kernel['campaignCounts']['runs'])} attempts</dd></div></dl>
      <p class="ticket-foot">Verified <time datetime="{e(publication['withdrawalVerifiedAt'])}">{e(fmt_date(publication['withdrawalVerifiedAt'], include_seconds=True))}</time></p>
    </aside>
  </header>

  <div class="article-layout shell">
    <aside class="article-rail" aria-label="Archive sections">
      <p class="rail-title">In this archive</p>
      <nav aria-label="Archive sections"><a href="#confirmed-copy">Confirmed copying</a><a href="#withdrawal-record">Withdrawal</a><a href="#historical-record">Historical record</a><a href="#provenance">Source evidence</a><a href="#eligibility">Limits</a></nav>
      <dl class="rail-counts"><div><dt>Historical runs</dt><dd>{e(kernel['campaignCounts']['runs'])}</dd></div><div><dt>Scored</dt><dd>{e(kernel['campaignCounts']['scored'])}</dd></div><div><dt>Checks</dt><dd>{e(kernel['campaignCounts']['checks'])}</dd></div></dl>
    </aside>
    <div class="article-body">
      <section id="confirmed-copy"><p class="section-number">01 · Confirmed copying</p><h2>Complete public kernels were submitted unchanged.</h2><p>Exact byte comparisons tied the two prefill scored submissions to pinned public sources. This is described plainly as verbatim copying, not as adaptation, inspiration, or an optimization authored by this project.</p><p>The earlier file was published by Madhavan113 and credits a vjeux base plus Madhavan113 modifications. The final integration was published by team keell0renz, whose NOTICE identifies Bohdan Agarkov; its source headers credit vjeux, alygg77, and ZhimaoL.</p><p>The workflow was AI-agent-driven. The agents selected and submitted the copied files; this page does not shift that conduct onto the upstream authors or claim their endorsement. Credit remains with those contributors.</p></section>
      <section id="withdrawal-record"><p class="section-number">02 · Withdrawal record</p><h2>Both scored entries were voluntarily withdrawn from the active leaderboard.</h2>
        <div class="table-scroll" role="region" aria-label="Confirmed copied prefill jobs and withdrawal actions" tabindex="0"><table class="copy-evidence-table"><thead><tr><th scope="col">Scored job</th><th scope="col">Exact public source</th><th scope="col">SHA-256</th><th scope="col">Action</th></tr></thead><tbody>{evidence_rows}</tbody></table></div>
        <p>The project voluntarily withdrew both scored entries. No active prefill row remained for the account when verified at <time datetime="{e(publication['withdrawalVerifiedAt'])}">{e(fmt_date(publication['withdrawalVerifiedAt'], include_seconds=True))}</time>.</p>
      </section>
      <section id="historical-record"><p class="section-number">03 · Historical record</p><h2>The 79-run dataset remains unchanged.</h2><p>Prefill's {e(kernel['campaignCounts']['runs'])} campaign attempts, including both scored measurements, remain in the public ledger so the correction does not falsify history. The two scored rows are labeled <strong>withdrawn</strong>; all other prefill rows are labeled <strong>excluded history</strong>.</p><p>Those measurements are not repeated here as a score ticket, rank, leader comparison, progress chart, baseline win, or project achievement.</p><a class="text-link" href="../experiments.html?problem={e(kernel['id'])}">View excluded historical rows <span aria-hidden="true">→</span></a></section>
      <section id="provenance"><p class="section-number">04 · Source and license evidence</p><h2>Preservation and licensing do not convert copying into authorship.</h2>
        <dl class="provenance-list"><div><dt>Published evidence file</dt><dd><a href="{e(source_url)}"><code>{e(source['path'])}</code></a></dd></div><div><dt>Published file SHA-256</dt><dd><code>{e(source['sha256'])}</code></dd></div><div><dt>Originality status</dt><dd>{e(source['originalityStatus'])}</dd></div><div><dt>License record</dt><dd>{e(source['license'])}</dd></div><div><dt>Disclosure source</dt><dd><a href="{e(disclosure_url)}">DISCLOSURE.md <span aria-hidden="true">↗</span></a></dd></div><div><dt>Withdrawal evidence</dt><dd><a href="../data/withdrawal.json">Public JSON <span aria-hidden="true">→</span></a></dd></div></dl>
        <p>Applicable licenses and notices remain preserved. License permission is a separate question from challenge eligibility, originality requirements, or organizer approval.</p><a class="text-link" href="{e(provenance_url)}">Open full provenance <span aria-hidden="true">↗</span></a>
      </section>
      <section id="eligibility"><p class="section-number">05 · What this correction does not claim</p><h2>No organizer ruling is asserted.</h2><p>The withdrawals were voluntary project actions. This publication does not claim a rule violation, ban, disqualification, final eligibility decision, or private organizer response. It also does not claim that a direct private message was sent to organizers. It records the project's withdrawals, verification, and public disclosure.</p></section>
      <section class="article-references" aria-labelledby="archive-references-title"><p class="section-number">Public evidence links</p><h2 id="archive-references-title">Pinned sources, notices, and problem material.</h2><ul class="reference-list">{references}</ul></section>
    </div>
  </div>
</article>
{article_pagination(results, index)}
"""
    return layout(
        results,
        route=f"kernels/{kernel['slug']}.html",
        title="Prefill copied and withdrawn archive",
        description="Correction archive for two confirmed verbatim-copy prefill submissions: both scored entries were withdrawn, no active result remains, and historical evidence is preserved without achievement claims.",
        active="notes",
        depth=1,
        content=content,
        body_class="kernel-page excluded-kernel-page",
    )


def build_kernel_article(results: dict[str, Any], kernel: dict[str, Any], index: int) -> str:
    if not kernel["includedInSummary"]:
        return build_excluded_kernel_article(results, kernel, index)
    scored = kernel["scored"]
    check = kernel["check"]
    ranking = kernel["ranking"]
    source = kernel["source"]
    project = results["project"]
    source_url = github_file(project, source["path"])
    provenance_url = github_file(project, source["provenancePath"])
    leader_gap = float(ranking["bestSpeedup"]) - float(scored["speedup"])
    improvement = (float(scored["speedup"]) / float(kernel["firstCampaignScoredSpeedup"]) - 1) * 100
    references = "".join(
        f'<li><a href="{e(reference["url"])}">{e(reference["label"])} <span aria-hidden="true">↗</span></a></li>'
        for reference in kernel["references"]
    )
    content = f"""
<nav class="breadcrumbs shell" aria-label="Breadcrumb"><ol><li><a href="../index.html">Field notes</a></li><li><span aria-current="page">{e(kernel['title'])}</span></li></ol></nav>
<article class="kernel-article included-kernel-article">
  <header class="kernel-hero shell">
    <div class="kernel-heading">
      <p class="eyebrow">Included measurement · {e(kernel['id'])}</p>
      <h1>{e(kernel['title'])}</h1>
      <p class="article-deck">{e(kernel['summary'])}</p>
      <p class="article-originality"><strong>Originality status:</strong> not comprehensively audited. The source may contain copied or adapted public code and is not certified original.</p>
      <div class="article-actions"><a class="button dark" href="{e(source_url)}">View published source <span aria-hidden="true">↗</span></a><a class="button outline" href="../experiments.html?problem={e(kernel['id'])}">Filter its history</a></div>
    </div>
    <aside class="result-ticket" aria-label="Included measured result summary">
      <p class="ticket-label">Included scored measurement</p>
      <strong class="ticket-speedup">{e(fmt_speedup(scored['speedup']))}</strong>
      <p>{e(fmt_us(scored['scoreUs']))} vs {e(fmt_us(scored['baselineUs']))} baseline</p>
      <dl><div><dt>95% CI</dt><dd>{e(fmt_ci(scored['ci95']))}</dd></div><div><dt>Rank</dt><dd>#{e(ranking['rank'])} / {e(ranking['placed'])}</dd></div><div><dt>Leader</dt><dd>{e(fmt_speedup(ranking['bestSpeedup']))}</dd></div><div><dt>Gap</dt><dd>{e(fmt_speedup(leader_gap))}</dd></div></dl>
      <p class="ticket-foot">Measurement retained; originality unaudited. Rank provisional at <time datetime="{e(results['leaderboardSnapshotAt'])}">{e(fmt_date(results['leaderboardSnapshotAt']))}</time></p>
    </aside>
  </header>

  <div class="article-layout shell">
    <aside class="article-rail" aria-label="Article summary">
      <p class="rail-title">In this note</p>
      <nav aria-label="Article sections"><a href="#problem">Problem</a><a href="#approach">Approach record</a><a href="#lessons">Lessons</a><a href="#failures">Failed ideas</a><a href="#caveats">Caveats</a><a href="#provenance">Provenance</a></nav>
      <dl class="rail-counts"><div><dt>Historical runs</dt><dd>{e(kernel['campaignCounts']['runs'])}</dd></div><div><dt>Scored</dt><dd>{e(kernel['campaignCounts']['scored'])}</dd></div><div><dt>Checks</dt><dd>{e(kernel['campaignCounts']['checks'])}</dd></div></dl>
    </aside>
    <div class="article-body">
      <section id="problem"><p class="section-number">01 · Problem record</p><h2>What the measured workload asked for.</h2>{paragraph_list(kernel['story']['intro'])}</section>
      <section id="approach"><p class="section-number">02 · Published approach record</p><h2>What the agentic workflow submitted.</h2>{paragraph_list(kernel['story']['approach'])}<p><strong>Originality caveat:</strong> this approach description is not a certification that the implementation was independently authored.</p></section>
      <aside class="measurement-callout" aria-label="Included measurement summary"><p class="eyebrow">Measurement evidence · originality unaudited</p><div><strong>{e(fmt_speedup(scored['speedup']))}</strong><span>included measurement</span><strong>{e(fmt_ci(scored['ci95']))}</strong><span>95% CI</span><strong>{e(fmt_number(improvement, 2))}%</strong><span>change from first scored record</span></div><p>Best passing check: {e(fmt_speedup(check['speedup']))} ({e(fmt_ci(check['ci95']))}). These values support measurement claims only.</p></aside>
      <section id="lessons"><p class="section-number">03 · Measured lessons</p><h2>What changed in the archived measurements.</h2>{paragraph_list(kernel['story']['lessons'])}</section>
      <section id="failures"><p class="section-number">04 · Recorded regressions</p><h2>Experiments that did not survive measurement.</h2>{paragraph_list(kernel['story']['failures'])}</section>
      <section id="caveats"><p class="section-number">05 · Boundaries</p><h2>Where this result claim stops.</h2>{paragraph_list(kernel['story']['caveats'])}<p>This source has not received a comprehensive originality audit and may contain copied or adapted public code.</p></section>
      <section id="provenance"><p class="section-number">06 · Source and provenance</p><h2>Exact bytes are evidence, not proof of authorship.</h2>
        <dl class="provenance-list"><div><dt>Published file</dt><dd><a href="{e(source_url)}"><code>{e(source['path'])}</code></a></dd></div><div><dt>SHA-256</dt><dd><code>{e(source['sha256'])}</code></dd></div><div><dt>Originality status</dt><dd>{e(source['originalityStatus'])}</dd></div><div><dt>Included scored job</dt><dd><code>{e(scored['jobId'])}</code></dd></div><div><dt>Highest passing check</dt><dd><code>{e(check['jobId'])}</code></dd></div><div><dt>Check paired to source</dt><dd><code>{e(source['pairedCheckJobId'])}</code></dd></div><div><dt>License record</dt><dd>{e(source['license'])}</dd></div></dl>
        <p>The highest passing campaign check can differ from the paired check. Neither matching job evidence nor a source hash certifies independent authorship.</p>
        <div class="inline-links"><a class="text-link" href="{e(provenance_url)}">Open full provenance <span aria-hidden="true">↗</span></a><a class="text-link" href="../disclosure.html#limits">Review originality limits <span aria-hidden="true">→</span></a></div>
      </section>
      <section class="article-references" aria-labelledby="article-references-title"><p class="section-number">Research and source links</p><h2 id="article-references-title">Public material cited by the agentic workflow.</h2><ul class="reference-list">{references}</ul></section>
    </div>
  </div>
</article>
{article_pagination(results, index)}
"""
    return layout(
        results,
        route=f"kernels/{kernel['slug']}.html",
        title=kernel["title"],
        description=f"AI-agent-produced, originality-unaudited measurement record for {kernel['title']}: {fmt_speedup(scored['speedup'])}, with source evidence, experiments, and limitations.",
        active="notes",
        depth=1,
        content=content,
        body_class="kernel-page",
    )


def build_disclosure(results: dict[str, Any]) -> str:
    publication = results["publication"]
    project = results["project"]
    prefill = excluded_kernel(results)
    withdrawal_times = {action["jobId"]: action["confirmedAt"] for action in results["_withdrawalEvidence"]["actions"]}
    evidence_rows = "".join(
        f"""<tr>
  <th scope="row"><code>{e(job_id)}</code></th>
  <td><strong>{e(publisher)}</strong><br><a href="{e(upstream_url)}">Pinned exact public source <span aria-hidden="true">↗</span></a></td>
  <td><code>{e(digest)}</code></td>
  <td><time datetime="{e(withdrawal_times[job_id])}">{e(fmt_date(withdrawal_times[job_id], include_seconds=True))}</time></td>
</tr>"""
        for job_id, digest, upstream_url, publisher in CONFIRMED_COPY_EVIDENCE
    )
    unaudited_cards = "".join(
        f"""<article><p class="eyebrow">Not comprehensively audited</p><h3><a href="kernels/{e(kernel['slug'])}.html">{e(kernel['title'])}</a></h3><p>This measured source remains in the summary, but may contain copied or adapted public code. It is not certified original.</p></article>"""
        for kernel in included_kernels(results)
    )
    disclosure_source = github_file(project, publication["disclosurePath"])
    provenance_source = github_file(project, "PROVENANCE.md")
    withdrawal_source = github_file(project, "data/withdrawal-evidence.json")
    content = f"""
<header class="page-hero shell disclosure-hero" id="disclosure-details">
  <p class="eyebrow">Public correction · {e(fmt_date(publication['correctedAt']))}</p>
  <h1>This project was fully agentic. One problem was copied outright.</h1>
  <p>AI agents assembled or adapted kernel code, directed experiments, made challenge submissions, and produced the documentation and website. The project is not independently human-authored and has not received a comprehensive originality audit.</p>
  <div class="page-meta"><span>Workflow: fully agentic</span><span>Originality audit: incomplete</span><span>Prefill: copied / withdrawn</span><span>Included results: 4 unaudited</span></div>
</header>

<section class="section shell disclosure-statement" aria-labelledby="statement-title">
  <div><p class="eyebrow">Disclosure statement</p><h2 id="statement-title">The correction applies to the whole workflow, not only one file.</h2></div>
  <blockquote>{e(publication['notice'])}</blockquote>
  <div class="disclosure-prose"><p>“Fully agentic” means the agents made material choices across code assembly and adaptation, experiment design, hosted checks, scored submissions, documentation, and site generation. Human ownership of the repository is not presented as independent human authorship of those outputs.</p><p>The confirmed copying finding is specific to prefill. The absence of a confirmed match for another problem is not an originality certification.</p></div>
</section>

<section class="section paper-section" id="confirmed-copy" aria-labelledby="copy-title">
  <div class="shell">
    <div class="section-heading"><p class="eyebrow">Confirmed verbatim copying</p><h2 id="copy-title">Two prefill scored submissions matched complete public kernels unchanged.</h2><p>This finding comes from exact byte comparison against pinned public sources, not from inference based on licensing or similar techniques.</p></div>
    <div class="table-scroll" role="region" aria-label="Confirmed copied prefill scored submissions" tabindex="0"><table class="copy-evidence-table"><thead><tr><th scope="col">Scored job</th><th scope="col">Exact public source</th><th scope="col">SHA-256</th><th scope="col">Withdrawn at</th></tr></thead><tbody>{evidence_rows}</tbody></table></div>
    <div class="disclosure-prose" id="source-authorship">
      <h3>Where it came from, and whose work it is</h3>
      <p>The earlier file came from <strong>Madhavan113's ParadigmKernelChallenge</strong> repository. Its source header credits <strong>vjeux</strong> for the base and <strong>Madhavan113</strong> for modifications.</p>
      <p>The final file came from <strong>team keell0renz's muse-open-kernel-challenge</strong> integration. The upstream NOTICE names <strong>Bohdan Agarkov</strong>, and the copied source headers credit <strong>vjeux, alygg77, and ZhimaoL</strong> for incorporated work.</p>
      <p>These repository owners are publishers or integrators, not necessarily sole authors. This project and its AI agents claim no authorship of the copied implementations and no endorsement from their contributors. The full <a href="{e(github_file(project, 'NOTICE'))}">upstream notices are preserved</a>.</p>
    </div>
    <div class="correction-status-grid"><article><span>Publication status</span><strong>Excluded</strong><p>The project voluntarily withdrew both scored entries. Prefill is absent from cards, rank comparisons, win counts, and achievement summaries.</p></article><article><span>Leaderboard verification</span><strong>No active row</strong><p>Verified at <time datetime="{e(publication['withdrawalVerifiedAt'])}">{e(fmt_date(publication['withdrawalVerifiedAt'], include_seconds=True))}</time>.</p></article><article><span>Historical evidence</span><strong>Preserved</strong><p>All {e(prefill['campaignCounts']['runs'])} prefill campaign attempts remain labeled in the unchanged ledger.</p></article></div>
    <a class="text-link" href="kernels/{e(prefill['slug'])}.html">Open the prefill correction archive <span aria-hidden="true">→</span></a>
  </div>
</section>

<section class="section shell" id="historical-record" aria-labelledby="history-title">
  <div class="section-heading split-heading"><div><p class="eyebrow">Historical preservation</p><h2 id="history-title">Withdrawal does not rewrite the 79-run record.</h2></div><p>The raw run JSON remains unchanged. Website annotations describe present publication disposition without changing pass flags, statuses, timings, confidence intervals, or attempt totals.</p></div>
  <div class="principle-grid"><article><strong>2 withdrawn rows</strong><p>The two prefill scored IDs are visibly marked withdrawn.</p></article><article><strong>8 excluded-history rows</strong><p>Other prefill checks remain accessible but do not support an active result.</p></article><article><strong>69 retained-history rows</strong><p>The other four problems remain measurement history with incomplete originality review.</p></article></div>
  <a class="button dark" href="experiments.html">Inspect all 79 dispositions</a>
</section>

<section class="section ink-section" id="limits" aria-labelledby="limits-title">
  <div class="shell disclosure-limits">
    <div><p class="eyebrow light">What remains unresolved</p><h2 id="limits-title">Four included problem sources may contain more copied or adapted code.</h2><p>No comprehensive originality audit has certified them. Matching local hashes and passing hosted tests establish byte identity and measurement, not independent creation.</p></div>
    <div class="unaudited-grid">{unaudited_cards}</div>
  </div>
</section>

<section class="section shell" aria-labelledby="license-title">
  <div class="section-heading"><p class="eyebrow">License, eligibility, and communication</p><h2 id="license-title">These are separate questions.</h2></div>
  <ul class="limitations-list">
    <li><strong>Licensing.</strong> A public license may permit copying or adaptation while still requiring attribution and notices.</li>
    <li><strong>Eligibility.</strong> License permission does not establish compliance with challenge originality or submission rules.</li>
    <li><strong>No organizer ruling claimed.</strong> This site does not assert a violation finding, ban, disqualification, or final eligibility decision by organizers.</li>
    <li><strong>No direct private contact claimed.</strong> This public correction is addressed to readers and organizers, but it does not claim that a private organizer message was sent or received.</li>
  </ul>
</section>

<section class="section shell disclosure-sources" aria-labelledby="sources-title">
  <div><p class="eyebrow">Repository records</p><h2 id="sources-title">Full disclosure and provenance documents.</h2></div>
  <div class="source-link-grid"><a href="{e(disclosure_source)}"><span>Disclosure</span><strong>DISCLOSURE.md ↗</strong></a><a href="{e(provenance_source)}"><span>Source record</span><strong>PROVENANCE.md ↗</strong></a><a href="data/withdrawal.json"><span>Published withdrawal evidence</span><strong>Public JSON →</strong></a><a href="{e(withdrawal_source)}"><span>Evidence source</span><strong>Repository JSON ↗</strong></a><a href="experiments.html"><span>Unchanged history</span><strong>79-run ledger →</strong></a><a href="kernels/{e(prefill['slug'])}.html"><span>Excluded problem</span><strong>Prefill archive →</strong></a></div>
</section>
"""
    return layout(
        results,
        route="disclosure.html",
        title="Agentic authorship and copying disclosure",
        description="Full disclosure that the project was AI-agent-driven, prefill submissions were confirmed verbatim copies and withdrawn, and the other four problem sources remain unaudited.",
        active="disclosure",
        depth=0,
        content=content,
        body_class="disclosure-page",
    )


def build_404(results: dict[str, Any], assets_dir: Path) -> str:
    content = """
<section class="not-found shell" aria-labelledby="not-found-title">
  <p class="error-code">404 · off the corrected record</p>
  <h1 id="not-found-title">This field note does not exist.</h1>
  <p>The corrected publication still provides four included measurement notes, one copied-and-withdrawn prefill archive, the full disclosure, and all 79 historical attempts.</p>
  <div class="hero-actions"><a class="button dark" href="index.html">Return to field notes</a><a class="button outline" href="disclosure.html">Read disclosure</a></div>
</section>
"""
    document = layout(
        results,
        route="404.html",
        title="Page not found",
        description="The requested corrected Hopper Field Notes page could not be found.",
        active="",
        depth=0,
        content=content,
        body_class="not-found-page",
        noindex=True,
    )
    # A missing URL may be arbitrarily deep, so its presentation is self-contained.
    css = read_asset(assets_dir, "style.css")
    if "</style" in css.lower():
        raise BuildError("site CSS cannot contain a closing style tag")
    icon = quote(read_asset(assets_dir, "favicon.svg"), safe="")
    document = document.replace(
        '<link rel="stylesheet" href="assets/style.css">',
        f"<style>{css}</style>",
    ).replace(
        '<link rel="icon" href="assets/favicon.svg" type="image/svg+xml">',
        f'<link rel="icon" href="data:image/svg+xml,{icon}" type="image/svg+xml">',
    ).replace(
        '<script defer src="assets/site.js"></script>',
        "",
    )
    for route in ("index.html", "methodology.html", "experiments.html", "disclosure.html"):
        document = document.replace(
            f'href="{route}"',
            f'href="{e(canonical(results["project"], route))}" data-site-route="{route}"',
        )
    config = json_ld({"path": urlparse(results["project"]["siteUrl"]).path.rstrip("/") + "/"})
    navigation = f"""<script>
(() => {{
  const publishedPath = {config}.path;
  const mount = window.location.pathname.startsWith(publishedPath) ? publishedPath : "/";
  document.querySelectorAll("[data-site-route]").forEach((link) => {{
    link.href = mount + link.dataset.siteRoute;
  }});
}})();
</script>
"""
    return document.replace("</body>", navigation + "</body>")


def build_sitemap(results: dict[str, Any]) -> str:
    routes = ["", "methodology.html", "experiments.html", "disclosure.html"] + [
        f"kernels/{kernel['slug']}.html" for kernel in results["kernels"]
    ]
    lastmod = results["publication"]["correctedAt"][:10]
    entries = "".join(
        f"  <url><loc>{e(results['project']['siteUrl'].rstrip('/') + '/' + route)}</loc><lastmod>{e(lastmod)}</lastmod></url>\n"
        for route in routes
    )
    return f'<?xml version="1.0" encoding="UTF-8"?>\n<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n{entries}</urlset>\n'


def write_text(path: Path, content: str) -> None:
    if path.suffix == ".html":
        content = re.sub(r"(?m)^[ \t]+$", "", content)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_text(encoding="utf-8") == content:
        return
    path.write_text(content, encoding="utf-8", newline="\n")


def read_asset(assets_dir: Path, filename: str) -> str:
    source = assets_dir / filename
    if not source.is_file():
        raise BuildError(f"missing required site asset: {source}")
    return source.read_text(encoding="utf-8")


def copy_assets(assets_dir: Path, output_dir: Path) -> None:
    for filename in ASSET_FILES:
        write_text(output_dir / "assets" / filename, read_asset(assets_dir, filename))


def build(data_dir: Path, output_dir: Path, assets_dir: Path) -> list[Path]:
    results = validate_results(load_json(data_dir / "results.json"))
    runs = validate_runs(load_json(data_dir / "runs.json"), results)
    results["_withdrawalEvidence"] = validate_withdrawal(load_json(data_dir / "withdrawal.json"), results)
    outputs: dict[Path, str] = {
        output_dir / "index.html": build_index(results),
        output_dir / "methodology.html": build_methodology(results),
        output_dir / "experiments.html": build_experiments(results, runs),
        output_dir / "disclosure.html": build_disclosure(results),
        output_dir / "404.html": build_404(results, assets_dir),
        output_dir / "sitemap.xml": build_sitemap(results),
        output_dir / "robots.txt": f"User-agent: *\nAllow: /\nSitemap: {results['project']['siteUrl']}sitemap.xml\n",
        output_dir / ".nojekyll": "",
    }
    for index, kernel in enumerate(results["kernels"]):
        outputs[output_dir / "kernels" / f"{kernel['slug']}.html"] = build_kernel_article(results, kernel, index)
    for path, content in outputs.items():
        write_text(path, content)
    copy_assets(assets_dir, output_dir)
    return [*outputs, *(output_dir / "assets" / filename for filename in ASSET_FILES)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR, help="directory containing results.json and runs.json")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="directory for generated static files")
    parser.add_argument("--assets-dir", type=Path, default=DEFAULT_ASSETS_DIR, help="directory containing source site assets")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        outputs = build(args.data_dir.resolve(), args.output_dir.resolve(), args.assets_dir.resolve())
    except BuildError as exc:
        print(f"build_site.py: error: {exc}", file=sys.stderr)
        return 1
    print(f"Built {len(outputs)} static files in {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
