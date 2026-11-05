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
    ("muse_prefill_layer", "prefill-layer", "Prefill layer", 1.919190),
)
EXPECTED_BY_ID = {item[0]: item for item in EXPECTED_KERNELS}
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


def validate_results(raw: Any) -> dict[str, Any]:
    results = require_dict(raw, "results.json")
    if require_key(results, "schemaVersion", "results.json") != 1:
        fail("results.json.schemaVersion", "only schema version 1 is supported")

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

    snapshot = parse_iso(
        require_key(results, "leaderboardSnapshotAt", "results.json"),
        "results.json.leaderboardSnapshotAt",
    )
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
    for index, expected in enumerate(EXPECTED_KERNELS):
        kernel = require_dict(kernels[index], f"results.json.kernels[{index}]")
        path = f"results.json.kernels[{index}]"
        expected_id, expected_slug, expected_title, expected_speedup = expected
        for key, wanted in (("id", expected_id), ("slug", expected_slug), ("title", expected_title)):
            actual = require_string(require_key(kernel, key, path), f"{path}.{key}")
            if actual != wanted:
                fail(f"{path}.{key}", f"expected {wanted!r}, found {actual!r}")
        require_string(require_key(kernel, "summary", path), f"{path}.summary")
        scored = validate_metric_block(require_key(kernel, "scored", path), f"{path}.scored", scored=True)
        validate_metric_block(require_key(kernel, "check", path), f"{path}.check", scored=False)
        scored_speedup = float(scored["speedup"])
        if round(scored_speedup, 6) != expected_speedup:
            fail(f"{path}.scored.speedup", f"expected verified six-decimal value {expected_speedup:.6f}")
        if scored_speedup <= 1:
            fail(f"{path}.scored.speedup", "all retained scored kernels must beat baseline")

        ranking = require_dict(require_key(kernel, "ranking", path), f"{path}.ranking")
        rank = require_int(require_key(ranking, "rank", f"{path}.ranking"), f"{path}.ranking.rank", minimum=1)
        placed = require_int(require_key(ranking, "placed", f"{path}.ranking"), f"{path}.ranking.placed", minimum=1)
        leader = require_number(require_key(ranking, "bestSpeedup", f"{path}.ranking"), f"{path}.ranking.bestSpeedup", positive=True)
        assert leader is not None
        if rank > placed:
            fail(f"{path}.ranking", "rank cannot exceed placed competitor count")
        if rank == 1:
            fail(f"{path}.ranking.rank", "the retained result did not win this board")
        if leader + 1e-9 < scored_speedup:
            fail(f"{path}.ranking.bestSpeedup", "leader speedup cannot trail the retained result")

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

        require_number(
            require_key(kernel, "firstCampaignScoredSpeedup", path),
            f"{path}.firstCampaignScoredSpeedup",
            positive=True,
        )
        source = require_dict(require_key(kernel, "source", path), f"{path}.source")
        expected_path = f"kernels/{expected_id}/submission.py"
        if require_string(require_key(source, "path", f"{path}.source"), f"{path}.source.path") != expected_path:
            fail(f"{path}.source.path", f"expected {expected_path!r}")
        digest = require_string(require_key(source, "sha256", f"{path}.source"), f"{path}.source.sha256")
        if not re.fullmatch(r"[0-9a-f]{64}", digest):
            fail(f"{path}.source.sha256", "expected a lowercase 64-character SHA-256")
        require_string(require_key(source, "pairedCheckJobId", f"{path}.source"), f"{path}.source.pairedCheckJobId")
        license_text = require_string(require_key(source, "license", f"{path}.source"), f"{path}.source.license")
        reused = require_bool(require_key(source, "reusedPublicSource", f"{path}.source"), f"{path}.source.reusedPublicSource")
        if require_string(require_key(source, "provenancePath", f"{path}.source"), f"{path}.source.provenancePath") != "PROVENANCE.md":
            fail(f"{path}.source.provenancePath", "expected 'PROVENANCE.md'")
        if expected_slug == "prefill-layer" and (not reused or "apache" not in license_text.lower()):
            fail(f"{path}.source", "prefill must disclose reused public Apache-licensed source")

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

    if campaign_sums != {"scored": 18, "checks": 61, "runs": 79}:
        fail("results.json.kernels[*].campaignCounts", f"aggregate must be 18 scored, 61 checks, 79 runs; found {campaign_sums}")
    if account_sums != {"scored": 20, "checks": 72, "runs": 92}:
        fail("results.json.kernels[*].accountCounts", f"aggregate must be 20 scored, 72 checks, 92 runs; found {account_sums}")
    return results


def validate_runs(raw: Any, results: dict[str, Any]) -> dict[str, Any]:
    payload = require_dict(raw, "runs.json")
    if require_key(payload, "schemaVersion", "runs.json") != 1:
        fail("runs.json.schemaVersion", "only schema version 1 is supported")
    if require_string(require_key(payload, "scope", "runs.json"), "runs.json.scope") != "five-agent-campaign":
        fail("runs.json.scope", "expected 'five-agent-campaign'")
    runs = require_list(require_key(payload, "runs", "runs.json"), "runs.json.runs")
    if len(runs) != 79:
        fail("runs.json.runs", f"expected exactly 79 campaign records, found {len(runs)}")

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
        fail("runs.json.runs", f"expected 18 scored runs and 61 checks, found {mode_counts}")

    for kernel in results["kernels"]:
        kernel_id = kernel["id"]
        if problem_counts[kernel_id] != kernel["campaignCounts"]:
            fail(
                f"runs.json.runs[{kernel_id}]",
                f"counts {problem_counts[kernel_id]} do not match results.json campaignCounts {kernel['campaignCounts']}",
            )
        for field, mode in (("scored", "run"), ("check", "check")):
            job_id = kernel[field]["jobId"]
            if job_id not in run_by_id:
                fail(f"results.json.kernels[{kernel_id}].{field}.jobId", f"job {job_id!r} is missing from runs.json")
            job = run_by_id[job_id]
            if job["problem"] != kernel_id or job["mode"] != mode:
                fail(f"results.json.kernels[{kernel_id}].{field}.jobId", "job problem or mode does not match")
            invalid_pass = job["passed"] is not True if field == "check" else job["passed"] is False
            if invalid_pass:
                fail(f"results.json.kernels[{kernel_id}].{field}.jobId", "referenced job does not have a valid pass state")
            if job["speedup"] is None:
                fail(f"results.json.kernels[{kernel_id}].{field}.jobId", "best retained job must report a speedup")
            if not math.isclose(float(job["speedup"]), float(kernel[field]["speedup"]), rel_tol=0, abs_tol=1e-9):
                fail(f"results.json.kernels[{kernel_id}].{field}.speedup", "does not match the referenced run")
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
                and (run["passed"] is True if mode == "check" else run["passed"] is not False)
                and run["speedup"] is not None
            ]
            if not passing:
                fail(f"runs.json.runs[{kernel_id}]", f"no passing {mode} with a reported speedup")
            if not math.isclose(max(passing), float(kernel[field]["speedup"]), rel_tol=0, abs_tol=1e-9):
                fail(f"results.json.kernels[{kernel_id}].{field}.speedup", f"is not the highest passing {mode} speedup")
    return payload


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
    page_title = title if title == project["title"] else f"{title} · {project['title']}"
    canonical_url = canonical(project, route)
    prefix = "../" * depth
    robots = '<meta name="robots" content="noindex">' if noindex else '<meta name="robots" content="index,follow">'
    nav_items = (
        ("notes", "index.html", "Field notes"),
        ("methodology", "methodology.html", "Methodology"),
        ("experiments", "experiments.html", "Experiments"),
    )
    nav_parts = []
    for key, href, label in nav_items:
        current = ' aria-current="page"' if key == active else ""
        nav_parts.append(f'<a href="{e(prefix + href)}"{current}>{e(label)}</a>')
    nav = "".join(nav_parts)
    closure = fmt_date(results["campaign"]["closedAt"])
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
        "author": {"@type": "Person", "name": project["author"]},
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
        <span><strong>Hopper Field Notes</strong><small>Open kernel research log</small></span>
      </a>
      <nav class="primary-nav" aria-label="Primary navigation">{nav}</nav>
      <a class="repo-link" href="{e(project['repositoryUrl'])}">Repository <span aria-hidden="true">↗</span></a>
    </div>
    <div class="closure-strip">
      <div class="shell"><strong>Track closed</strong><span>{e(closure)}. No new challenge runs are being started.</span><span>Ranks are provisional at the published snapshot.</span></div>
    </div>
  </header>
  <main id="main" tabindex="-1">{content}</main>
  <footer class="site-footer">
    <div class="shell footer-grid">
      <div><p class="footer-title">Hopper Field Notes</p><p>Five retained kernels, the full 79-attempt campaign, and the evidence behind every published result.</p></div>
      <div><p class="footer-label">Evidence</p><ul class="footer-links">
        <li><a href="{e(github_file(project, 'PERFORMANCE.md'))}">Performance report</a></li>
        <li><a href="{e(github_file(project, 'EXPERIMENTS.md'))}">Experiment log</a></li>
        <li><a href="{e(github_file(project, 'PROVENANCE.md'))}">Provenance</a></li>
        <li><a href="{e(github_file(project, 'LICENSE'))}">License</a></li>
      </ul></div>
      <div><p class="footer-label">Publication</p><p>Static HTML from committed JSON. No tracking, CDN, server, or runtime dependency.</p><p class="footer-small">Leaderboard snapshot: <time datetime="{e(results['leaderboardSnapshotAt'])}">{e(fmt_date(results['leaderboardSnapshotAt'], include_seconds=True))}</time></p></div>
    </div>
  </footer>
</body>
</html>
"""


def paragraph_list(paragraphs: list[str]) -> str:
    return "".join(f"<p>{e(paragraph)}</p>" for paragraph in paragraphs)


def result_card(kernel: dict[str, Any], *, depth: int = 0) -> str:
    scored = kernel["scored"]
    ranking = kernel["ranking"]
    leader = float(ranking["bestSpeedup"])
    ours = float(scored["speedup"])
    ratio = min(100.0, max(0.0, ours / leader * 100.0))
    gap = leader - ours
    time_reduction = (1 - float(scored["scoreUs"]) / float(scored["baselineUs"])) * 100
    source_project = {"repositoryUrl": REPOSITORY_URL, "branch": "main"}
    return f"""
<article class="result-card">
  <div class="card-topline"><span class="kernel-index">{e(kernel['id'])}</span><span class="status-badge success">Baseline beaten</span></div>
  <h3><a href="{e(local(depth, 'kernels/' + kernel['slug'] + '.html'))}">{e(kernel['title'])}</a></h3>
  <p>{e(kernel['summary'])}</p>
  <div class="headline-metric"><strong>{e(fmt_speedup(scored['speedup']))}</strong><span>best scored speedup</span></div>
  <dl class="metric-pairs">
    <div><dt>Measured</dt><dd>{e(fmt_us(scored['scoreUs']))}</dd></div>
    <div><dt>Baseline</dt><dd>{e(fmt_us(scored['baselineUs']))}</dd></div>
    <div><dt>95% CI</dt><dd>{e(fmt_ci(scored['ci95']))}</dd></div>
    <div><dt>Provisional rank</dt><dd>#{e(ranking['rank'])} of {e(ranking['placed'])}</dd></div>
  </dl>
  <div class="comparison" role="img" aria-label="Retained speedup {e(fmt_speedup(ours))}; provisional leader {e(fmt_speedup(leader))}">
    <div class="comparison-row"><span>Leader</span><span class="bar-track"><span class="bar-fill leader" style="width:100%"></span></span><strong>{e(fmt_speedup(leader))}</strong></div>
    <div class="comparison-row"><span>Ours</span><span class="bar-track"><span class="bar-fill ours" style="width:{ratio:.3f}%"></span></span><strong>{e(fmt_speedup(ours))}</strong></div>
  </div>
  <p class="card-note">{e(fmt_number(time_reduction, 2))}% lower measured time than baseline. Gap to the provisional leader: {e(fmt_speedup(gap))}.</p>
  <div class="card-actions"><a class="text-link" href="{e(local(depth, 'kernels/' + kernel['slug'] + '.html'))}">Read the field note <span aria-hidden="true">→</span></a><a class="text-link quiet" href="{e(github_file(source_project, kernel['source']['path']))}">View source <span aria-hidden="true">↗</span></a></div>
</article>"""


def build_index(results: dict[str, Any]) -> str:
    campaign = results["campaign"]
    account = results["accountTotals"]
    cards = "".join(result_card(kernel) for kernel in results["kernels"])
    notebook_cards = "".join(
        f"""<article class="notebook-card">
  <p class="eyebrow">Kernel {index:02d}</p>
  <h3><a href="kernels/{e(kernel['slug'])}.html">{e(kernel['title'])}</a></h3>
  <p>{e(kernel['story']['intro'][0])}</p>
  <p class="lesson"><strong>Measured lesson.</strong> {e(kernel['story']['lessons'][0])}</p>
  <a class="text-link" href="kernels/{e(kernel['slug'])}.html">Open notebook <span aria-hidden="true">→</span></a>
</article>"""
        for index, kernel in enumerate(results["kernels"], 1)
    )
    duration = fmt_duration(campaign["startedAt"], campaign["closedAt"])
    snapshot = fmt_date(results["leaderboardSnapshotAt"], include_seconds=True)
    content = f"""
<section class="hero shell" aria-labelledby="hero-title">
  <div class="hero-copy">
    <p class="eyebrow light">Muse Spark Open Kernel Challenge · Published research log</p>
    <h1 id="hero-title">Optimizing in the dark.<br><em>Measuring in the light.</em></h1>
    <p class="hero-lede">Five GPU kernels prepared and tuned with CPU-only Linux tooling, then benchmarked remotely on an NVIDIA H200. This is the complete field record: retained source, confidence intervals, provisional gaps, failed ideas, and all 79 campaign attempts.</p>
    <div class="hero-actions"><a class="button primary" href="#results">Read the results</a><a class="button ghost" href="experiments.html">Inspect all attempts</a></div>
    <p class="hero-disclosure"><strong>Outcome:</strong> every retained kernel beat its scored baseline; none won its leaderboard. Rankings remain provisional at the {e(snapshot)} snapshot.</p>
  </div>
  <aside class="hero-panel" aria-label="Campaign at a glance">
    <p class="panel-kicker">Field window</p>
    <p class="window-time">{e(duration)}</p>
    <p>{e(fmt_date(campaign['startedAt']))}<br>to {e(fmt_date(campaign['closedAt']))}</p>
    <div class="hero-stat-grid">
      <div><strong>5</strong><span>kernels</span></div>
      <div><strong>{e(campaign['runCount'])}</strong><span>campaign runs</span></div>
      <div><strong>{e(campaign['scoredCount'])}</strong><span>scored</span></div>
      <div><strong>{e(campaign['checkCount'])}</strong><span>checks</span></div>
    </div>
    <p class="panel-foot"><span class="signal-dot" aria-hidden="true"></span> Challenge track closed 5 Nov, 08:00 UTC</p>
  </aside>
</section>

<section class="context-band" aria-label="Development context">
  <div class="shell context-grid">
    <div><span class="context-number">01</span><p><strong>Local work:</strong> Linux, CPU-only. No local GPU was used.</p></div>
    <div><span class="context-number">02</span><p><strong>Measurement:</strong> hosted checks and scored runs on NVIDIA H200.</p></div>
    <div><span class="context-number">03</span><p><strong>Publication:</strong> exact retained sources, hashes, run ledger, and provenance.</p></div>
  </div>
</section>

<section class="section shell" id="results" aria-labelledby="results-title">
  <div class="section-heading split-heading">
    <div><p class="eyebrow">The result board</p><h2 id="results-title">Five baseline wins, five honest gaps.</h2></div>
    <p>Speedup is baseline time divided by measured time. The bars compare each retained scored result with the raw provisional leader at one timestamp; they are not claims of final placement.</p>
  </div>
  <div class="results-grid">{cards}</div>
</section>

<section class="section ink-section" aria-labelledby="scope-title">
  <div class="shell scope-layout">
    <div>
      <p class="eyebrow light">One campaign, two scopes</p>
      <h2 id="scope-title">79 attempts tell this story.<br>92 runs belong to the account.</h2>
      <p>The experiment ledger deliberately covers the five-agent optimization campaign only: {e(campaign['scoredCount'])} scored runs and {e(campaign['checkCount'])} checks. Account totals for 5 November also include work outside that bounded campaign and must not be used as the campaign denominator.</p>
      <a class="button ghost" href="methodology.html#scope">See the scope rules</a>
    </div>
    <div class="scope-comparison" aria-label="Campaign and account run totals">
      <article><span>Published campaign</span><strong>{e(campaign['runCount'])}</strong><p>{e(campaign['scoredCount'])} scored + {e(campaign['checkCount'])} checks</p></article>
      <article><span>Account total</span><strong>{e(account['runCount'])}</strong><p>{e(account['scoredCount'])} scored + {e(account['checkCount'])} checks</p></article>
      <p class="scope-delta">13 account runs sit outside the five-agent campaign.</p>
    </div>
  </div>
</section>

<section class="section shell" aria-labelledby="notebooks-title">
  <div class="section-heading"><p class="eyebrow">Kernel notebooks</p><h2 id="notebooks-title">The useful part is what changed—and what did not work.</h2><p>Each article ties the retained measurement to its exact source, then separates approach, lessons, failures, and caveats.</p></div>
  <div class="notebook-grid">{notebook_cards}</div>
</section>

<section class="section shell methodology-teaser" aria-labelledby="evidence-title">
  <div class="evidence-mark" aria-hidden="true">CI</div>
  <div><p class="eyebrow">How to read the evidence</p><h2 id="evidence-title">Checks guide iteration. Scored runs support the published result.</h2><p>A passing check can be faster than the retained scored run without becoming the leaderboard result. Confidence intervals report measurement uncertainty, not certainty that one implementation dominates every shape or future runtime. Source hashes and paired checks connect the published file to verification.</p><div class="inline-links"><a class="text-link" href="methodology.html">Read methodology <span aria-hidden="true">→</span></a><a class="text-link" href="experiments.html">Browse 79 attempts <span aria-hidden="true">→</span></a></div></div>
</section>

<aside class="section shell disclosure-card" aria-labelledby="attribution-title">
  <div><p class="eyebrow">Provenance matters</p><h2 id="attribution-title">The prefill result includes public Apache-licensed winner work.</h2></div>
  <p>That source is explicitly attributed as reused public work, not claimed as original authorship. The article and repository provenance record distinguish adaptation and verification from upstream contribution.</p>
  <a class="button dark" href="kernels/prefill-layer.html#provenance">Read the attribution</a>
</aside>
"""
    return layout(
        results,
        route="index.html",
        title=results["project"]["title"],
        description="Five Open Kernel Challenge implementations developed CPU-only on Linux and benchmarked remotely on NVIDIA H200, with complete results and experiment history.",
        active="notes",
        depth=0,
        content=content,
        body_class="home-page",
    )


def build_methodology(results: dict[str, Any]) -> str:
    campaign = results["campaign"]
    account = results["accountTotals"]
    source_rows = "".join(
        f"""<tr>
  <th scope="row"><a href="kernels/{e(kernel['slug'])}.html">{e(kernel['title'])}</a></th>
  <td><code>{e(kernel['scored']['jobId'])}</code></td>
  <td><code>{e(kernel['source']['pairedCheckJobId'])}</code></td>
  <td><a href="{e(github_file(results['project'], kernel['source']['path']))}"><code>{e(kernel['source']['sha256'])}</code></a></td>
</tr>"""
        for kernel in results["kernels"]
    )
    references: dict[str, str] = {}
    for kernel in results["kernels"]:
        for reference in kernel["references"]:
            references.setdefault(reference["url"], reference["label"])
    reference_items = "".join(
        f'<li><a href="{e(url)}">{e(label)} <span aria-hidden="true">↗</span></a></li>'
        for url, label in references.items()
    )
    content = f"""
<header class="page-hero shell">
  <p class="eyebrow">Measurement notes</p>
  <h1>How the evidence was made.</h1>
  <p>Local development used only CPU-side Linux tools. GPU execution, correctness checks, and timing happened through the challenge's hosted NVIDIA H200 workflow. The distinction shapes every claim on this site.</p>
  <div class="page-meta"><span>Hardware: {e(campaign['gpu'])}</span><span>Campaign: {e(campaign['runCount'])} attempts</span><span>Status: closed</span></div>
</header>

<section class="section shell" aria-labelledby="workflow-title">
  <div class="section-heading"><p class="eyebrow">Hosted workflow</p><h2 id="workflow-title">Code moved to the GPU; raw infrastructure did not move into this publication.</h2></div>
  <ol class="workflow">
    <li><span>1</span><div><h3>Develop locally</h3><p>Kernel source and experiment hypotheses were prepared on Linux with <code>localGpuUsed: false</code>. No local timing is presented as H200 evidence.</p></div></li>
    <li><span>2</span><div><h3>Run a hosted check</h3><p>Checks tested correctness and returned off-board timing evidence when available. They supported iteration but did not set leaderboard placement.</p></div></li>
    <li><span>3</span><div><h3>Submit a scored run</h3><p>Scored runs produced the retained publication metrics: measured microseconds, scored baseline, speedup, and a 95% confidence interval.</p></div></li>
    <li><span>4</span><div><h3>Freeze source and snapshot</h3><p>The retained file was hashed, associated with a passing check, and compared with a timestamped leaderboard snapshot after the track closed.</p></div></li>
  </ol>
</section>

<section class="section paper-section" aria-labelledby="modes-title">
  <div class="shell two-column-prose">
    <div><p class="eyebrow">Check vs scored</p><h2 id="modes-title">Two modes answer different questions.</h2></div>
    <div class="definition-stack">
      <article><h3><span class="mode-dot check" aria-hidden="true"></span>Check</h3><p>Did this candidate pass the hosted evaluator, and what timing evidence was returned? The site reports the highest passing campaign check separately. It can differ from the check paired to the exact retained source.</p></article>
      <article><h3><span class="mode-dot scored" aria-hidden="true"></span>Scored run</h3><p>What result was placed on the scored path? The five headline values—{', '.join(e(fmt_speedup(kernel['scored']['speedup'])) for kernel in results['kernels'])}—come from verified scored jobs, not cherry-picked checks.</p></article>
    </div>
  </div>
</section>

<section class="section shell" aria-labelledby="uncertainty-title">
  <div class="section-heading split-heading"><div><p class="eyebrow">Uncertainty</p><h2 id="uncertainty-title">A confidence interval is a range, not a trophy.</h2></div><p>Each 95% interval summarizes the evaluator's reported uncertainty around speedup. Narrower is more precise; overlap does not by itself prove a tie, and separation does not make the result portable to other shapes or hardware.</p></div>
  <div class="principle-grid">
    <article><strong>Use the interval</strong><p>Read the point estimate with both bounds. This site never replaces a missing interval with a guessed one.</p></article>
    <article><strong>Keep modes separate</strong><p>A fast check is useful evidence, but only a scored run supports the displayed scored result and provisional rank.</p></article>
    <article><strong>Keep the denominator visible</strong><p>Speedup compares the evaluator's scored baseline with measured kernel time for that problem and run.</p></article>
  </div>
</section>

<section class="section ink-section" id="scope" aria-labelledby="scope-method-title">
  <div class="shell">
    <div class="section-heading split-heading light"><div><p class="eyebrow light">Run scope</p><h2 id="scope-method-title">The ledger is complete for the campaign—not the whole account.</h2></div><p>The bounded campaign began {e(fmt_date(campaign['startedAt'], include_seconds=True))} and closed {e(fmt_date(campaign['closedAt'], include_seconds=True))}. Account totals are disclosed to prevent accidental conflation.</p></div>
    <div class="table-scroll light-table" role="region" aria-label="Campaign and account scope comparison" tabindex="0">
      <table><thead><tr><th scope="col">Scope</th><th scope="col">Scored</th><th scope="col">Checks</th><th scope="col">Total</th><th scope="col">Published here</th></tr></thead>
      <tbody><tr><th scope="row">Five-agent campaign</th><td>{e(campaign['scoredCount'])}</td><td>{e(campaign['checkCount'])}</td><td>{e(campaign['runCount'])}</td><td>All attempts</td></tr>
      <tr><th scope="row">5 Nov account total</th><td>{e(account['scoredCount'])}</td><td>{e(account['checkCount'])}</td><td>{e(account['runCount'])}</td><td>Aggregate disclosure only</td></tr></tbody></table>
    </div>
  </div>
</section>

<section class="section shell" id="provenance" aria-labelledby="provenance-title">
  <div class="section-heading"><p class="eyebrow">Source chain</p><h2 id="provenance-title">Every article resolves to code, a digest, and verification.</h2><p>The highest passing check and the paired check are intentionally separate fields: the first summarizes the campaign; the second ties verification to the exact retained source.</p></div>
  <div class="table-scroll" role="region" aria-label="Kernel source verification matrix" tabindex="0">
    <table class="source-table"><thead><tr><th scope="col">Kernel</th><th scope="col">Scored job</th><th scope="col">Paired check</th><th scope="col">Retained source SHA-256</th></tr></thead><tbody>{source_rows}</tbody></table>
  </div>
  <aside class="attribution-note"><h3>Prefill attribution</h3><p>The retained prefill kernel reuses publicly released Apache-licensed winner work. Original credit and license obligations are preserved; this publication claims the campaign's adaptation, verification, and measurements—not authorship of the upstream source.</p><a class="text-link" href="kernels/prefill-layer.html#provenance">Read the kernel-specific record <span aria-hidden="true">→</span></a></aside>
</section>

<section class="section shell" aria-labelledby="limits-title">
  <div class="section-heading"><p class="eyebrow">Limitations</p><h2 id="limits-title">What these numbers do not establish.</h2></div>
  <ul class="limitations-list">
    <li><strong>Hardware specificity.</strong> Measurements come from the hosted NVIDIA H200 path; they are not local benchmarks or guarantees for other accelerators.</li>
    <li><strong>Workload specificity.</strong> A retained implementation can be excellent for judged shapes while leaving other shapes, batching regimes, or concurrency patterns unexplored.</li>
    <li><strong>Runtime specificity.</strong> Compiler, driver, evaluator, and scheduling conditions are pinned to the hosted environment and may change outside the archived campaign.</li>
    <li><strong>Leaderboard timing.</strong> Rank, placed count, and leader speedup are provisional values captured at <time datetime="{e(results['leaderboardSnapshotAt'])}">{e(fmt_date(results['leaderboardSnapshotAt'], include_seconds=True))}</time>.</li>
    <li><strong>Campaign selection.</strong> The experiment table contains all 79 in-scope attempts, including failures, but not the 13 account runs outside the defined campaign.</li>
  </ul>
</section>

<section class="section shell references-section" aria-labelledby="references-title">
  <div><p class="eyebrow">Research links</p><h2 id="references-title">Sources used across the field notes.</h2></div>
  <ul class="reference-list">{reference_items}</ul>
</section>
"""
    return layout(
        results,
        route="methodology.html",
        title="Methodology",
        description="Hosted H200 benchmarking workflow, scored versus check runs, confidence intervals, source verification, campaign scope, and limitations.",
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


def build_experiments(results: dict[str, Any], runs_payload: dict[str, Any]) -> str:
    runs = runs_payload["runs"]
    title_by_id = {kernel["id"]: kernel["title"] for kernel in results["kernels"]}
    slug_by_id = {kernel["id"]: kernel["slug"] for kernel in results["kernels"]}
    options = "".join(f'<option value="{e(kernel_id)}">{e(title)}</option>' for kernel_id, title in title_by_id.items())
    rows = []
    for index, run in enumerate(runs, 1):
        outcome_key, outcome_label = run_outcome(run)
        mode_label = "Scored" if run["mode"] == "run" else "Check"
        finished = (
            f'<time datetime="{e(run["finishedAt"])}">{e(fmt_date(run["finishedAt"], include_seconds=True))}</time>'
            if run["finishedAt"]
            else "—"
        )
        rows.append(
            f"""<tr data-run-row data-problem="{e(run['problem'])}" data-mode="{e(run['mode'])}" data-outcome="{e(outcome_key)}">
  <td class="row-number">{index}</td>
  <td><time datetime="{e(run['submittedAt'])}">{e(fmt_date(run['submittedAt'], include_seconds=True))}</time></td>
  <th scope="row"><a href="kernels/{e(slug_by_id[run['problem']])}.html">{e(title_by_id[run['problem']])}</a><code>{e(run['problem'])}</code></th>
  <td><span class="mode-badge {e(run['mode'])}">{e(mode_label)}</span></td>
  <td><span class="outcome-badge {e(outcome_key)}">{e(outcome_label)}</span><small>{e(run['status'])}</small></td>
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
  <p class="eyebrow">Attempt ledger</p>
  <h1>All 79 campaign attempts.<br>No survivor bias.</h1>
  <p>Every in-scope scored run and check appears below in submission order, including failed attempts and records without reported metrics. Null evidence stays null; it is never converted into a zero or an invented speedup.</p>
  <div class="page-meta"><span>{e(campaign['scoredCount'])} scored</span><span>{e(campaign['checkCount'])} checks</span><span>{e(campaign['runCount'])} campaign total</span></div>
</header>

<section class="section shell compact-top" aria-labelledby="ledger-title" data-experiment-ledger>
  <div class="ledger-intro"><div><p class="eyebrow">Filter the record</p><h2 id="ledger-title">Experiment history</h2></div><p>This table is the five-agent campaign, not the {e(account['runCount'])}-run account total. Filters run locally in your browser and do not send data anywhere.</p></div>
  <form class="filter-panel" data-filter-form aria-label="Filter experiment history">
    <div class="filter-field"><label for="problem-filter">Problem</label><select id="problem-filter" name="problem" data-filter="problem"><option value="all">All five kernels</option>{options}</select></div>
    <div class="filter-field"><label for="mode-filter">Mode</label><select id="mode-filter" name="mode" data-filter="mode"><option value="all">Scored and checks</option><option value="run">Scored only</option><option value="check">Checks only</option></select></div>
    <div class="filter-field"><label for="outcome-filter">Pass / fail</label><select id="outcome-filter" name="outcome" data-filter="outcome"><option value="all">Every outcome</option><option value="pass">Pass</option><option value="fail">Fail</option><option value="unknown">Not reported</option></select></div>
    <button class="button filter-reset" type="reset">Reset filters</button>
  </form>
  <div class="ledger-status"><p data-filter-summary aria-live="polite" aria-atomic="true">Showing all {len(runs)} attempts.</p><p><span class="legend-key"><span class="mode-dot scored" aria-hidden="true"></span>Scored</span><span class="legend-key"><span class="mode-dot check" aria-hidden="true"></span>Check</span></p></div>
  <noscript><p class="noscript-note">JavaScript is off, so filters are unavailable. The complete 79-row ledger remains visible below.</p></noscript>
  <div class="no-match" data-no-match hidden><h3>No attempts match those filters.</h3><p>Reset one or more controls to return to the complete campaign record.</p></div>
  <div class="table-scroll ledger-scroll" role="region" aria-label="All campaign attempts; scroll horizontally for more columns" tabindex="0">
    <table class="ledger-table">
      <caption>Chronological five-agent campaign ledger: {e(campaign['runCount'])} attempts from {e(fmt_date(campaign['startedAt'], include_seconds=True))} through track closure.</caption>
      <thead><tr><th scope="col">#</th><th scope="col">Submitted</th><th scope="col">Problem</th><th scope="col">Mode</th><th scope="col">Outcome</th><th scope="col">Speedup</th><th scope="col">95% CI</th><th scope="col">Score</th><th scope="col">Baseline</th><th scope="col">Job</th></tr></thead>
      <tbody>{''.join(rows)}</tbody>
    </table>
  </div>
</section>

<aside class="section shell ledger-note" aria-labelledby="ledger-note-title">
  <div><p class="eyebrow">Reading the ledger</p><h2 id="ledger-note-title">A fast check is not a scored result.</h2></div>
  <p>Use checks to understand iteration and correctness. Use the retained scored jobs for headline speedups and ranks. A dash means the archive did not report that metric; it does not mean zero performance.</p>
  <a class="text-link" href="methodology.html#modes-title">Review the mode distinction <span aria-hidden="true">→</span></a>
</aside>
"""
    return layout(
        results,
        route="experiments.html",
        title="Experiment history",
        description="A complete, filterable record of all 79 scored and check attempts in the five-kernel Open Kernel Challenge campaign.",
        active="experiments",
        depth=0,
        content=content,
        body_class="experiments-page",
    )


def build_kernel_article(results: dict[str, Any], kernel: dict[str, Any], index: int) -> str:
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
    reused_notice = ""
    if source["reusedPublicSource"]:
        reused_notice = f"""<aside class="attribution-note prominent">
  <p class="eyebrow">Required attribution</p>
  <h2>Public licensed work, clearly credited.</h2>
  <p>The retained prefill kernel reuses publicly released Apache-licensed winner work. Credit remains with the original authors. This field note documents adaptation, source verification, and campaign measurement; it does not claim original authorship of the reused implementation.</p>
  <p><strong>Recorded license:</strong> {e(source['license'])}</p>
  <a class="text-link" href="{e(provenance_url)}">Read the complete provenance record <span aria-hidden="true">↗</span></a>
</aside>"""
    previous_kernel = results["kernels"][index - 1] if index > 0 else None
    next_kernel = results["kernels"][index + 1] if index + 1 < len(results["kernels"]) else None
    previous_link = (
        f'<a rel="prev" href="{e(previous_kernel["slug"])}.html"><span>Previous note</span><strong>← {e(previous_kernel["title"])}</strong></a>'
        if previous_kernel
        else '<span aria-hidden="true"></span>'
    )
    next_link = (
        f'<a rel="next" href="{e(next_kernel["slug"])}.html"><span>Next note</span><strong>{e(next_kernel["title"])} →</strong></a>'
        if next_kernel
        else '<a href="../index.html#results"><span>Return</span><strong>All results →</strong></a>'
    )
    content = f"""
<nav class="breadcrumbs shell" aria-label="Breadcrumb"><ol><li><a href="../index.html">Field notes</a></li><li><span aria-current="page">{e(kernel['title'])}</span></li></ol></nav>
<article class="kernel-article">
  <header class="kernel-hero shell">
    <div class="kernel-heading">
      <p class="eyebrow">Kernel note {index + 1:02d} · {e(kernel['id'])}</p>
      <h1>{e(kernel['title'])}</h1>
      <p class="article-deck">{e(kernel['summary'])}</p>
      <div class="article-actions"><a class="button dark" href="{e(source_url)}">View retained source <span aria-hidden="true">↗</span></a><a class="button outline" href="../experiments.html?problem={e(kernel['id'])}">Filter its attempts</a></div>
    </div>
    <aside class="result-ticket" aria-label="Retained result summary">
      <p class="ticket-label">Best scored result</p>
      <strong class="ticket-speedup">{e(fmt_speedup(scored['speedup']))}</strong>
      <p>{e(fmt_us(scored['scoreUs']))} vs {e(fmt_us(scored['baselineUs']))} baseline</p>
      <dl><div><dt>95% CI</dt><dd>{e(fmt_ci(scored['ci95']))}</dd></div><div><dt>Rank</dt><dd>#{e(ranking['rank'])} / {e(ranking['placed'])}</dd></div><div><dt>Leader</dt><dd>{e(fmt_speedup(ranking['bestSpeedup']))}</dd></div><div><dt>Gap</dt><dd>{e(fmt_speedup(leader_gap))}</dd></div></dl>
      <p class="ticket-foot">Provisional at <time datetime="{e(results['leaderboardSnapshotAt'])}">{e(fmt_date(results['leaderboardSnapshotAt']))}</time></p>
    </aside>
  </header>

  <div class="article-layout shell">
    <aside class="article-rail" aria-label="Article summary">
      <p class="rail-title">In this note</p>
      <nav aria-label="Article sections"><a href="#problem">Problem</a><a href="#approach">Approach</a><a href="#lessons">Lessons</a><a href="#failures">Failed ideas</a><a href="#caveats">Caveats</a><a href="#provenance">Provenance</a></nav>
      <dl class="rail-counts"><div><dt>Campaign runs</dt><dd>{e(kernel['campaignCounts']['runs'])}</dd></div><div><dt>Scored</dt><dd>{e(kernel['campaignCounts']['scored'])}</dd></div><div><dt>Checks</dt><dd>{e(kernel['campaignCounts']['checks'])}</dd></div></dl>
    </aside>
    <div class="article-body">
      <section id="problem"><p class="section-number">01 · Problem</p><h2>What this kernel asked for.</h2>{paragraph_list(kernel['story']['intro'])}</section>
      <section id="approach"><p class="section-number">02 · Retained approach</p><h2>What survived the campaign.</h2>{paragraph_list(kernel['story']['approach'])}</section>
      <aside class="measurement-callout" aria-label="Measurement summary"><p class="eyebrow">Scored evidence</p><div><strong>{e(fmt_speedup(scored['speedup']))}</strong><span>retained</span><strong>{e(fmt_ci(scored['ci95']))}</strong><span>95% CI</span><strong>{e(fmt_number(improvement, 2))}%</strong><span>change from first campaign score</span></div><p>Best passing check: {e(fmt_speedup(check['speedup']))} ({e(fmt_ci(check['ci95']))}). Check evidence remains off-board.</p></aside>
      <section id="lessons"><p class="section-number">03 · Measured lessons</p><h2>What the numbers taught us.</h2>{paragraph_list(kernel['story']['lessons'])}</section>
      <section id="failures"><p class="section-number">04 · Experiments that did not survive</p><h2>Regression is part of the record.</h2>{paragraph_list(kernel['story']['failures'])}</section>
      <section id="caveats"><p class="section-number">05 · Boundaries</p><h2>Where this result stops.</h2>{paragraph_list(kernel['story']['caveats'])}</section>
      <section id="provenance"><p class="section-number">06 · Source and provenance</p><h2>Exact bytes, explicit credit.</h2>
        <dl class="provenance-list"><div><dt>Retained file</dt><dd><a href="{e(source_url)}"><code>{e(source['path'])}</code></a></dd></div><div><dt>SHA-256</dt><dd><code>{e(source['sha256'])}</code></dd></div><div><dt>Scored job</dt><dd><code>{e(scored['jobId'])}</code></dd></div><div><dt>Highest passing check</dt><dd><code>{e(check['jobId'])}</code></dd></div><div><dt>Check paired to source</dt><dd><code>{e(source['pairedCheckJobId'])}</code></dd></div><div><dt>License record</dt><dd>{e(source['license'])}</dd></div></dl>
        <p>The highest passing campaign check can differ from the paired check for the retained source. Both are published so performance exploration is not confused with byte-for-byte verification.</p>
        <a class="text-link" href="{e(provenance_url)}">Open full provenance <span aria-hidden="true">↗</span></a>
      </section>
      {reused_notice}
      <section class="article-references" aria-labelledby="article-references-title"><p class="section-number">Research links</p><h2 id="article-references-title">References and upstream material.</h2><ul class="reference-list">{references}</ul></section>
    </div>
  </div>
</article>
<nav class="article-pagination shell" aria-label="Kernel article pagination">{previous_link}{next_link}</nav>
"""
    return layout(
        results,
        route=f"kernels/{kernel['slug']}.html",
        title=kernel["title"],
        description=f"{kernel['summary']} Retained scored result: {fmt_speedup(scored['speedup'])}, with source provenance, experiments, lessons, and limitations.",
        active="notes",
        depth=1,
        content=content,
        body_class="kernel-page",
    )


def build_404(results: dict[str, Any], assets_dir: Path) -> str:
    content = """
<section class="not-found shell" aria-labelledby="not-found-title">
  <p class="error-code">404 · off the measured path</p>
  <h1 id="not-found-title">This field note does not exist.</h1>
  <p>The campaign is closed, but the published notebooks, methodology, and complete experiment ledger are still available.</p>
  <div class="hero-actions"><a class="button dark" href="index.html">Return to field notes</a><a class="button outline" href="experiments.html">Browse experiments</a></div>
</section>
"""
    document = layout(
        results,
        route="404.html",
        title="Page not found",
        description="The requested Hopper Field Notes page could not be found.",
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
    for route in ("index.html", "methodology.html", "experiments.html"):
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
    routes = ["", "methodology.html", "experiments.html"] + [
        f"kernels/{kernel['slug']}.html" for kernel in results["kernels"]
    ]
    lastmod = results["leaderboardSnapshotAt"][:10]
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
    outputs: dict[Path, str] = {
        output_dir / "index.html": build_index(results),
        output_dir / "methodology.html": build_methodology(results),
        output_dir / "experiments.html": build_experiments(results, runs),
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
