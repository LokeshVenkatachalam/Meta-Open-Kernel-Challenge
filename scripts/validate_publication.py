#!/usr/bin/env python3
"""Check published data, exact kernel bytes, privacy boundaries, and static links."""

import ast
import hashlib
import json
import math
import re
import sys
from collections import Counter
from datetime import datetime
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit


ROOT = Path(__file__).resolve().parents[1]
PROBLEMS = {
    "muse_bf16_proj",
    "muse_attention_decode",
    "muse_moe_w4a16",
    "muse_decode_layer",
    "muse_prefill_layer",
}
RUN_FIELDS = {
    "id", "problem", "mode", "submittedAt", "finishedAt", "status",
    "passed", "speedup", "ci95", "scoreUs", "baselineUs",
}
TERMINAL = {"done", "failed", "rejected", "duplicate", "cancelled"}
SECRET_PATTERNS = {
    "GitHub token": re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})"),
    "private key": re.compile(r"-----BEGIN (?:[A-Z ]+ )?PRIVATE KEY-----"),
    "bearer credential": re.compile(r"Authorization:\s*Bearer\s+[A-Za-z0-9_.-]{16,}", re.I),
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite_number(value, label, positive=False):
    require(
        not isinstance(value, bool) and isinstance(value, (int, float)),
        label + " must be a number",
    )
    require(math.isfinite(value), label + " must be finite")
    if positive:
        require(value > 0, label + " must be positive")
    return value


def integer(value, label):
    require(type(value) is int and value >= 0, label + " must be a nonnegative integer")
    return value


def timestamp(value, label):
    require(isinstance(value, str), label + " must be an ISO timestamp")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    require(parsed.tzinfo is not None, label + " needs a timezone")
    require(parsed.utcoffset().total_seconds() == 0, label + " must be UTC")
    return parsed


def interval(value, center, label):
    if value is None:
        return
    require(isinstance(value, list) and len(value) == 2, label + " needs two bounds")
    require(center is not None, label + " cannot describe an unreported score")
    lower, upper = [finite_number(item, label, positive=True) for item in value]
    require(lower <= center <= upper, label + " does not contain its score")


def safe_path(root, relative):
    require(isinstance(relative, str) and relative, "File path must be a nonempty string")
    require(not Path(relative).is_absolute(), "Published file paths must be relative")
    path = (root / relative).resolve()
    require(path.is_relative_to(root.resolve()), "File path escapes the repository")
    require(path.is_file(), "Missing publication file: " + relative)
    return path


def load_json(path):
    def reject_constant(value):
        raise ValueError("Non-standard JSON number in {}: {}".format(path.name, value))
    return json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant)


def validate_runs(document, campaign):
    require(set(document) == {"schemaVersion", "scope", "runs"}, "Unexpected run dataset fields")
    require(document["schemaVersion"] == 1, "Unsupported run schema")
    require(document["scope"] == "five-agent-campaign", "Ambiguous run scope")
    require(isinstance(document["runs"], list), "runs must be an array")
    start = timestamp(campaign["startedAt"], "Campaign start")
    cutoff = timestamp(campaign["closedAt"], "Campaign cutoff")
    require(start < cutoff, "Campaign window is invalid")
    indexed = {}
    times = []
    for row in document["runs"]:
        require(isinstance(row, dict) and set(row) == RUN_FIELDS, "Run field allowlist mismatch")
        job_id = row["id"]
        require(isinstance(job_id, str) and re.fullmatch(r"[a-f0-9]{16}", job_id), "Invalid job ID")
        require(job_id not in indexed, "Duplicate job ID: " + job_id)
        require(row["problem"] in PROBLEMS, "Unknown run problem")
        require(row["mode"] in {"check", "run"}, "Unknown run mode")
        require(row["status"] in TERMINAL, "Unfinished run in closed campaign")
        require(row["passed"] is None or type(row["passed"]) is bool, "Invalid correctness verdict")
        submitted = timestamp(row["submittedAt"], "Run submission time")
        require(start <= submitted < cutoff, "Run lies outside the campaign window")
        if row["finishedAt"] is not None:
            require(timestamp(row["finishedAt"], "Run finish time") >= submitted, "Run finishes before submission")
        speedup = row["speedup"]
        if row["passed"] is not True:
            require(speedup is None and row["ci95"] is None, "Unverified run has a performance claim")
            require(row["scoreUs"] is None, "Unverified run has a measured score")
        if speedup is not None:
            finite_number(speedup, "Run speedup", positive=True)
        interval(row["ci95"], speedup, "Run confidence interval")
        for field in ("scoreUs", "baselineUs"):
            if row[field] is not None:
                finite_number(row[field], field, positive=True)
        indexed[job_id] = row
        times.append(submitted)
    require(times == sorted(times), "Run history must be chronological")
    counts = Counter(row["mode"] for row in indexed.values())
    require(len(indexed) == campaign["runCount"], "Campaign run total is wrong")
    require(counts["run"] == campaign["scoredCount"], "Campaign scored total is wrong")
    require(counts["check"] == campaign["checkCount"], "Campaign check total is wrong")
    return indexed


def validate_data(root):
    results = load_json(root / "docs/data/results.json")
    run_document = load_json(root / "docs/data/runs.json")
    require(results["schemaVersion"] == 1, "Unsupported results schema")
    campaign = results["campaign"]
    require(campaign["status"] == "closed", "The published campaign must be marked closed")
    require(campaign["gpu"] == "NVIDIA H200" and campaign["localGpuUsed"] is False, "Hardware provenance is wrong")
    require(
        (campaign["runCount"], campaign["scoredCount"], campaign["checkCount"]) == (79, 18, 61),
        "Expected the complete 79-run campaign, not recent API pagination",
    )
    timestamp(results["leaderboardSnapshotAt"], "Leaderboard snapshot")
    jobs = validate_runs(run_document, campaign)
    kernels = results["kernels"]
    require(isinstance(kernels, list) and len(kernels) == 5, "Exactly five kernels are required")
    require({item["id"] for item in kernels} == PROBLEMS, "Missing or duplicate kernel")
    require(len({item["slug"] for item in kernels}) == 5, "Duplicate article slug")
    account_counts = Counter()
    for kernel in kernels:
        problem = kernel["id"]
        require(re.fullmatch(r"[a-z0-9-]+", kernel["slug"]), "Invalid article slug")
        rows = [row for row in jobs.values() if row["problem"] == problem]
        counts = Counter(row["mode"] for row in rows)
        actual = {"scored": counts["run"], "checks": counts["check"], "runs": len(rows)}
        require(kernel["campaignCounts"] == actual, "Wrong per-problem campaign counts: " + problem)
        account = kernel["accountCounts"]
        for key in ("scored", "checks", "runs"):
            integer(account[key], "Account count")
            require(account[key] >= actual[key], "Account count is smaller than the campaign")
            account_counts[key] += account[key]
        require(account["runs"] == account["scored"] + account["checks"], "Account count sum is wrong")
        scored = kernel["scored"]
        checked = kernel["check"]
        for entry, mode in ((scored, "run"), (checked, "check")):
            require(entry["jobId"] in jobs, "Missing referenced job")
            row = jobs[entry["jobId"]]
            require(row["problem"] == problem and row["mode"] == mode and row["passed"] is True, "Incorrect winner association")
            require(entry["speedup"] == row["speedup"] and entry["ci95"] == row["ci95"], "Winner metrics differ from run history")
            passing = [item["speedup"] for item in rows if item["mode"] == mode and item["passed"] is True and item["speedup"] is not None]
            require(entry["speedup"] == max(passing), "Winner is not the best observed result")
        require(scored["ci95"] is not None and scored["ci95"][0] > 1, "Baseline win is not supported by its interval")
        row = jobs[scored["jobId"]]
        require(scored["scoreUs"] == row["scoreUs"] and scored["baselineUs"] == row["baselineUs"], "Scored timing differs from its record")
        first = next(item for item in rows if item["mode"] == "run" and item["passed"] is True and item["speedup"] is not None)
        require(kernel["firstCampaignScoredSpeedup"] == first["speedup"], "First scored campaign result is wrong")
        rank = kernel["ranking"]
        integer(rank["rank"], "Leaderboard rank")
        integer(rank["placed"], "Ranked entry count")
        require(1 <= rank["rank"] <= rank["placed"], "Leaderboard denominator is inconsistent")
        finite_number(rank["bestSpeedup"], "Leader speedup", positive=True)
        require(rank["bestSpeedup"] >= scored["speedup"] - 0.0001, "Leader is slower than our own recorded score")
        source = kernel["source"]
        require(source["path"] == "kernels/{}/submission.py".format(problem), "Unexpected kernel path")
        path = safe_path(root, source["path"])
        data = path.read_bytes()
        require(hashlib.sha256(data).hexdigest() == source["sha256"], "Verified kernel bytes changed: " + problem)
        require(data.splitlines()[0].decode("utf-8") == "# okc: problem=" + problem, "Kernel problem header changed")
        ast.parse(data, filename=str(path))
        paired = jobs[source["pairedCheckJobId"]]
        require(paired["problem"] == problem and paired["mode"] == "check" and paired["passed"] is True, "Invalid source-associated check")
        safe_path(root, source["provenancePath"])
        for field in ("intro", "approach", "lessons", "failures", "caveats"):
            require(
                isinstance(kernel["story"][field], list)
                and all(isinstance(text, str) and text.strip() for text in kernel["story"][field]),
                "Invalid article story section: " + field,
            )
        require(kernel["story"]["intro"] and kernel["references"], "Article lacks narrative or sources")
        for reference in kernel["references"]:
            parsed = urlsplit(reference["url"])
            require(parsed.scheme == "https" and parsed.netloc, "Research reference is not HTTPS")
        if problem == "muse_prefill_layer":
            require(source["reusedPublicSource"] is True, "Prefill reuse must be disclosed")
    total = results["accountTotals"]
    require(
        (account_counts["scored"], account_counts["checks"], account_counts["runs"])
        == (total["scoredCount"], total["checkCount"], total["runCount"]),
        "Account totals do not match per-problem counts",
    )
    return results, jobs


class ParsedPage(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.ids = set()
        self.duplicate_ids = set()
        self.links = []
        self.h1_count = 0
        self.lang = None
        self.viewport = False
        self.has_title = False

    def handle_starttag(self, tag, attributes):
        attrs = dict(attributes)
        identifier = attrs.get("id")
        if identifier:
            if identifier in self.ids:
                self.duplicate_ids.add(identifier)
            self.ids.add(identifier)
        if tag == "html":
            self.lang = attrs.get("lang")
        if tag == "h1":
            self.h1_count += 1
        if tag == "title":
            self.has_title = True
        if tag == "meta" and attrs.get("name") == "viewport":
            self.viewport = True
        for name in ("href", "src"):
            if attrs.get(name):
                self.links.append((tag, attrs[name]))


def validate_site(root):
    docs = root / "docs"
    pages = {}
    for path in docs.rglob("*.html"):
        parsed = ParsedPage()
        parsed.feed(path.read_text(encoding="utf-8"))
        require(parsed.lang and parsed.viewport and parsed.has_title, "Missing page metadata: " + str(path.relative_to(root)))
        require(parsed.h1_count == 1, "Each page needs one h1: " + str(path.relative_to(root)))
        require(not parsed.duplicate_ids, "Duplicate HTML IDs in " + str(path.relative_to(root)))
        pages[path.resolve()] = parsed
    require(len(pages) >= 9, "Homepage, methodology, experiments, five articles and 404 are required")
    for path, page in pages.items():
        for tag, link in page.links:
            parsed = urlsplit(link)
            require(parsed.scheme not in {"javascript", "file"}, "Unsafe published URL")
            if parsed.scheme or parsed.netloc:
                require(tag != "script", "External runtime scripts are not permitted")
                continue
            require(not parsed.path.startswith("/"), "Site link breaks portable project hosting: " + link)
            target = (path.parent / unquote(parsed.path)).resolve() if parsed.path else path
            require(target.is_relative_to(docs.resolve()), "Site link escapes deployed docs/: " + link)
            if target.is_dir():
                target = target / "index.html"
            require(target.is_file(), "Broken site link in {}: {}".format(path.name, link))
            if parsed.fragment and target.suffix == ".html":
                require(unquote(parsed.fragment) in pages[target.resolve()].ids, "Broken page fragment: " + link)
    require((docs / ".nojekyll").is_file(), "Missing static Pages marker")
    return len(pages)


def validate_public_files(root):
    ignored = {".git", ".cache", "node_modules", "test-results", "playwright-report", "blob-report", "__pycache__"}
    for path in root.rglob("*"):
        relative = path.relative_to(root)
        if ignored.intersection(relative.parts) or not path.is_file():
            continue
        require(not path.is_symlink(), "Branch-based Pages cannot publish symbolic links")
        if path.suffix not in {".json", ".md", ".py", ".js", ".html", ".css", ".txt"}:
            continue
        text = path.read_text(encoding="utf-8")
        for label, pattern in SECRET_PATTERNS.items():
            require(not pattern.search(text), "{} detected in {}".format(label, relative))
        if path.suffix in {".json", ".md", ".html"}:
            require("/home/lokesh" not in text, "Private workspace path in " + str(relative))
            require(".copilot/session-state/" not in text, "Private session path in " + str(relative))
    for name in ("README.md", "PERFORMANCE.md", "EXPERIMENTS.md", "PROVENANCE.md", "verified_sources.json", "LICENSE", "NOTICE"):
        safe_path(root, name)


def main():
    try:
        results, jobs = validate_data(ROOT)
        pages = validate_site(ROOT)
        validate_public_files(ROOT)
    except (OSError, ValueError, KeyError, TypeError, SyntaxError) as error:
        print("Publication validation failed: " + str(error), file=sys.stderr)
        return 1
    print("Validated {} exact kernels, {} sanitized runs, and {} static pages.".format(len(results["kernels"]), len(jobs), pages))
    return 0


if __name__ == "__main__":
    sys.exit(main())
