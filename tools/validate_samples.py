#!/usr/bin/env python3
"""Validate the public sample catalog without requiring cloud access."""

from __future__ import annotations

import argparse
import datetime as dt
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import tomllib
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SAMPLES_DIR = ROOT / "samples"
CATALOG_PATH = ROOT / "samples.toml"
README_PATH = ROOT / "README.md"
INDEX_BEGIN = "<!-- BEGIN INDEX -->"
INDEX_END = "<!-- END INDEX -->"
README_HEADINGS = [
    "What this shows",
    "Prerequisites",
    "Assets",
    "Files and APIs",
    "Run it",
    "What to expect",
    "Inspecting results",
    "Troubleshooting",
    "Extending it",
    "Assets, licensing, attribution",
    "Validated against",
]
REQUIRED_FIELDS = {
    "slug": str,
    "title": str,
    "robot": str,
    "task": str,
    "concepts": list,
    "assets": list,
    "jobs": list,
    "gpu": bool,
    "difficulty": str,
    "learning_goal": str,
    "runtime_minutes": int,
    "est_cost_usd": float,
    "validated_against": str,
    "validated_on": str,
    "published": bool,
}
DIFFICULTIES = ("introductory", "intermediate", "advanced")
SAMPLE_FILES = {"app.py", "README.md", ".simuloignore"}
SLUG_PATTERN = re.compile(r"[a-z0-9][a-z0-9-]*\Z")
CLIENT_VERSION_PATTERN = re.compile(r"(\d+)\.(\d+)\.(\d+)\Z")


class ValidationError(Exception):
    """A catalog or repository structure violation."""


def fail(message: str) -> None:
    raise ValidationError(message)


def require_exact_type(value: Any, expected: type, label: str) -> None:
    if type(value) is not expected:
        fail(f"{label} must be a {expected.__name__}")


def is_compatible_client_version(value: str) -> bool:
    match = CLIENT_VERSION_PATTERN.fullmatch(value)
    if match is None:
        return False
    version = tuple(int(part) for part in match.groups())
    return (0, 23, 1) <= version < (0, 24, 0)


def load_catalog() -> list[dict[str, Any]]:
    try:
        catalog = tomllib.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as error:
        fail(f"samples.toml is invalid TOML: {error}")

    compat = catalog.get("compat")
    if not isinstance(compat, dict) or compat.get("simulo") != ">=0.23.1,<0.24":
        fail('samples.toml must contain [compat] simulo = ">=0.23.1,<0.24"')

    entries = catalog.get("samples", [])
    if not isinstance(entries, list):
        fail("samples must be an array of tables")
    return entries


def validate_entry(entry: Any, index: int) -> dict[str, Any]:
    if not isinstance(entry, dict):
        fail(f"samples[{index}] must be a table")

    for field, expected_type in REQUIRED_FIELDS.items():
        if field not in entry:
            fail(f"samples[{index}] is missing {field}")
        require_exact_type(entry[field], expected_type, f"samples[{index}].{field}")

    slug = entry["slug"]
    if not SLUG_PATTERN.fullmatch(slug):
        fail(f"samples[{index}].slug must be lowercase letters, digits, and hyphens")

    for field in ("title", "robot", "task", "learning_goal"):
        if not entry[field].strip():
            fail(f"samples[{index}].{field} must not be empty")

    for field in ("concepts", "assets", "jobs"):
        if not all(type(item) is str and item.strip() for item in entry[field]):
            fail(f"samples[{index}].{field} must contain only non-empty strings")
    if not entry["jobs"]:
        fail(f"samples[{index}].jobs must list at least one job")

    if entry["difficulty"] not in DIFFICULTIES:
        fail(f"samples[{index}].difficulty must be one of: {', '.join(DIFFICULTIES)}")

    if entry["published"]:
        if entry["runtime_minutes"] <= 0:
            fail(f"published samples[{index}].runtime_minutes must be greater than zero")
        if not math.isfinite(entry["est_cost_usd"]) or entry["est_cost_usd"] < 0:
            fail(f"published samples[{index}].est_cost_usd must be a finite value zero or greater")
        if not is_compatible_client_version(entry["validated_against"]):
            fail(
                f"published samples[{index}].validated_against must be a compatible client version"
            )
        if entry["validated_on"] == "unvalidated":
            fail(f"published samples[{index}].validated_on must be an ISO date")
        try:
            dt.date.fromisoformat(entry["validated_on"])
        except ValueError:
            raise ValidationError(
                f"published samples[{index}].validated_on must be an ISO date"
            ) from None

    return entry


def read_index_block(readme: str) -> str:
    begin = readme.find(INDEX_BEGIN)
    end = readme.find(INDEX_END)
    if begin == -1 or end == -1 or end < begin:
        fail("README.md must contain one ordered generated index block")
    if readme.find(INDEX_BEGIN, begin + 1) != -1 or readme.find(INDEX_END, end + 1) != -1:
        fail("README.md must contain exactly one generated index block")
    content_start = begin + len(INDEX_BEGIN)
    return readme[content_start:end]


def render_sample(entry: dict[str, Any]) -> str:
    concepts = ", ".join(entry["concepts"])
    return (
        f"- [{entry['title']}](samples/{entry['slug']}/): {entry['robot']}. "
        f"{entry['task']} Concepts: {concepts}. Runtime: {entry['runtime_minutes']} minutes. "
        f"Estimated cost: ${entry['est_cost_usd']:.2f}."
    )


def render_index(entries: list[dict[str, Any]]) -> str:
    published = [entry for entry in entries if entry["published"]]
    if not published:
        return "\nNo samples are published yet.\n"

    by_goal: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_difficulty: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for entry in published:
        by_goal[entry["learning_goal"]].append(entry)
        by_difficulty[entry["difficulty"]].append(entry)

    lines = ["", "### By learning goal", ""]
    for learning_goal, grouped_entries in by_goal.items():
        lines.extend((f"#### {learning_goal}", ""))
        lines.extend(render_sample(entry) for entry in grouped_entries)
        lines.append("")

    lines.extend(("### By difficulty", ""))
    for difficulty in DIFFICULTIES:
        grouped_entries = by_difficulty.get(difficulty, [])
        if not grouped_entries:
            continue
        lines.extend((f"#### {difficulty.capitalize()}", ""))
        lines.extend(render_sample(entry) for entry in grouped_entries)
        lines.append("")
    return "\n".join(lines)


def write_index(entries: list[dict[str, Any]]) -> None:
    readme = README_PATH.read_text(encoding="utf-8")
    block = render_index(entries)
    begin = readme.find(INDEX_BEGIN)
    end = readme.find(INDEX_END)
    if begin == -1 or end == -1 or end < begin:
        fail("README.md must contain one ordered generated index block")
    start = begin + len(INDEX_BEGIN)
    README_PATH.write_text(readme[:start] + block + readme[end:], encoding="utf-8")


def validate_directories(entries: list[dict[str, Any]]) -> None:
    slugs = [entry["slug"] for entry in entries]
    if len(slugs) != len(set(slugs)):
        fail("samples.toml contains duplicate slugs")

    directories = (
        {path.name: path for path in SAMPLES_DIR.iterdir() if path.is_dir()}
        if SAMPLES_DIR.is_dir()
        else {}
    )
    entries_by_slug = {entry["slug"]: entry for entry in entries}
    catalog_slugs = set(entries_by_slug)
    unpublished_directories = sorted(
        slug
        for slug in directories
        if slug in entries_by_slug and not entries_by_slug[slug]["published"]
    )
    if unpublished_directories:
        slug = unpublished_directories[0]
        fail(f"samples/{slug}/ exists but samples.toml marks it published = false")

    unknown_directories = sorted(set(directories) - catalog_slugs)
    published_slugs = {slug for slug, entry in entries_by_slug.items() if entry["published"]}
    missing_directories = sorted(published_slugs - set(directories))
    if unknown_directories:
        fail(f"sample directories missing from samples.toml: {', '.join(unknown_directories)}")
    if missing_directories:
        fail(f"samples.toml entries missing directories: {', '.join(missing_directories)}")

    for slug, directory in directories.items():
        files = {path.name for path in directory.iterdir()}
        if files != SAMPLE_FILES:
            expected = ", ".join(sorted(SAMPLE_FILES))
            actual = ", ".join(sorted(files)) or "(empty)"
            fail(f"samples/{slug}/ must contain exactly {expected}; found {actual}")
        for file_name in SAMPLE_FILES:
            if not (directory / file_name).is_file():
                fail(f"samples/{slug}/{file_name} must be a regular file")


def validate_readmes(entries: list[dict[str, Any]]) -> None:
    for entry in entries:
        if not entry["published"]:
            continue
        readme_path = SAMPLES_DIR / entry["slug"] / "README.md"
        headings = [
            match.group(1).strip()
            for line in readme_path.read_text(encoding="utf-8").splitlines()
            if (match := re.fullmatch(r"##\s+(.+?)\s*", line))
        ]
        if headings != README_HEADINGS:
            fail(
                f"samples/{entry['slug']}/README.md level-two headings must be: "
                + ", ".join(README_HEADINGS)
            )


def validate_index(entries: list[dict[str, Any]]) -> None:
    current = read_index_block(README_PATH.read_text(encoding="utf-8"))
    expected = render_index(entries)
    if current != expected:
        fail(
            "README.md generated index is stale; run python tools/validate_samples.py --write-index"
        )


def discover(entries: list[dict[str, Any]]) -> None:
    executable = shutil.which("simulo")
    if executable is None:
        fail("--discover requires the simulo command on PATH")

    environment = os.environ.copy()
    environment.pop("SIMULO_API_URL", None)
    environment.pop("SIMULO_API_TOKEN", None)
    environment.pop("SIMULO_ENV", None)
    checked = 0
    with tempfile.TemporaryDirectory(prefix="simulo-samples-home-") as home:
        environment["HOME"] = home
        for entry in entries:
            if not entry["published"]:
                continue
            app_path = SAMPLES_DIR / entry["slug"] / "app.py"
            package_dir = app_path.parent / ".simulo"
            try:
                for job in entry["jobs"]:
                    checked += 1
                    result = subprocess.run(
                        [executable, "run", str(app_path.relative_to(ROOT)), "--job", job],
                        cwd=ROOT,
                        env=environment,
                        text=True,
                        capture_output=True,
                        check=False,
                    )
                    if result.returncode != 0:
                        fail(
                            f"discovery failed for {entry['slug']} job {job!r} with exit "
                            f"{result.returncode}: {result.stderr.strip() or result.stdout.strip()}"
                        )
            finally:
                shutil.rmtree(package_dir, ignore_errors=True)

    # An unknown job exits 1 in the public command, so this packages every declared job
    # through the same path a user takes and turns an incorrect jobs[] entry red.
    if checked == 0:
        print("Zero samples were checked.")
    else:
        print(f"Discovery packaged {checked} declared jobs without executing them.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--discover",
        action="store_true",
        help="package every declared job with the installed public client",
    )
    parser.add_argument(
        "--write-index",
        action="store_true",
        help="regenerate the README index from samples.toml",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        entries = [validate_entry(entry, index) for index, entry in enumerate(load_catalog())]
        validate_directories(entries)
        if entries:
            validate_readmes(entries)
        if args.write_index:
            write_index(entries)
        validate_index(entries)
        if args.discover:
            discover(entries)
        elif not any(entry["published"] for entry in entries):
            print("Zero samples were checked.")
        else:
            print(f"Structure validated {sum(entry['published'] for entry in entries)} samples.")
    except (OSError, ValidationError) as error:
        print(f"Validation failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
