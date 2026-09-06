#!/usr/bin/env python3
"""Check the sample catalog and repository structure without cloud access."""

from __future__ import annotations

import argparse
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
ASSETS_DIR = ROOT / "assets"
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
    "published": bool,
}
DIFFICULTIES = ("introductory", "intermediate", "advanced")
SAMPLE_FILES = {"app.py", "README.md", ".simuloignore"}
IGNORED_SAMPLE_ENTRIES = {"__pycache__", ".simulo"}
SLUG_PATTERN = re.compile(r"[a-z0-9][a-z0-9-]*\Z")
# The kinds `simulo asset publish --kind` accepts. A catalog reference is
# `[<publisher>/]<kind>/<name>:v<N>`, so a directory at `assets/<kind>/<name>/`
# maps onto the reference a sample declares without any further bookkeeping.
ASSET_KINDS = ("robot", "world", "prop")
# The file formats the platform starts an asset package from. Each asset
# directory keeps exactly one of these at its own root, beside the meshes and
# other files it references, so `--entry <file>` is always a bare filename.
ASSET_ENTRY_SUFFIXES = (".urdf", ".usd", ".usda", ".usdc", ".usdz")
# The platform assigns asset versions as v1, v2, and so on.
ASSET_VERSION_PATTERN = re.compile(r"v[1-9][0-9]*\Z")
ASSETS_README = "README.md"


class ValidationError(Exception):
    """A catalog or repository structure violation."""


def fail(message: str) -> None:
    raise ValidationError(message)


def require_exact_type(value: Any, expected: type, label: str) -> None:
    if type(value) is not expected:
        fail(f"{label} must be a {expected.__name__}")


def load_catalog() -> list[dict[str, Any]]:
    try:
        catalog = tomllib.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as error:
        fail(f"samples.toml is invalid TOML: {error}")

    compat = catalog.get("compat")
    if not isinstance(compat, dict) or compat.get("simulo") != ">=0.23.1,<0.26":
        fail('samples.toml must contain [compat] simulo = ">=0.23.1,<0.26"')

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

    # The generated index prints this for every row, so a zero or negative value
    # would render as nonsense.
    if entry["runtime_minutes"] <= 0:
        fail(f"samples[{index}].runtime_minutes must be greater than zero")

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
    assets = ", ".join(f"`{asset}`" for asset in entry["assets"]) or "none"
    runtime_unit = "minute" if entry["runtime_minutes"] == 1 else "minutes"
    hardware = (
        "L4-class cloud GPU; no local GPU required"
        if entry["gpu"]
        else "cloud CPU; no GPU requested"
    )
    return (
        f"- [{entry['title']}](samples/{entry['slug']}/): {entry['robot']}. "
        f"{entry['task']} Concepts: {concepts}. Assets: {assets}. Hardware: {hardware}. "
        f"Runtime: about {entry['runtime_minutes']} {runtime_unit}."
    )


def render_index(entries: list[dict[str, Any]]) -> str:
    visible = present_entries(entries)
    if not visible:
        return "\nNo sample directories are available in this checkout.\n"

    by_goal: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_difficulty: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for entry in visible:
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

    unknown_directories = sorted(set(directories) - catalog_slugs)
    published_slugs = {slug for slug, entry in entries_by_slug.items() if entry["published"]}
    missing_directories = sorted(published_slugs - set(directories))
    if unknown_directories:
        fail(f"sample directories missing from samples.toml: {', '.join(unknown_directories)}")
    if missing_directories:
        fail(f"samples.toml entries missing directories: {', '.join(missing_directories)}")

    for slug, directory in directories.items():
        # Importing a sample, which `simulo run` and therefore --discover do, leaves
        # __pycache__/ behind, and an offline `simulo run` leaves .simulo/. Both are
        # ignored by git and neither is part of the sample.
        present = {path.name for path in directory.iterdir()} - IGNORED_SAMPLE_ENTRIES
        if present != SAMPLE_FILES:
            expected = ", ".join(sorted(SAMPLE_FILES))
            actual = ", ".join(sorted(present)) or "(empty)"
            fail(f"samples/{slug}/ must contain exactly {expected}; found {actual}")
        for file_name in SAMPLE_FILES:
            if not (directory / file_name).is_file():
                fail(f"samples/{slug}/{file_name} must be a regular file")


def parse_asset_reference(reference: str, label: str) -> tuple[str, str, bool] | None:
    """Split a catalog reference into its kind, its name, and whether it omits a publisher.

    Returns None for a reference that names something other than a catalog asset. The
    assets column carries those too: `simulo/gpu-rl:2026.06` is a job runtime, and it has
    the same two-segment shape as the publisher-less asset reference
    `robot/byo-urdf-arm:v1`. The version is what separates them, because the catalog
    assigns asset versions as v1, v2, and so on while a runtime carries its image's
    calendar version. Every entry therefore has to name a version for this to decide
    anything.
    """
    body, separator, version = reference.partition(":")
    if not separator or not version:
        fail(f"{label} must name a version: {reference}")
    if not ASSET_VERSION_PATTERN.fullmatch(version):
        return None

    segments = body.split("/")
    if len(segments) == 2:
        kind, name = segments
        publisher_omitted = True
    elif len(segments) == 3:
        _publisher, kind, name = segments
        publisher_omitted = False
    else:
        fail(f"{label} must read [<publisher>/]<kind>/<name>:v<N>: {reference}")

    if kind not in ASSET_KINDS:
        fail(f"{label} names the kind {kind!r}, which is not one of: {', '.join(ASSET_KINDS)}")
    return kind, name, publisher_omitted


def declared_assets(entries: list[dict[str, Any]]) -> dict[tuple[str, str], list[str]]:
    """Publisher-less catalog references, each mapped to the samples that name it.

    A reference with a publisher, such as `simulo/robot/cartpole:v1`, names somebody
    else's catalog and this repository ships nothing for it. A reference without one
    resolves against whichever organization the reader is signed in as, so the files to
    publish have to be here.
    """
    declared: dict[tuple[str, str], list[str]] = defaultdict(list)
    for entry in entries:
        for reference in entry["assets"]:
            parsed = parse_asset_reference(reference, f"samples[{entry['slug']}].assets")
            if parsed is None:
                continue
            kind, name, publisher_omitted = parsed
            if publisher_omitted:
                declared[(kind, name)].append(entry["slug"])
    return declared


def present_assets() -> dict[tuple[str, str], Path]:
    """Every asset package directory in the tree, with its shape checked as it goes."""
    if not ASSETS_DIR.is_dir():
        return {}

    packages: dict[tuple[str, str], Path] = {}
    for kind_path in sorted(ASSETS_DIR.iterdir()):
        if kind_path.is_file() and kind_path.name == ASSETS_README:
            continue
        if not kind_path.is_dir():
            fail(f"assets/{kind_path.name} must be a directory named for an asset kind")
        if kind_path.name not in ASSET_KINDS:
            fail(
                f"assets/{kind_path.name}/ does not name an asset kind; "
                f"the accepted kinds are: {', '.join(ASSET_KINDS)}"
            )
        for asset_path in sorted(kind_path.iterdir()):
            if not asset_path.is_dir():
                fail(
                    f"assets/{kind_path.name}/{asset_path.name} must be a directory: one "
                    "asset is one directory, published as a whole"
                )
            has_entry = any(
                child.is_file() and child.suffix.lower() in ASSET_ENTRY_SUFFIXES
                for child in asset_path.iterdir()
            )
            if not has_entry:
                fail(
                    f"assets/{kind_path.name}/{asset_path.name}/ has no entry file at its "
                    f"root; the platform starts from one file ending in "
                    f"{', '.join(ASSET_ENTRY_SUFFIXES)}"
                )
            packages[(kind_path.name, asset_path.name)] = asset_path
    return packages


def validate_assets(entries: list[dict[str, Any]]) -> int:
    """Match the assets tree against the references the catalog declares, both ways."""
    declared = declared_assets(entries)
    packages = present_assets()

    missing = sorted(f"{kind}/{name}" for kind, name in set(declared) - set(packages))
    if missing:
        fail(
            "these references name no publisher, so they resolve against the reader's own "
            "organization and this repository has to ship the files to publish, but they "
            f"have no assets/<kind>/<name>/ directory: {', '.join(missing)}"
        )

    orphaned = sorted(f"{kind}/{name}" for kind, name in set(packages) - set(declared))
    if orphaned:
        fail(
            "no sample names these asset directories in its samples.toml assets list, so "
            f"nothing in this repository tells a reader to publish them: {', '.join(orphaned)}"
        )
    return len(packages)


def present_entries(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The entries whose sample directory exists in this checkout."""
    return [entry for entry in entries if (SAMPLES_DIR / entry["slug"]).is_dir()]


def validate_readmes(entries: list[dict[str, Any]]) -> None:
    for entry in present_entries(entries):
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
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    checked = 0
    with tempfile.TemporaryDirectory(prefix="simulo-samples-home-") as home:
        environment["HOME"] = home
        for entry in present_entries(entries):
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
        assets = validate_assets(entries)
        if entries:
            validate_readmes(entries)
        if args.write_index:
            write_index(entries)
        validate_index(entries)
        if args.discover:
            discover(entries)
        elif not present_entries(entries):
            print("Zero samples were checked.")
        else:
            noun = "asset" if assets == 1 else "assets"
            print(f"Structure checked {len(present_entries(entries))} samples and {assets} {noun}.")
    except (OSError, ValidationError) as error:
        print(f"Check failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
