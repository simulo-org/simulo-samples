#!/usr/bin/env python3
"""Unit tests for the samples structure checker."""

from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

MODULE_PATH = Path(__file__).with_name("validate_samples.py")
SPEC = importlib.util.spec_from_file_location("validate_samples", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
validate_samples = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(validate_samples)


class ValidateDirectoriesTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.samples_dir = self.root / "samples"
        self.samples_dir.mkdir()
        self.original_samples_dir = validate_samples.SAMPLES_DIR
        validate_samples.SAMPLES_DIR = self.samples_dir

    def tearDown(self) -> None:
        validate_samples.SAMPLES_DIR = self.original_samples_dir
        self.temporary_directory.cleanup()

    def sample_directory(self, files: set[str]) -> Path:
        directory = self.samples_dir / "example"
        directory.mkdir()
        for name in files:
            (directory / name).write_text("", encoding="utf-8")
        return directory

    def test_reports_every_missing_required_entry_point(self) -> None:
        self.sample_directory({"README.md", ".simuloignore", "train.py", "eval.py"})

        with self.assertRaisesRegex(
            validate_samples.ValidationError,
            r"(?s)requires task.py, preview.py, train.py, eval.py, and play.py.*"
            r"missing play.py, preview.py, task.py",
        ):
            validate_samples.validate_directories([{"slug": "example", "published": False}])

    def test_accepts_the_complete_sample_file_set(self) -> None:
        self.sample_directory(validate_samples.SAMPLE_FILES)

        validate_samples.validate_directories([{"slug": "example", "published": False}])


class EvaluationValidationTests(unittest.TestCase):
    def test_finds_calls_but_not_comments_or_strings(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "eval.py"
            path.write_text(
                "# task.get_observations()\nmessage = 'get_observations()'\n",
                encoding="utf-8",
            )
            self.assertEqual(validate_samples.find_observation_calls(path), [])

            path.write_text("task.get_observations()\n", encoding="utf-8")
            self.assertEqual(validate_samples.find_observation_calls(path), [1])


class CatalogValidationTests(unittest.TestCase):
    def test_jobs_means_the_one_training_job(self) -> None:
        entry = {
            "slug": "example",
            "title": "Example",
            "robot": "Robot",
            "task": "Task",
            "concepts": ["Concept"],
            "assets": [],
            "jobs": ["train"],
            "gpu": False,
            "difficulty": "introductory",
            "learning_goal": "Learn",
            "runtime_minutes": 1,
            "published": False,
        }
        self.assertEqual(validate_samples.validate_entry(entry, 0), entry)

        entry["jobs"] = ["train", "preview"]
        with self.assertRaisesRegex(validate_samples.ValidationError, "exactly one job"):
            validate_samples.validate_entry(entry, 0)


if __name__ == "__main__":
    unittest.main()
