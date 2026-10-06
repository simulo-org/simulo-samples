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


TRAIN_WITH_SPACING = """import simulo
from task import ExampleTask, app


@app.job(type="train")
def train(num_envs: int = 16):
    env = simulo.LearningEnv(task=ExampleTask(), num_envs=num_envs, env_spacing={spacing})
    return env
"""
TRAIN_WITH_SHARED_HELPER = """from task import _train, app


@app.job(type="train")
def train(num_envs: int = 16):
    return _train(num_envs)
"""
TASK_WITH_HELPER = """import simulo


def _make_env(num_envs):
    return simulo.LearningEnv(task=ExampleTask(), num_envs=num_envs, env_spacing={spacing})


def _train(num_envs):
    return _make_env(num_envs)
"""
EVAL_WITH_SPACING = """import simulo
from task import ExampleTask, app


@app.job(type="eval")
def evaluate(policy, episodes: int = 10):
    return simulo.evaluate(ExampleTask(), policy, episodes=episodes{keywords})
"""


class EnvSpacingValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.samples_dir = self.root / "samples"
        self.directory = self.samples_dir / "example"
        self.directory.mkdir(parents=True)
        self.original_root = validate_samples.ROOT
        self.original_samples_dir = validate_samples.SAMPLES_DIR
        validate_samples.ROOT = self.root
        validate_samples.SAMPLES_DIR = self.samples_dir

    def tearDown(self) -> None:
        validate_samples.ROOT = self.original_root
        validate_samples.SAMPLES_DIR = self.original_samples_dir
        self.temporary_directory.cleanup()

    def write_sample(self, *, train: str, eval_keywords: str, task: str = "") -> None:
        (self.directory / "train.py").write_text(train, encoding="utf-8")
        (self.directory / "task.py").write_text(task, encoding="utf-8")
        (self.directory / "eval.py").write_text(
            EVAL_WITH_SPACING.format(keywords=eval_keywords), encoding="utf-8"
        )

    def validate(self) -> None:
        validate_samples.validate_env_spacing([{"slug": "example"}])

    def test_accepts_evaluation_spacing_equal_to_training(self) -> None:
        self.write_sample(
            train=TRAIN_WITH_SPACING.format(spacing="2.5"), eval_keywords=", env_spacing=2.5"
        )
        self.validate()

    def test_accepts_an_integer_and_a_float_of_the_same_value(self) -> None:
        self.write_sample(
            train=TRAIN_WITH_SPACING.format(spacing="2"), eval_keywords=", env_spacing=2.0"
        )
        self.validate()

    def test_rejects_evaluation_spacing_that_differs_from_training(self) -> None:
        self.write_sample(
            train=TRAIN_WITH_SPACING.format(spacing="2.5"), eval_keywords=", env_spacing=4.0"
        )
        with self.assertRaisesRegex(
            validate_samples.ValidationError,
            r"(?s)same env_spacing as training.*samples/example/train.py trains with "
            r"env_spacing=2.5 but samples/example/eval.py evaluates with env_spacing=4",
        ):
            self.validate()

    def test_evaluation_without_spacing_uses_the_client_default_of_four(self) -> None:
        self.assertEqual(validate_samples.EVALUATE_DEFAULT_ENV_SPACING, 4.0)
        self.write_sample(train=TRAIN_WITH_SPACING.format(spacing="4.0"), eval_keywords="")
        self.validate()

        self.write_sample(train=TRAIN_WITH_SPACING.format(spacing="7.0"), eval_keywords="")
        with self.assertRaisesRegex(
            validate_samples.ValidationError,
            r"trains with env_spacing=7 but samples/example/eval.py evaluates with env_spacing=4",
        ):
            self.validate()

    def test_reads_training_spacing_from_a_task_helper_when_train_builds_no_environment(
        self,
    ) -> None:
        self.write_sample(
            train=TRAIN_WITH_SHARED_HELPER,
            task=TASK_WITH_HELPER.format(spacing="7.0"),
            eval_keywords=", env_spacing=7.0",
        )
        self.validate()

        self.write_sample(
            train=TRAIN_WITH_SHARED_HELPER,
            task=TASK_WITH_HELPER.format(spacing="7.0"),
            eval_keywords=", env_spacing=4.0",
        )
        with self.assertRaisesRegex(
            validate_samples.ValidationError, r"samples/example/task.py trains with env_spacing=7"
        ):
            self.validate()

    def test_rejects_training_spacing_that_is_not_a_literal(self) -> None:
        self.write_sample(
            train=TRAIN_WITH_SPACING.format(spacing="SPACING"), eval_keywords=", env_spacing=4.0"
        )
        with self.assertRaisesRegex(
            validate_samples.ValidationError,
            r"samples/example/train.py:\d+ sets env_spacing to something other than a "
            r"positive number",
        ):
            self.validate()

    def test_rejects_training_without_explicit_spacing(self) -> None:
        self.write_sample(
            train=TRAIN_WITH_SPACING.replace(", env_spacing={spacing}", ""),
            eval_keywords=", env_spacing=4.0",
        )
        with self.assertRaisesRegex(
            validate_samples.ValidationError,
            r"samples/example/train.py:\d+ does not pass env_spacing",
        ):
            self.validate()

    def test_rejects_training_that_builds_no_environment_anywhere(self) -> None:
        self.write_sample(train=TRAIN_WITH_SHARED_HELPER, eval_keywords=", env_spacing=4.0")
        with self.assertRaisesRegex(
            validate_samples.ValidationError, r"training env_spacing cannot be found"
        ):
            self.validate()

    def test_rejects_evaluation_spacing_that_is_not_a_literal(self) -> None:
        self.write_sample(
            train=TRAIN_WITH_SPACING.format(spacing="4.0"), eval_keywords=", env_spacing=spacing"
        )
        with self.assertRaisesRegex(
            validate_samples.ValidationError,
            r"samples/example/eval.py:\d+ sets env_spacing to something other than a "
            r"positive number",
        ):
            self.validate()

    def test_rejects_evaluation_keywords_passed_by_unpacking(self) -> None:
        self.write_sample(
            train=TRAIN_WITH_SPACING.format(spacing="4.0"), eval_keywords=", **settings"
        )
        with self.assertRaisesRegex(
            validate_samples.ValidationError,
            r"samples/example/eval.py:\d+ passes \*\*keyword arguments",
        ):
            self.validate()

    def test_rejects_evaluation_that_never_calls_simulo_evaluate(self) -> None:
        self.write_sample(train=TRAIN_WITH_SPACING.format(spacing="4.0"), eval_keywords="")
        eval_path = self.directory / "eval.py"
        eval_path.write_text(
            eval_path.read_text(encoding="utf-8").replace("simulo.evaluate(", "run("),
            encoding="utf-8",
        )
        with self.assertRaisesRegex(
            validate_samples.ValidationError,
            r"samples/example/eval.py must call simulo.evaluate\(...\) exactly once.*found 0 calls",
        ):
            self.validate()


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
