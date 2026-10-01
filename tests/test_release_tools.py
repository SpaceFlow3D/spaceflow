"""CPU regression checks for release validation and actionable Blender errors."""

import importlib.util
import contextlib
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock
from types import SimpleNamespace

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "tools"))
from validate_examples import validate_npz
from verify_release import prepare_matrix
from lib.util.pipeline_compat import can_reuse_appearance_pipeline
from lib.util.model_revisions import dinov2_hub_repository
from lib.util.checkpoint_compat import partfield_checkpoint_scope


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class PrimitiveValidationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.path = Path(self.temporary.name) / "all.npz"
        self.arrays = {
            "scales": np.ones((1, 3)), "shapes": np.ones((1, 2)),
            "translations": np.zeros((1, 3)), "rotations": np.eye(3)[None],
        }

    def tearDown(self):
        self.temporary.cleanup()

    def test_valid_numeric_bundle(self):
        np.savez(self.path, **self.arrays)
        self.assertEqual(validate_npz(self.path), 1)

    def test_nonfinite_geometry_is_rejected(self):
        self.arrays["translations"][0, 0] = np.nan
        np.savez(self.path, **self.arrays)
        with self.assertRaisesRegex(ValueError, "finite"):
            validate_npz(self.path)

    def test_wrong_shape_is_rejected(self):
        self.arrays["rotations"] = np.ones((1, 3))
        np.savez(self.path, **self.arrays)
        with self.assertRaisesRegex(ValueError, "rotations"):
            validate_npz(self.path)

    def test_pickled_geometry_is_rejected(self):
        self.arrays["scales"] = np.array([[1, 1, 1]], dtype=object)
        np.savez(self.path, **self.arrays)
        with self.assertRaisesRegex(ValueError, "Object arrays"):
            validate_npz(self.path)


class AppearanceCompatibilityTests(unittest.TestCase):
    def test_text_pipeline_cannot_be_reused_for_images(self):
        self.assertFalse(can_reuse_appearance_pipeline(object(), "image"))

    def test_image_pipeline_can_be_reused_for_images(self):
        class ImagePipeline:
            def preprocess_image(self, image):
                return image
        self.assertTrue(can_reuse_appearance_pipeline(ImagePipeline(), "image"))

    def test_text_reuse_is_preserved(self):
        self.assertTrue(can_reuse_appearance_pipeline(object(), "text"))
        self.assertFalse(can_reuse_appearance_pipeline(None, "text"))


class PartFieldCheckpointTests(unittest.TestCase):
    def scoped_modules(self):
        self.config_class = type("CfgNode", (), {})
        self.scope = mock.MagicMock()
        self.allow = mock.Mock(return_value=self.scope)
        return {
            "torch": SimpleNamespace(serialization=SimpleNamespace(safe_globals=self.allow)),
            "yacs": SimpleNamespace(),
            "yacs.config": SimpleNamespace(CfgNode=self.config_class),
        }

    def test_only_the_yacs_config_class_is_allowed_during_loading(self):
        with mock.patch.dict(sys.modules, self.scoped_modules()):
            with partfield_checkpoint_scope():
                self.allow.assert_called_once_with([self.config_class])
                self.scope.__enter__.assert_called_once()
                self.scope.__exit__.assert_not_called()
        self.scope.__exit__.assert_called_once_with(None, None, None)

    def test_the_scope_closes_when_checkpoint_loading_fails(self):
        with mock.patch.dict(sys.modules, self.scoped_modules()):
            with self.assertRaisesRegex(ValueError, "checkpoint"):
                with partfield_checkpoint_scope():
                    raise ValueError("checkpoint")
        self.scope.__exit__.assert_called_once()
        self.assertIs(self.scope.__exit__.call_args.args[0], ValueError)


class ModelRevisionTests(unittest.TestCase):
    def test_dinov2_uses_an_immutable_upstream_commit(self):
        self.assertRegex(dinov2_hub_repository(), r"^facebookresearch/dinov2:[0-9a-f]{40}$")

    def test_moving_dinov2_revision_is_rejected(self):
        import json
        with tempfile.TemporaryDirectory() as temporary:
            pins = Path(temporary) / "pins.json"
            pins.write_text(json.dumps({"dinov2": {"repository": "facebookresearch/dinov2", "revision": "main"}}))
            with self.assertRaisesRegex(ValueError, "full commit SHA"):
                dinov2_hub_repository(pins)


class GenerationPreflightTests(unittest.TestCase):
    def setUp(self):
        self.doctor = load_module("release_doctor", REPO_ROOT / "tools/doctor.py")
        self.torch = SimpleNamespace(__version__="2.8.0", version=SimpleNamespace(cuda="12.8"),
                                     cuda=SimpleNamespace(is_available=lambda: False))

    def fake_import(self, name):
        if name == "torch":
            return self.torch
        if name == "run_local_tau":
            print("TRELLIS backend announcement")
        return SimpleNamespace(__version__="test")

    def test_missing_sklearn_is_reported_with_runtime_install_action(self):
        def importing(name):
            if name == "sklearn":
                raise ModuleNotFoundError("No module named 'sklearn'")
            return self.fake_import(name)
        with mock.patch.object(self.doctor.importlib, "import_module", side_effect=importing), \
                mock.patch.object(self.doctor.shutil, "which", return_value=None):
            report = self.doctor.inspect_environment(gpu=True)
        check = next(item for item in report["checks"] if item["name"] == "sklearn")
        self.assertFalse(check["passed"])
        self.assertIn("SPACEFLOW_SETUP_STAGE=deps", check["action"])

    def test_entrypoint_import_does_not_pollute_json_stdout(self):
        output = io.StringIO()
        with mock.patch.object(self.doctor.importlib, "import_module", side_effect=self.fake_import), \
                mock.patch.object(self.doctor.shutil, "which", return_value=None), \
                contextlib.redirect_stdout(output):
            report = self.doctor.inspect_environment(gpu=True)
        self.assertEqual(output.getvalue(), "")
        self.assertTrue(next(item for item in report["checks"] if item["name"] == "run_local_tau")["passed"])


class ReleaseMatrixTests(unittest.TestCase):
    def test_prepared_matrix_keeps_full_steps_and_baselines(self):
        import json
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "matrix"
            cases = prepare_matrix(output, REPO_ROOT / "docs/media/sailboat_spin_poster.png")
            self.assertEqual(len(cases), 5)
            self.assertTrue(all(case["status"] == "prepared" for case in cases))
            for case in cases:
                config = json.loads(Path(case["config"]).read_text())
                for variant in config["variants"]:
                    if "argv" in variant:
                        argv = variant["argv"]
                        self.assertEqual(argv[argv.index("--texture_optim_steps") + 1], "300")
                if case["name"] == "comparisons-teacup":
                    self.assertEqual(len(config["variants"]), 7)
                if case["name"] == "image-sailboat":
                    argv = config["variants"][0]["argv"]
                    self.assertNotIn("--appearance_text", argv)
                    self.assertNotIn("--local_text_prompts", argv)
                    self.assertTrue(Path(argv[argv.index("--appearance_image") + 1]).is_file())
                    self.assertTrue(json.loads((Path(case["config"]).parent / "replay_provenance.json").read_text())["parameters_changed"])

    def test_missing_image_does_not_create_a_run(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "matrix"
            with self.assertRaises(FileNotFoundError):
                prepare_matrix(output, Path(temporary) / "missing.png")
            self.assertFalse(output.exists())


class BlenderErrorTests(unittest.TestCase):
    def setUp(self):
        self.render = load_module("release_render", REPO_ROOT / "lib/util/render.py")

    def test_missing_explicit_blender_has_an_actionable_error(self):
        with mock.patch.dict("os.environ", {"SPACEFLOW_BLENDER_PATH": "/missing/blender"}), \
                mock.patch.object(self.render.shutil, "which", return_value=None):
            with self.assertRaisesRegex(FileNotFoundError, "SPACEFLOW_BLENDER_PATH"):
                self.render._install_blender()

    def test_failed_blender_process_exposes_stderr(self):
        result = mock.Mock(returncode=7, stderr="renderer initialization failed")
        with mock.patch.object(self.render.subprocess, "run", return_value=result):
            with self.assertRaisesRegex(RuntimeError, "renderer initialization failed"):
                self.render._run_blender(["blender"])


if __name__ == "__main__":
    unittest.main()
