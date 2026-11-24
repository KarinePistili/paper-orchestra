# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import contextlib
import io
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import paper_writing_cli


class ImageModelConfigurationTest(unittest.TestCase):
    def run_cli(self, dotenv_content="", environment=None, extra_args=()):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / ".env").write_text(dotenv_content)
            (root / "raw_materials").mkdir()
            writer = Mock()
            writer_module = types.ModuleType("methods.paper_writer_with_plotting")
            writer_module.write_single_paper = writer
            args = [
                "paper_writing_cli.py",
                "--raw_materials_dir",
                str(root / "raw_materials"),
                "--output_dir",
                str(root),
                "--latex_template_dir",
                str(root / "template"),
                "--use_plotting",
                "true",
                *extra_args,
            ]
            with (
                patch.object(
                    paper_writing_cli, "__file__", str(root / "paper_writing_cli.py")
                ),
                patch.dict(os.environ, environment or {}, clear=True),
                patch.dict(
                    sys.modules, {"methods.paper_writer_with_plotting": writer_module}
                ),
                patch.object(sys, "argv", args),
                contextlib.redirect_stdout(io.StringIO()),
                contextlib.redirect_stderr(io.StringIO()),
            ):
                paper_writing_cli.main()
            return writer.call_args.kwargs["image_model_name"]

    def test_default_ignores_environment_and_dotenv(self):
        self.assertEqual(self.run_cli(), "gemini-3-pro-image")
        self.assertEqual(
            self.run_cli(
                "IMAGE_MODEL_NAME=dotenv-model\n",
                environment={"IMAGE_MODEL_NAME": "environment-model"},
            ),
            "gemini-3-pro-image",
        )

    def test_cli_selects_image_model(self):
        self.assertEqual(
            self.run_cli(
                extra_args=("--image_model_name", "gemini-3.1-flash-lite-image"),
            ),
            "gemini-3.1-flash-lite-image",
        )

    def test_empty_model_is_rejected(self):
        with self.assertRaises(SystemExit) as error:
            self.run_cli(extra_args=("--image_model_name", ""))
        self.assertEqual(error.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
