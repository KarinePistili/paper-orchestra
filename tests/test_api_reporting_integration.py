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

import importlib.util
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from utils.api_reporting import api_stage, report_api_run


ROOT = Path(__file__).resolve().parents[1]


def load_module(name, filename, dependencies):
    spec = importlib.util.spec_from_file_location(name, ROOT / filename)
    module = importlib.util.module_from_spec(spec)
    with (
        patch.dict(sys.modules, dependencies),
        patch.dict(os.environ, {}, clear=True),
        patch("dotenv.load_dotenv"),
    ):
        spec.loader.exec_module(module)
    return module


class ApiReportingIntegrationTest(unittest.TestCase):
    def test_gemini_text_retry_and_image(self):
        google = types.ModuleType("google")
        genai = types.ModuleType("google.genai")
        genai.Client = object
        gemini_types = types.ModuleType("google.genai.types")
        gemini_types.Content = object
        gemini_types.GenerateContentConfig = lambda **kwargs: kwargs
        gemini_types.ImageConfig = lambda **kwargs: kwargs
        gemini_types.Part = types.SimpleNamespace(from_text=lambda **kwargs: kwargs)
        google.genai = genai
        genai.types = gemini_types
        module = load_module(
            "utils.test_gemini",
            "utils/gemini_utils.py",
            {"google": google, "google.genai": genai, "google.genai.types": gemini_types},
        )
        text_response = types.SimpleNamespace(
            text='{"ok": true}',
            usage_metadata=types.SimpleNamespace(prompt_token_count=12, total_token_count=20),
        )
        image_response = types.SimpleNamespace(
            candidates=[
                types.SimpleNamespace(
                    content=types.SimpleNamespace(
                        parts=[
                            types.SimpleNamespace(
                                inline_data=types.SimpleNamespace(data=b"image")
                            )
                        ]
                    )
                )
            ],
            usage_metadata=types.SimpleNamespace(total_token_count=30),
        )
        generate = Mock(side_effect=[RuntimeError("private"), text_response, image_response])
        module.get_genai_client = Mock(return_value=types.SimpleNamespace(
            models=types.SimpleNamespace(generate_content=generate)
        ))

        @report_api_run
        @api_stage("figure_generation")
        def pipeline(base_dir):
            module.call_gemini_with_contents(
                contents=[], model_name="text-model", max_retries=2, base_interval_sec=0
            )
            module.generate_image_with_gemini(
                model_name="image-model", prompt="private prompt", max_retries=1
            )

        with tempfile.TemporaryDirectory() as directory:
            pipeline(directory)
            contents = next(Path(directory).glob("api_requests_*.json")).read_text()
            report = json.loads(contents)

        self.assertNotIn("private", contents)
        self.assertEqual(len(report["requests"]), 3)
        self.assertEqual(
            [x["status"] for x in report["requests"]],
            ["failed", "succeeded", "succeeded"],
        )
        self.assertEqual(report["requests"][0]["request_id"], report["requests"][1]["request_id"])
        self.assertEqual(report["requests"][2]["model"], "image-model")
        self.assertEqual(report["requests"][2]["tokens"]["total"], 30)

    def test_openai_and_semantic_scholar(self):
        openai = types.ModuleType("openai")
        openai.OpenAI = object
        cv2 = types.ModuleType("cv2")
        module = load_module(
            "utils.test_openai", "utils/openai_utils.py", {"openai": openai, "cv2": cv2}
        )
        response = types.SimpleNamespace(
            choices=[types.SimpleNamespace(message=types.SimpleNamespace(content='{"ok": true}'))],
            usage=types.SimpleNamespace(prompt_tokens=7, completion_tokens=3, total_tokens=10),
        )
        create = Mock(return_value=response)
        module.get_openai_client = Mock(
            return_value=types.SimpleNamespace(
                chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=create))
            )
        )
        fuzz = types.ModuleType("thefuzz")
        fuzz.fuzz = types.SimpleNamespace(ratio=lambda *args: 100)
        scholar = load_module("utils.test_scholar", "utils/scholar_utils.py", {"thefuzz": fuzz})

        @report_api_run
        @api_stage("literature_review")
        def pipeline(base_dir):
            module.call_openai_models_with_content([], "model", max_retries=1)
            with patch.object(
                scholar.requests, "get", return_value=types.SimpleNamespace(status_code=429)
            ):
                self.assertIsNone(scholar.s2_title_search("private title"))

        with tempfile.TemporaryDirectory() as directory:
            pipeline(directory)
            contents = next(Path(directory).glob("api_requests_*.json")).read_text()
            report = json.loads(contents)

        self.assertNotIn("private title", contents)
        self.assertEqual(
            [x["provider"] for x in report["requests"]], ["openai", "semantic_scholar"]
        )
        self.assertEqual(report["requests"][0]["tokens"]["total"], 10)
        self.assertEqual(report["requests"][1]["http_status"], 429)
        self.assertEqual(report["requests"][1]["status"], "failed")


if __name__ == "__main__":
    unittest.main()
