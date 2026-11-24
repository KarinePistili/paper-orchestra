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

import json
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

from utils.api_reporting import api_attempt, api_stage, report_api_run, submit_with_context


class ApiReportingTest(unittest.TestCase):
    def test_parallel_stages_retries_and_usage(self):
        @api_stage("literature_review")
        def literature():
            try:
                with api_attempt("gemini", "generate_content", "text-model", "req-1", 1):
                    raise ValueError("private request details")
            except ValueError:
                pass
            with api_attempt("gemini", "generate_content", "text-model", "req-1", 2) as attempt:
                attempt.set_response(
                    SimpleNamespace(
                        usage_metadata=SimpleNamespace(
                            prompt_token_count=100,
                            candidates_token_count=20,
                            thoughts_token_count=5,
                            total_token_count=125,
                        )
                    )
                )
            with api_attempt("semantic_scholar", "paper_search", None, "req-2", 1) as attempt:
                attempt.set_response(SimpleNamespace(status_code=429))

        @api_stage("figure_generation")
        def figure():
            with ThreadPoolExecutor(max_workers=1) as executor:
                submit_with_context(executor, image_request).result()

        def image_request():
            with api_attempt("gemini", "generate_image", "image-model", "req-3", 1) as attempt:
                attempt.set_response(SimpleNamespace(usage_metadata=None))

        @report_api_run
        def pipeline(base_dir):
            with ThreadPoolExecutor(max_workers=2) as executor:
                futures = [
                    submit_with_context(executor, literature),
                    submit_with_context(executor, figure),
                ]
                for future in futures:
                    future.result()

        with tempfile.TemporaryDirectory() as directory:
            pipeline(directory)
            path = next(Path(directory).glob("api_requests_*.json"))
            contents = path.read_text()
            report = json.loads(contents)

        self.assertNotIn("private request details", contents)
        self.assertIsNotNone(report["finished_at"])
        self.assertIsNone(report["estimated_cost_usd"])
        stages = {entry["stage"]: entry for entry in report["stage_summary"]}
        self.assertEqual(stages["literature_review"]["attempts"], 3)
        self.assertEqual(stages["literature_review"]["logical_requests"], 2)
        self.assertEqual(stages["literature_review"]["retries"], 1)
        self.assertEqual(stages["literature_review"]["failed"], 2)
        self.assertEqual(stages["literature_review"]["tokens"]["thinking"], 5)
        self.assertEqual(stages["literature_review"]["usage_unavailable"], 2)
        self.assertEqual(stages["figure_generation"]["attempts"], 1)
        self.assertIsNone(stages["figure_generation"]["tokens"])
        errors = [item for item in report["requests"] if item["status"] == "failed"]
        self.assertEqual({item["error_type"] for item in errors}, {"ValueError", None})
        self.assertIn(429, [item["http_status"] for item in errors])

    def test_report_survives_pipeline_exception(self):
        @report_api_run
        def pipeline(base_dir):
            with api_attempt("openai", "chat_completions", "model", "req", 1):
                raise RuntimeError("sensitive text")

        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(RuntimeError):
                pipeline(directory)
            report = json.loads(next(Path(directory).glob("api_requests_*.json")).read_text())

        self.assertIsNotNone(report["finished_at"])
        self.assertEqual(report["requests"][0]["status"], "failed")
        self.assertEqual(report["requests"][0]["error_type"], "RuntimeError")
        self.assertNotIn("sensitive text", json.dumps(report))

    def test_openai_usage(self):
        @report_api_run
        def pipeline(base_dir):
            with api_attempt("openai", "chat_completions", "model", "req", 1) as attempt:
                attempt.set_response(
                    SimpleNamespace(
                        usage=SimpleNamespace(
                            prompt_tokens=30,
                            completion_tokens=15,
                            total_tokens=45,
                            prompt_tokens_details=SimpleNamespace(cached_tokens=10),
                            completion_tokens_details=SimpleNamespace(reasoning_tokens=4),
                        )
                    )
                )

        with tempfile.TemporaryDirectory() as directory:
            pipeline(directory)
            report = json.loads(next(Path(directory).glob("api_requests_*.json")).read_text())

        self.assertEqual(
            report["requests"][0]["tokens"],
            {"input": 30, "output": 15, "total": 45, "thinking": 4, "cached_input": 10},
        )


if __name__ == "__main__":
    unittest.main()
