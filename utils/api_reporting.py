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

import contextvars
import functools
import inspect
import json
import os
import sys
import threading
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone


_report = contextvars.ContextVar("api_report", default=None)
_stage = contextvars.ContextVar("api_stage", default="unclassified")


def _now():
    return datetime.now(timezone.utc).isoformat()


def _component():
    frame = inspect.currentframe()
    try:
        while frame:
            module = frame.f_globals.get("__name__", "")
            if module.startswith("methods.agents.") or module in (
                "utils.paper_banana_utils",
                "utils.scholar_utils",
            ):
                return f"{module}.{frame.f_code.co_name}"
            frame = frame.f_back
    finally:
        del frame
    return None


def _usage(response, provider):
    if provider == "gemini":
        metadata = getattr(response, "usage_metadata", None)
        fields = {
            "input": "prompt_token_count",
            "output": "candidates_token_count",
            "thinking": "thoughts_token_count",
            "cached_input": "cached_content_token_count",
            "tool_input": "tool_use_prompt_token_count",
            "total": "total_token_count",
        }
    elif provider == "openai":
        metadata = getattr(response, "usage", None)
        fields = {
            "input": "prompt_tokens",
            "output": "completion_tokens",
            "total": "total_tokens",
        }
    else:
        return None
    if metadata is None:
        return None
    values = {
        key: value
        for key, name in fields.items()
        if isinstance(value := getattr(metadata, name, None), int)
    }
    if provider == "openai":
        completion_details = getattr(metadata, "completion_tokens_details", None)
        prompt_details = getattr(metadata, "prompt_tokens_details", None)
        thinking = getattr(completion_details, "reasoning_tokens", None)
        cached_input = getattr(prompt_details, "cached_tokens", None)
        if isinstance(thinking, int):
            values["thinking"] = thinking
        if isinstance(cached_input, int):
            values["cached_input"] = cached_input
    return values or None


class ApiReport:
    def __init__(self, base_dir):
        os.makedirs(base_dir, exist_ok=True)
        self.path = os.path.join(
            base_dir,
            f"api_requests_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}_{uuid.uuid4().hex[:8]}.json",
        )
        self.lock = threading.Lock()
        self.data = {
            "started_at": _now(),
            "finished_at": None,
            "notes": [
                "Attempts count application SDK/HTTP calls; internal SDK retries are not visible.",
                "Succeeded means the API returned successfully, not that response parsing succeeded.",
                "Retries share a request_id; separate calls are not compared by prompt or payload.",
                "Token usage is recorded only when provided by the API; missing usage is not zero.",
                "Cost is unavailable: pricing, billing tier, modalities and tool "
                "charges are not recorded.",
                "Google Search grounding is included in Gemini calls; internal "
                "search requests are not counted.",
                "Prompts, request bodies, responses, URLs with query parameters "
                "and credentials are not stored.",
            ],
            "estimated_cost_usd": None,
            "stage_summary": [],
            "summary": [],
            "requests": [],
        }
        self._write()
        print(f"API request report: {self.path}", flush=True)

    def _write(self):
        groups = {}
        stages = {}
        for request in self.data["requests"]:
            key = (
                request["stage"],
                request["provider"],
                request["operation"],
                request["model"],
                request["component"],
            )
            group = groups.setdefault(
                key,
                {
                    "stage": key[0],
                    "provider": key[1],
                    "operation": key[2],
                    "model": key[3],
                    "component": key[4],
                    "attempts": 0,
                    "logical_requests": set(),
                    "retries": 0,
                    "succeeded": 0,
                    "failed": 0,
                    "pending": 0,
                    "usage_reported": 0,
                    "tokens": {},
                },
            )
            stage = stages.setdefault(
                request["stage"],
                {
                    "stage": request["stage"],
                    "attempts": 0,
                    "logical_requests": set(),
                    "retries": 0,
                    "succeeded": 0,
                    "failed": 0,
                    "pending": 0,
                    "usage_reported": 0,
                    "tokens": {},
                },
            )
            for item in (group, stage):
                item["attempts"] += 1
                item["logical_requests"].add(request["request_id"])
                item["retries"] += int(request["attempt"] > 1)
                item[request["status"]] += 1
                if request["tokens"] is not None:
                    item["usage_reported"] += 1
                    for name, value in request["tokens"].items():
                        item["tokens"][name] = item["tokens"].get(name, 0) + value
        for key, items in (("summary", groups), ("stage_summary", stages)):
            self.data[key] = []
            for item in items.values():
                item["logical_requests"] = len(item["logical_requests"])
                item["usage_unavailable"] = item["attempts"] - item["usage_reported"]
                if not item["tokens"]:
                    item["tokens"] = None
                self.data[key].append(item)
        temp_path = self.path + ".tmp"
        with open(temp_path, "w", encoding="utf-8") as file:
            json.dump(self.data, file, indent=2, ensure_ascii=False)
        os.replace(temp_path, self.path)

    def persist(self):
        try:
            self._write()
        except OSError as error:
            print(
                f"ERROR: Cannot write API request report {self.path}: {error}",
                file=sys.stderr,
                flush=True,
            )


def report_api_run(function):
    @functools.wraps(function)
    def wrapper(base_dir, *args, **kwargs):
        report = ApiReport(base_dir)
        token = _report.set(report)
        try:
            return function(base_dir, *args, **kwargs)
        finally:
            with report.lock:
                report.data["finished_at"] = _now()
                report.persist()
            _report.reset(token)

    return wrapper


def api_stage(name):
    def decorator(function):
        @functools.wraps(function)
        def wrapper(*args, **kwargs):
            token = _stage.set(name)
            try:
                return function(*args, **kwargs)
            finally:
                _stage.reset(token)

        return wrapper

    return decorator


def submit_with_context(executor, function, *args):
    context = contextvars.copy_context()
    return executor.submit(context.run, function, *args)


class _Attempt:
    def __init__(self):
        self.response = None

    def set_response(self, response):
        self.response = response


@contextmanager
def api_attempt(provider, operation, model, request_id, attempt):
    report = _report.get()
    tracked = _Attempt()
    if report is None:
        yield tracked
        return
    entry = {
        "request_id": request_id,
        "attempt": attempt,
        "stage": _stage.get(),
        "provider": provider,
        "operation": operation,
        "model": model,
        "component": _component(),
        "started_at": _now(),
        "duration_seconds": None,
        "status": "pending",
        "http_status": None,
        "error_type": None,
        "tokens": None,
    }
    with report.lock:
        report.data["requests"].append(entry)
        report.persist()
    start = time.monotonic()
    status = "succeeded"
    error_type = None
    status_code = None
    tokens = None
    try:
        yield tracked
    except BaseException as error:
        status = "failed"
        error_type = type(error).__name__
        error_code = getattr(error, "code", None)
        if isinstance(error_code, int):
            status_code = error_code
        raise
    else:
        response_status = getattr(tracked.response, "status_code", None)
        status_code = response_status if isinstance(response_status, int) else None
        if status_code is not None and status_code >= 400:
            status = "failed"
        tokens = _usage(tracked.response, provider)
    finally:
        with report.lock:
            entry.update(
                status=status,
                error_type=error_type,
                http_status=status_code,
                tokens=tokens,
                duration_seconds=round(time.monotonic() - start, 3),
            )
            report.persist()
