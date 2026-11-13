import base64
import json
import math
import os
from http.client import HTTPException
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

from ai_analyzer import (
    AnalyzerError,
    DetectionScores,
    build_questions,
    parse_scores,
)


class NoRedirectHandler(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


urlopen = build_opener(NoRedirectHandler()).open


class LlamaCppDecisionAnalyzer:
    def __init__(self) -> None:
        self.url = os.environ.get("LLAMA_CPP_URL", "").strip()
        try:
            parts = urlsplit(self.url)
            valid_url = (
                parts.scheme in ("http", "https")
                and parts.hostname
                and "REPLACE_" not in self.url
                and not any(character.isspace() for character in self.url)
                and parts.port != 0
                and parts.username is None
                and parts.password is None
                and not parts.query
                and not parts.fragment
                and parts.path.endswith("/v1/systemone")
            )
        except ValueError:
            valid_url = False
        if not valid_url:
            raise AnalyzerError(
                "Set LLAMA_CPP_URL to an http(s) URL ending in /v1/systemone "
                "without credentials, query, or fragment."
            )

        self.api_key = os.environ.get("LLAMA_CPP_API_KEY", "").strip()
        if self.api_key.startswith("REPLACE_"):
            raise AnalyzerError("Set LLAMA_CPP_API_KEY or leave it empty.")
        if "\r" in self.api_key or "\n" in self.api_key:
            raise AnalyzerError("LLAMA_CPP_API_KEY must not contain line breaks.")

        timeout = os.environ.get("LLAMA_CPP_TIMEOUT_SECONDS", "30").strip()
        try:
            self.timeout = float(timeout)
        except ValueError:
            raise AnalyzerError("LLAMA_CPP_TIMEOUT_SECONDS must be a positive number.") from None
        if not math.isfinite(self.timeout) or self.timeout <= 0:
            raise AnalyzerError("LLAMA_CPP_TIMEOUT_SECONDS must be a positive number.")

    def analyze_frame(self, image: bytes, person_present: bool) -> DetectionScores:
        questions = build_questions(person_present)
        image_data = base64.b64encode(image).decode("ascii")
        payload = {
            "state": None,
            "images": [f"data:image/jpeg;base64,{image_data}"],
            "questions": questions,
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = Request(
            self.url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )

        try:
            with urlopen(request, timeout=self.timeout) as response:
                response_body = response.read()
        except HTTPError as error:
            raise AnalyzerError(
                f"llama.cpp Decision API returned HTTP {error.code}."
            ) from None
        except (URLError, OSError, HTTPException) as error:
            raise AnalyzerError(f"llama.cpp Decision API request failed: {error}") from None

        try:
            result = json.loads(response_body)
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            raise AnalyzerError("llama.cpp Decision API returned invalid JSON.") from error
        if not isinstance(result, dict):
            raise AnalyzerError("llama.cpp Decision API returned an unexpected response.")
        if result.get("error"):
            raise AnalyzerError("llama.cpp Decision API returned an error response.")

        answers = result.get("answers")
        if not isinstance(answers, dict) or set(answers) != set(questions):
            raise AnalyzerError("llama.cpp Decision API returned unexpected answer keys.")
        return parse_scores(answers, person_present, "llama.cpp")
