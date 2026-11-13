import base64
import json
import os
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from ai_analyzer import AnalyzerError, DetectionScores, build_questions, parse_scores


class OpenRouterDecisionAnalyzer:
    def __init__(self) -> None:
        self.api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
        self.model = os.environ.get(
            "MODEL", "liquid/d1.1"
        ).strip()
        if not self.api_key or self.api_key.startswith("REPLACE_"):
            raise AnalyzerError("Set OPENROUTER_API_KEY in .env.")
        if not self.model:
            raise AnalyzerError("MODEL cannot be empty.")

    def analyze_frame(self, image: bytes, person_present: bool) -> DetectionScores:
        image_data = base64.b64encode(image).decode("ascii")
        questions = build_questions(person_present)
        if person_present and "drinking" in questions:
            questions["drinking"]["criteria"] = {
                "true": "A person appears to be drinking or taking a sip.",
                "false": "No plausible visual cues of drinking are present.",
            }
        payload = {
            "model": self.model,
            "state": [
                (
                    "Inspect this camera image for the situations in the questions."
                ),
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image_data}",
                    },
                },
            ],
            "questions": questions,
        }
        request = Request(
            "https://openrouter.ai/api/alpha/decisions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        try:
            with urlopen(request, timeout=30) as response:
                response_body = response.read()
        except HTTPError as error:
            details = error.read().decode("utf-8", errors="replace").strip()
            details = details.replace(self.api_key, "[redacted]")
            suffix = f": {details}" if details else ""
            raise AnalyzerError(
                f"OpenRouter decisions API returned HTTP {error.code}{suffix}"
            ) from None
        except (TimeoutError, URLError) as error:
            raise AnalyzerError(f"OpenRouter request failed: {error}") from None

        try:
            result = json.loads(response_body)
        except json.JSONDecodeError as error:
            raise AnalyzerError("OpenRouter returned invalid JSON.") from error

        if not isinstance(result, dict):
            raise AnalyzerError("OpenRouter returned an unexpected response.")
        if result.get("error"):
            error = result["error"]
            detail = (
                error.get("message", "unspecified API error")
                if isinstance(error, dict)
                else str(error)
            )
            detail = detail.replace(self.api_key, "[redacted]")
            raise AnalyzerError(f"OpenRouter API error: {detail}")

        answers = result.get("answers")
        if answers is None and isinstance(result.get("result"), dict):
            answers = result["result"].get("answers")
        return parse_scores(answers, person_present, "OpenRouter")
