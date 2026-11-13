import base64
import json
import os
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

from ai_analyzer import AnalyzerError, DetectionScores, build_questions, parse_scores


class CloudflareClefAnalyzer:
    def __init__(self) -> None:
        self.account_id = os.environ.get("CLOUDFLARE_ACCOUNT_ID", "").strip()
        self.api_token = os.environ.get("CLOUDFLARE_API_TOKEN", "").strip()
        self.model = os.environ.get("MODEL", "clef").strip()

        if not self.account_id or self.account_id.startswith("REPLACE_"):
            raise AnalyzerError("Set CLOUDFLARE_ACCOUNT_ID in .env.")
        if not self.api_token or self.api_token.startswith("REPLACE_"):
            raise AnalyzerError("Set CLOUDFLARE_API_TOKEN in .env.")
        if self.model not in ("clef", "clef-flash"):
            raise AnalyzerError(
                "For AI_PROVIDER=cloudflare, MODEL must be 'clef' or 'clef-flash'."
            )

    def analyze_frame(self, image: bytes, person_present: bool) -> DetectionScores:
        payload = {
            "model": self.model,
            "state": (
                "Inspect this camera image for the situations in the questions."
            ),
            "images": [
                {
                    "content_type": "image/jpeg",
                    "base64": base64.b64encode(image).decode("ascii"),
                }
            ],
            "questions": build_questions(person_present),
        }
        endpoint = (
            "https://api.cloudflare.com/client/v4/accounts/"
            f"{quote(self.account_id, safe='')}/ai/run/@cf/cloudflare/{self.model}"
        )
        request = Request(
            endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_token}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        try:
            with urlopen(request, timeout=30) as response:
                response_body = response.read()
        except HTTPError as error:
            details = error.read().decode("utf-8", errors="replace").strip()
            details = details.replace(self.api_token, "[redacted]")
            suffix = f": {details}" if details else ""
            raise AnalyzerError(
                f"Cloudflare API returned HTTP {error.code}{suffix}"
            ) from None
        except (TimeoutError, URLError) as error:
            raise AnalyzerError(f"Cloudflare API request failed: {error}") from None

        try:
            result = json.loads(response_body)
        except json.JSONDecodeError as error:
            raise AnalyzerError("Cloudflare API returned invalid JSON.") from error

        if not isinstance(result, dict):
            raise AnalyzerError("Cloudflare API returned an unexpected response.")
        if result.get("success") is False:
            errors = result.get("errors", [])
            if not isinstance(errors, list):
                errors = []
            messages = [
                item.get("message", "")
                for item in errors
                if isinstance(item, dict) and item.get("message")
            ]
            detail = "; ".join(messages) or "unspecified API error"
            detail = detail.replace(self.api_token, "[redacted]")
            raise AnalyzerError(f"Cloudflare API error: {detail}")

        result = result.get("result", result)
        if not isinstance(result, dict):
            raise AnalyzerError("Cloudflare API result is missing.")
        return parse_scores(result.get("answers"), person_present, "Cloudflare")
