import os
from dataclasses import dataclass
from typing import Optional, Protocol


class AnalyzerError(Exception):
    pass


DRINKING_INSTRUCTIONS = (
    "Is any person visibly drinking or likely taking a sip of any beverage in this "
    "image? Include water and all other drinks, hot or cold, such as tea, coffee, "
    "juice, and soda. Treat a cup, glass, bottle, can, or straw at or very near the "
    "mouth, especially with a sipping posture or tilted head, as possible drinking "
    "even when the exact liquid or moment of sipping is unclear. Do not require a "
    "visible liquid or readable label. Do not count a person merely carrying or "
    "holding a drink away from the mouth, eating, or smoking. Answer no when there "
    "are no plausible visual cues of sipping."
)

BOILWATER_INSTRUCTIONS = (
    "Is a person visibly opening or actively using a running water faucet or tap "
    "in this image? Answer yes only when water is visibly flowing from the tap "
    "and a person is interacting with that same tap or its water, such as a hand "
    "at the tap control or hands under the running water. A person merely nearby "
    "and an unattended running tap do not qualify. Do not infer water flow from "
    "a hand on the control, handle position, a wet sink, standing water, steam, "
    "or a faucet alone. An open or boiling kettle or water boiler alone does not "
    "qualify. If there is no visible water flow or no clear interaction, answer no."
)


@dataclass(frozen=True)
class DetectionScores:
    drinking: Optional[float]
    boilwater: float


def build_questions(person_present: bool) -> dict:
    if not person_present:
        raise AnalyzerError("AI analysis requires a person detected by YOLO.")
    return {
        "boilwater": {
            "type": "noul",
            "instructions": BOILWATER_INSTRUCTIONS,
        },
        "drinking": {
            "type": "noul",
            "instructions": DRINKING_INSTRUCTIONS,
        },
    }


def parse_scores(answers: object, person_present: bool, provider: str) -> DetectionScores:
    if not isinstance(answers, dict):
        raise AnalyzerError(f"{provider} response has no answers object.")

    def probability(name: str) -> float:
        answer = answers.get(name)
        if not isinstance(answer, dict) or answer.get("type") != "noul":
            raise AnalyzerError(f"{provider} response has no {name} probability.")
        score = answer.get("noul")
        if (
            isinstance(score, bool)
            or not isinstance(score, (int, float))
            or not 0 <= score <= 1
        ):
            raise AnalyzerError(f"{provider} returned an invalid {name} probability.")
        return float(score)

    return DetectionScores(
        drinking=probability("drinking") if person_present else None,
        boilwater=probability("boilwater"),
    )


class FrameAnalyzer(Protocol):
    def analyze_frame(self, image: bytes, person_present: bool) -> DetectionScores:
        ...


def create_analyzer() -> FrameAnalyzer:
    provider = os.environ.get("AI_PROVIDER", "cloudflare").strip().lower()
    if provider in ("cloudflare", "clef"):
        from cloudflare_clef import CloudflareClefAnalyzer

        return CloudflareClefAnalyzer()
    if provider == "openrouter":
        from openrouter_decision import OpenRouterDecisionAnalyzer

        return OpenRouterDecisionAnalyzer()

    raise AnalyzerError(
        f"Unsupported AI_PROVIDER: {provider or '(empty)'}. "
        "Choose 'cloudflare' or 'openrouter'."
    )
