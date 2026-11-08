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

FAUCET_INSTRUCTIONS = (
    "Is a person visibly opening or actively using a running water faucet or tap "
    "in this image? Answer yes only when water is visibly flowing from the tap "
    "and a person is interacting with that same tap or its water, such as a hand "
    "at the tap control or hands under the running water. A person merely nearby "
    "and an unattended running tap do not qualify. Do not infer water flow from "
    "a hand on the control, handle position, a wet sink, standing water, steam, "
    "or a faucet alone. An open or boiling kettle or water boiler alone does not "
    "qualify. If there is no visible water flow or no clear interaction, answer no."
)


# A pantry camera sees a person who is either drinking or using the faucet, so one
# exclusive question replaces the two independent ones. The choice answer carries a
# probability per option and those probabilities sum to one, so drinking and faucet
# can never both pass POSITIVE_THRESHOLD for the same frame.
CHOICE_QUESTION = "activity"
CHOICE_OPTIONS = ("drinking", "faucet", "other")

CHOICE_INSTRUCTIONS = (
    "What is the person in this image most likely doing? Choose drinking when a "
    "cup, glass, bottle, can, or straw is at or very near the mouth, especially "
    "with a sipping posture or tilted head; do not require a visible liquid or a "
    "readable label. Choose faucet only when water is visibly flowing from a tap "
    "and the person interacts with that same tap or its water, such as a hand at "
    "the control or hands under the running water; a person merely nearby, an "
    "unattended running tap, a wet sink, steam, standing water, or a kettle or "
    "water boiler alone does not qualify, and do not infer water flow from a hand "
    "on the control or the handle position. Choose other for anything else, "
    "including eating, smoking, carrying a drink away from the mouth, or standing "
    "or walking without either cue."
)

CHOICE_CRITERIA = {
    "drinking": (
        "A beverage container at or very near the mouth, or a sipping posture, "
        "tilted head, or straw in use, with or without a visible liquid."
    ),
    "faucet": (
        "Visible water flowing from a tap while the person touches the control or "
        "holds their hands under that flow."
    ),
    "other": (
        "No plausible drinking cue and no visible running-water interaction: "
        "walking, standing, eating, smoking, working, a dry or unattended tap, or "
        "a kettle or water boiler on its own."
    ),
}

QUESTION_MODE_ENV = "QUESTION_MODE"
DEFAULT_QUESTION_MODE = "choice"


def get_question_mode() -> str:
    mode = os.environ.get(QUESTION_MODE_ENV, DEFAULT_QUESTION_MODE).strip().lower()
    mode = mode or DEFAULT_QUESTION_MODE
    if mode not in ("binary", "choice"):
        raise AnalyzerError(
            f"Unsupported {QUESTION_MODE_ENV}: {mode}. Choose 'binary' or 'choice'."
        )
    return mode


@dataclass(frozen=True)
class DetectionScores:
    drinking: Optional[float]
    faucet: float


def build_questions(person_present: bool) -> dict:
    if not person_present:
        raise AnalyzerError("AI analysis requires a person detected by YOLO.")
    if get_question_mode() == "choice":
        return {
            CHOICE_QUESTION: {
                "type": "choice",
                "instructions": CHOICE_INSTRUCTIONS,
                "criteria": dict(CHOICE_CRITERIA),
            }
        }
    return {
        "faucet": {
            "type": "noul",
            "instructions": FAUCET_INSTRUCTIONS,
        },
        "drinking": {
            "type": "noul",
            "instructions": DRINKING_INSTRUCTIONS,
        },
    }


def validate_probability(value: object, label: str, provider: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not 0 <= value <= 1
    ):
        raise AnalyzerError(f"{provider} returned an invalid {label} probability.")
    return float(value)


def parse_choice_answer(answer: object, provider: str) -> tuple[float, float]:
    if not isinstance(answer, dict) or answer.get("type") != "choice":
        raise AnalyzerError(f"{provider} response has no choice answer.")

    probabilities = answer.get("probabilities")
    if isinstance(probabilities, dict) and probabilities:
        scores = []
        for name in ("drinking", "faucet"):
            if name not in probabilities:
                scores.append(0.0)
                continue
            scores.append(
                validate_probability(probabilities[name], f"{name} choice", provider)
            )
        return scores[0], scores[1]

    label = answer.get("choice")
    if label not in CHOICE_OPTIONS:
        raise AnalyzerError(f"{provider} returned an unknown choice: {label!r}.")
    return (1.0 if label == "drinking" else 0.0, 1.0 if label == "faucet" else 0.0)


def parse_scores(answers: object, person_present: bool, provider: str) -> DetectionScores:
    if not isinstance(answers, dict):
        raise AnalyzerError(f"{provider} response has no answers object.")

    if CHOICE_QUESTION in answers:
        drinking, faucet = parse_choice_answer(answers[CHOICE_QUESTION], provider)
        return DetectionScores(
            drinking=drinking if person_present else None,
            faucet=faucet,
        )

    def probability(name: str) -> float:
        answer = answers.get(name)
        if not isinstance(answer, dict) or answer.get("type") != "noul":
            raise AnalyzerError(f"{provider} response has no {name} probability.")
        return validate_probability(answer.get("noul"), name, provider)

    return DetectionScores(
        drinking=probability("drinking") if person_present else None,
        faucet=probability("faucet"),
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
