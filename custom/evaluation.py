"""Judge one Extracted Resume against a Role with upstream's evaluator."""

from typing import Optional

from pydantic import BaseModel

import evaluator
from custom.llm import LLMChoice, bound_to
from models import JSONResume, OpenAICompatibleProvider, build_evaluation_model
from roles import Role
from transform import convert_github_data_to_text, convert_json_resume_to_text


def evaluate(
    resume: JSONResume,
    github_data: Optional[dict],
    role: Role,
    choice: LLMChoice,
    client: OpenAICompatibleProvider,
) -> BaseModel:
    """Same input upstream's score.py builds: resume text plus any GitHub Enrichment."""
    text = convert_json_resume_to_text(resume)
    if github_data:
        text += convert_github_data_to_text(github_data)
    with bound_to(evaluator, choice, client):
        resume_evaluator = evaluator.ResumeEvaluator(
            role=role,
            evaluation_model=build_evaluation_model(role),
            model_name=choice.model,
            model_params=choice.params,
        )
        return resume_evaluator.evaluate_resume(text)
