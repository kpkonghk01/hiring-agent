"""The ai_application_engineer Role's bonus is computed by Python from its bonus_rules."""

import pytest
from pydantic import ValidationError

from custom.ranking import overall_score
from models import build_bonus_model, build_evaluation_model
from roles import load_role

ROLE = load_role("ai_application_engineer")
KEYS = [rule["key"] for rule in ROLE.bonus_rules]


def bonus_items(**points):
    return {"items": {k: {"points": points.get(k, 0), "evidence": "e"} for k in KEYS}}


def make_evaluation(**bonus):
    return build_evaluation_model(ROLE).model_validate(
        {
            "scores": {
                c.key: {"score": c.max, "max": c.max, "evidence": "e"}
                for c in ROLE.categories
            },
            "bonus_points": bonus_items(**bonus),
            "deductions": {"total": 0, "reasons": "-"},
            "key_strengths": ["s"],
            "areas_for_improvement": ["a"],
        }
    )


def test_total_is_the_sum_of_the_items():
    bonus = build_bonus_model(ROLE).model_validate(
        bonus_items(adtech_ecommerce=3, ai_framework=3, internal_tools=2)
    )

    assert bonus.total == 8


def test_total_is_capped_at_bonus_max():
    bonus = build_bonus_model(ROLE).model_validate(
        bonus_items(
            cantonese=5, english=2, adtech_ecommerce=4, ai_framework=3, internal_tools=3
        )
    )

    assert sum(rule["points"][-1] for rule in ROLE.bonus_rules) > ROLE.bonus_max
    assert bonus.total == ROLE.bonus_max


@pytest.mark.parametrize("off_rule", [{"english": 3}, {"adtech_ecommerce": 5}])
def test_points_outside_a_rule_are_rejected(off_rule):
    with pytest.raises(ValidationError):
        build_bonus_model(ROLE).model_validate(bonus_items(**off_rule))


def test_a_model_supplied_total_is_rejected():
    with pytest.raises(ValidationError):
        build_bonus_model(ROLE).model_validate({"total": 9, "breakdown": "-"})


def test_overall_score_uses_the_computed_total():
    evaluation = make_evaluation(adtech_ecommerce=3, ai_framework=3, internal_tools=2)
    stored = evaluation.model_dump(round_trip=True)

    # 100 from the categories + 8 bonus, with no deductions
    assert overall_score(stored, ROLE) == 108
