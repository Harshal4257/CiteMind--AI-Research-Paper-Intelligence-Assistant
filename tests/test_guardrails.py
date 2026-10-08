import pytest

from src.guardrails import validate_query
from src.utils import QueryValidationError


def test_guardrails_normalize_valid_question() -> None:
    assert validate_query("  Explain self-attention.  ", 100) == "Explain self-attention."


@pytest.mark.parametrize("query", ["", "   ", None])
def test_guardrails_reject_empty_question(query) -> None:
    with pytest.raises(QueryValidationError, match="cannot be empty"):
        validate_query(query, 100)


def test_guardrails_reject_overlong_question() -> None:
    with pytest.raises(QueryValidationError, match="exceeds"):
        validate_query("x" * 101, 100)

