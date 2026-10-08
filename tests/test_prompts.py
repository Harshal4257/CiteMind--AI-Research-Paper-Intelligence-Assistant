from src.prompts import answer_depth_instruction


def test_answer_depth_instruction_has_a_safe_standard_fallback() -> None:
    assert answer_depth_instruction("Unknown") == answer_depth_instruction("Standard")


def test_answer_depth_instruction_changes_the_response_guidance() -> None:
    assert answer_depth_instruction("Concise") != answer_depth_instruction("Detailed")
    assert "### Short answer" in answer_depth_instruction("Standard")
