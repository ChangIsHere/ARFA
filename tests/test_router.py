from src.phase3_arfa.run_arfa import phase3_status


def test_formal_phase3_remains_locked():
    status = phase3_status({"locked": True, "reason": "formal gate pending"})

    assert status["ready"] is False
    assert status["reason"] == "formal gate pending"


def test_exploratory_phase3_scaffold_can_be_unlocked():
    status = phase3_status(
        {
            "locked": False,
            "stage": "phase3_p_exploratory",
            "scope": "engineering_pilot",
            "formal_claims_enabled": False,
            "use_phase3_final_tasks": False,
            "router": {"fail_open_to_reasoning": True},
        }
    )

    assert status["ready"] is True
    assert status["formal_claims_enabled"] is False
    assert status["use_phase3_final_tasks"] is False
    assert status["router"]["fail_open_to_reasoning"] is True
