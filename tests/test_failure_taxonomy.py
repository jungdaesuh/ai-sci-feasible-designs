from ai_scientist.failure_taxonomy import classify_failure


def test_classify_failure_ok() -> None:
    classification = classify_failure(None)
    assert classification.failure_label == "ok"
    assert classification.failure_source == "none"
    assert classification.failure_signature is None


def test_classify_failure_vmec_jacobian_bad() -> None:
    classification = classify_failure("VMEC failed: JACOBIAN IS BAD, 75 TIMES BAD")
    assert classification.failure_label == "vmec_jacobian_bad"
    assert classification.failure_source == "vmec"
    assert classification.failure_signature is not None


def test_classify_failure_signature_normalizes_whitespace() -> None:
    first = classify_failure("Physics evaluation failed:   timeout at solver")
    second = classify_failure("physics evaluation failed:\ntimeout at solver")
    assert first.failure_label == "evaluation_timeout"
    assert second.failure_label == "evaluation_timeout"
    assert first.failure_signature == second.failure_signature


def test_classify_failure_accepts_non_string_error_payload() -> None:
    classification = classify_failure(RuntimeError("JACOBIAN IS BAD, 75 TIMES BAD"))
    assert classification.failure_label == "vmec_jacobian_bad"
    assert classification.failure_source == "vmec"
    assert classification.failure_signature is not None


def test_classify_failure_qi_convergence_precedence() -> None:
    classification = classify_failure("Boozer transform failed to converge")
    assert classification.failure_label == "qi_computation_failure"
    assert classification.failure_source == "qi"
