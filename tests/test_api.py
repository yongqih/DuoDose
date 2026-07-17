from duodose import DuoDose


def test_api_defaults_to_rf() -> None:
    detector = DuoDose(training_preset="fast")
    assert detector.config.backend == "rf"
    assert detector.get_params()["expected_doublet_rate"] == 0.08


def test_set_params_invalidates_fit() -> None:
    detector = DuoDose(backend="rf")
    detector.backend_ = object()
    detector.set_params(expected_doublet_rate=0.1)
    assert detector.backend_ is None
    assert detector.config.expected_doublet_rate == 0.1
