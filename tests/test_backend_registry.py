from duodose.models.registry import DEFAULT_DUODOSE_BACKEND, DUODOSE_BACKENDS, PUBLIC_METHOD_NAMES


def test_registry_is_frozen() -> None:
    assert DEFAULT_DUODOSE_BACKEND == "rf"
    assert list(DUODOSE_BACKENDS) == ["rf", "dl"]
    assert DUODOSE_BACKENDS["rf"] == "DuoDose-ML-CalibratedRF-SafeFeatures"
    assert DUODOSE_BACKENDS["dl"] == "DuoDose-DL-ConditionalMultiTaskMLP-SafeFeatures"
    assert PUBLIC_METHOD_NAMES[DUODOSE_BACKENDS["rf"]] == "DuoDose"
    assert PUBLIC_METHOD_NAMES[DUODOSE_BACKENDS["dl"]] == "DuoDose-DL"
    assert list(PUBLIC_METHOD_NAMES.values()).count("DuoDose") == 1
