def test_package_import() -> None:
    import duodose

    assert duodose.__version__ == "0.1.0"
    assert callable(duodose.DuoDose)
    assert callable(duodose.detect_doublets)
