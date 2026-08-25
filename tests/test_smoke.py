import esmoe


def test_public_api():
    assert hasattr(esmoe, "ESMoE")
    assert callable(esmoe.inject_esmoe)
    assert esmoe.__version__
