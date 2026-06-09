from jxlit import decode


def test_decode_returns_empty_buffer() -> None:
    assert decode(b"not-a-jxl") == b""
