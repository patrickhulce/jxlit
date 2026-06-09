from jxlit import decode


def test_decode_returns_empty_buffer() -> None:
    assert decode(b"not-a-jxl") == b""


def test_decode_rejects_non_bytes_like() -> None:
    try:
        decode("not-a-jxl")  # type: ignore[arg-type]
    except TypeError as exc:
        assert "bytes-like" in str(exc)
    else:
        raise AssertionError("expected TypeError")
