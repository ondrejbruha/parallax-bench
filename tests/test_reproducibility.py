"""Reproducibility machinery: normalization, content addressing, redaction."""

import hashlib

import pytest

from parallax_bench.config import redacted_config
from parallax_bench.data import store_path
from parallax_bench.fetch import html_to_text, normalize_text, sha256_str


def test_normalize_is_idempotent():
    messy = "a\r\nb  \n\n\n\n\nč"  # CRLF, trailing spaces, 4 blank lines, combining háček
    once = normalize_text(messy)
    assert normalize_text(once) == once


def test_normalize_rules():
    out = normalize_text("line one   \r\nline two\n\n\n\n\nline three")
    assert "\r" not in out
    assert "line one\n" in out                # trailing whitespace stripped
    assert "\n\n\n\n" not in out              # >=3 blank lines collapsed to 2
    assert out.endswith("\n") and not out.endswith("\n\n")


def test_normalize_nfc():
    import unicodedata

    precomposed = unicodedata.normalize("NFC", "na\u0159\u00edzen\u00ed")
    decomposed = unicodedata.normalize("NFD", precomposed)
    assert precomposed != decomposed  # genuinely different byte sequences
    assert sha256_str(normalize_text(decomposed)) == sha256_str(normalize_text(precomposed))


def test_html_to_text_is_normalized():
    text = html_to_text("<html><p>Hello  </p><p>World</p></html>")
    assert normalize_text(text) == text


def test_store_path_shape(tmp_path):
    digest = hashlib.sha256(b"x").hexdigest()
    p = store_path(tmp_path, digest)
    assert p == tmp_path / "by-hash" / digest[:2] / digest[2:4] / digest


def test_redacted_config_masks_but_keeps_keys(monkeypatch):
    monkeypatch.setenv("SUT_URL", "https://prod.example.internal")
    monkeypatch.setenv("SUT_JWT", "supersecret")
    cfg = redacted_config(
        {
            "base_url": "${SUT_URL}",
            "jwt_secret": "${SUT_JWT}",
            "prefetch_k": 50,
            "nested": {"api_key": "literal-key", "result_count": 10},
        }
    )
    assert cfg["base_url"] == "<sut-endpoint>"
    assert cfg["jwt_secret"] == "<redacted>"          # present, visibly masked
    assert cfg["nested"]["api_key"] == "<redacted>"
    assert cfg["prefetch_k"] == 50                    # everything else verbatim
    assert cfg["nested"]["result_count"] == 10
    assert "supersecret" not in str(cfg)


def test_redacted_config_fails_on_unset_env():
    with pytest.raises(KeyError):
        redacted_config({"base_url": "${DEFINITELY_UNSET_VAR_42}"})
