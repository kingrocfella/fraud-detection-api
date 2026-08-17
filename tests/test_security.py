"""Access-control unit tests.

These import `app.security` directly rather than `app.main`, so they run in a
lightweight environment: importing the app pulls in the Dramatiq actors, which
pull in transformers/torch.
"""

import pytest
from fastapi import HTTPException

from app.security import (
    SECURITY_KEY_HEADER,
    verify_any_security_key,
    verify_security_key,
)


class TestVerifySecurityKey:
    def test_accepts_the_configured_key(self):
        verify_security_key("s3cret-key-value", "s3cret-key-value")

    def test_rejects_a_wrong_key_with_401(self):
        with pytest.raises(HTTPException) as excinfo:
            verify_security_key("wrong", "s3cret-key-value")
        assert excinfo.value.status_code == 401

    def test_rejects_an_empty_key_with_401(self):
        with pytest.raises(HTTPException) as excinfo:
            verify_security_key("", "s3cret-key-value")
        assert excinfo.value.status_code == 401

    def test_fails_closed_when_nothing_is_configured(self):
        """An unset configured key must never match an empty request key."""
        with pytest.raises(HTTPException) as excinfo:
            verify_security_key("", "")
        assert excinfo.value.status_code == 503


class TestVerifyAnySecurityKey:
    def test_accepts_the_first_configured_key(self):
        verify_any_security_key("detect-key", ("detect-key", "finetune-key"))

    def test_accepts_the_second_configured_key(self):
        verify_any_security_key("finetune-key", ("detect-key", "finetune-key"))

    def test_rejects_a_key_that_matches_neither(self):
        with pytest.raises(HTTPException) as excinfo:
            verify_any_security_key("nope", ("detect-key", "finetune-key"))
        assert excinfo.value.status_code == 401

    def test_ignores_blank_configured_entries(self):
        verify_any_security_key("detect-key", ("detect-key", ""))

    def test_fails_closed_when_every_configured_key_is_blank(self):
        with pytest.raises(HTTPException) as excinfo:
            verify_any_security_key("", ("", ""))
        assert excinfo.value.status_code == 503

    def test_rejects_an_empty_key_against_configured_keys(self):
        with pytest.raises(HTTPException) as excinfo:
            verify_any_security_key("", ("detect-key", "finetune-key"))
        assert excinfo.value.status_code == 401


def test_key_is_carried_in_a_header_not_a_query_parameter():
    """Regression guard: a key in a URL is copied into every log on the path."""
    assert SECURITY_KEY_HEADER == "X-API-Key"
