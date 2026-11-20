"""The text, JSON and limit rules in `engram.validate`."""

from __future__ import annotations

import pytest

from engram.validate import check_json, check_limit, check_text


class TestCheckText:
    def test_returns_good_text_unchanged(self) -> None:
        text = "Alice \U0001f469\u200d\U0001f4bb keeps a bike \u05e9\u05dc\u05d5\u05dd \u4f60\u597d"
        assert check_text("text", text, max_bytes=100) == text

    @pytest.mark.parametrize("value", ["a\x00b", "\x00", "tail\x00"])
    def test_refuses_nul(self, value: str) -> None:
        with pytest.raises(ValueError, match="NUL"):
            check_text("text", value, max_bytes=100)

    @pytest.mark.parametrize("value", ["\ud800", "ok\udfffok"])
    def test_refuses_a_lone_surrogate(self, value: str) -> None:
        with pytest.raises(ValueError, match="surrogate"):
            check_text("text", value, max_bytes=100)

    def test_refuses_a_non_string(self) -> None:
        with pytest.raises(ValueError, match="string"):
            check_text("text", b"bytes", max_bytes=100)  # type: ignore[arg-type]

    def test_the_limit_counts_utf8_bytes(self) -> None:
        # Four characters, eight bytes.
        assert check_text("text", "\u00e9" * 4, max_bytes=8)
        with pytest.raises(ValueError, match="limit is 7"):
            check_text("text", "\u00e9" * 4, max_bytes=7)

    @pytest.mark.parametrize("value", ["", "   ", "\n\t"])
    def test_blank_is_refused_unless_allowed(self, value: str) -> None:
        with pytest.raises(ValueError, match="blank"):
            check_text("text", value, max_bytes=10)
        assert check_text("text", value, max_bytes=10, allow_blank=True) == value

    def test_allow_blank_still_refuses_nul(self) -> None:
        with pytest.raises(ValueError, match="NUL"):
            check_text("text", "\x00", max_bytes=10, allow_blank=True)

    def test_the_message_names_the_field_and_not_the_value(self) -> None:
        secret = "hunter2-not-for-logs\x00"
        with pytest.raises(ValueError) as info:
            check_text("reason", secret, max_bytes=100)
        assert "reason" in str(info.value)
        assert "hunter2" not in str(info.value)


class TestCheckJson:
    def test_accepts_plain_json(self) -> None:
        check_json("metadata", {"a": 1, "b": [1.5, "x", None, True], "c": {"d": "\u4f60"}})

    def test_accepts_an_empty_mapping(self) -> None:
        check_json("metadata", {})

    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
    def test_refuses_nan_and_infinity_at_any_depth(self, bad: float) -> None:
        with pytest.raises(ValueError, match="metadata"):
            check_json("metadata", {"a": bad})
        with pytest.raises(ValueError, match="metadata"):
            check_json("metadata", {"a": {"b": [1, bad]}})

    @pytest.mark.parametrize("bad", ["x\x00y", "\ud800"])
    def test_refuses_nul_and_surrogates_in_keys_and_values(self, bad: str) -> None:
        with pytest.raises(ValueError, match="metadata"):
            check_json("metadata", {bad: 1})
        with pytest.raises(ValueError, match="metadata"):
            check_json("metadata", {"k": [{"deep": bad}]})

    def test_refuses_what_json_cannot_hold(self) -> None:
        with pytest.raises(ValueError, match="metadata"):
            check_json("metadata", {"k": object()})

    def test_refuses_over_the_size_limit(self) -> None:
        check_json("metadata", {"k": "x" * 10}, max_bytes=20)
        with pytest.raises(ValueError, match="limit"):
            check_json("metadata", {"k": "x" * 30}, max_bytes=20)

    def test_refuses_nesting_too_deep_for_json(self) -> None:
        deep: dict[str, object] = {}
        node = deep
        for _ in range(5000):
            child: dict[str, object] = {}
            node["k"] = child
            node = child
        with pytest.raises(ValueError, match="deeply"):
            check_json("metadata", deep)


class TestCheckLimit:
    @pytest.mark.parametrize("value", [0, 1, 1000])
    def test_accepts_zero_through_the_maximum(self, value: int) -> None:
        assert check_limit("top_k", value, maximum=1000) == value

    @pytest.mark.parametrize("value", [-1, 1001])
    def test_refuses_negative_and_over_the_maximum(self, value: int) -> None:
        with pytest.raises(ValueError, match="top_k"):
            check_limit("top_k", value, maximum=1000)

    @pytest.mark.parametrize("value", [True, False, 1.0, "3", None])
    def test_refuses_a_bool_and_non_integers(self, value: object) -> None:
        with pytest.raises(ValueError, match="integer"):
            check_limit("top_k", value, maximum=1000)  # type: ignore[arg-type]
