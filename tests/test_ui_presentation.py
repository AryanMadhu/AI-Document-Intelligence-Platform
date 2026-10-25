"""Unit tests for ui.presentation — the pure, security-critical UI logic.

Fully offline: no Streamlit import, no network. Covers Markdown/HTML injection
defense, error-code mapping (including the regex-gated detail for the two limit
codes and the no-leak guarantees), response-shape validation, and page-badge
formatting.
"""

from __future__ import annotations

from ui import presentation as P
from ui import styles as S


# ---------------------------------------------------------------------------
# escape_markdown (Change D)
# ---------------------------------------------------------------------------
class TestEscapeMarkdown:
    def test_image_payload_is_neutralized(self):
        out = P.escape_markdown("![a](http://evil/x.png)")
        # The characters that would form an image are escaped, so no active
        # "![...](...)" image syntax remains intact.
        assert "![a](http://evil/x.png)" not in out
        assert "\\!" in out and "\\[" in out and "\\]" in out
        assert "\\(" in out and "\\)" in out
        # The URL text itself still appears (as literal text), just inert.
        assert "evil" in out

    def test_link_payload_is_neutralized(self):
        out = P.escape_markdown("[a](http://evil)")
        assert "[a](http://evil)" not in out
        assert "\\[" in out and "\\]" in out and "\\(" in out and "\\)" in out

    def test_raw_html_is_neutralized(self):
        out = P.escape_markdown("<img src=x onerror=alert(1)>")
        # Angle brackets are backslash-escaped, so Markdown renders them as
        # literal text and no real <img> tag can form. Every '<' and '>' must be
        # preceded by a backslash (i.e. never an UNescaped "<img").
        assert "\\<" in out and "\\>" in out
        assert "<img" not in out.replace("\\<", "")
        # The parentheses in alert(1) are also neutralized.
        assert "\\(" in out and "\\)" in out

    def test_benign_text_remains_readable(self):
        text = "The cat sat on the mat in 2024."
        out = P.escape_markdown(text)
        # No markdown-significant chars here, so the text is unchanged & readable.
        assert out == text

    def test_backslash_escaped_first(self):
        out = P.escape_markdown("a\\b")
        assert out == "a\\\\b"

    def test_non_string_is_coerced(self):
        assert P.escape_markdown(123) == "123"


# ---------------------------------------------------------------------------
# friendly_error — mapping for every code (Change C)
# ---------------------------------------------------------------------------
class TestFriendlyError:
    def test_every_backend_code_maps_to_generic(self):
        for code in [
            "EMPTY_QUESTION",
            "INVALID_PDF",
            "FILE_TOO_LARGE",
            "TOO_MANY_PAGES",
            "EMPTY_DOCUMENT",
            "REEMBEDDING_REQUIRED",
            "LLM_FAILED",
            "EMBEDDING_FAILED",
            "DOCUMENT_NOT_FOUND",
            "VALIDATION_ERROR",
            "INTERNAL_ERROR",
            "MISSING_CREDENTIAL",
            "CONNECTION_ERROR",
            "TIMEOUT",
            "UNEXPECTED_RESPONSE",
        ]:
            msg = P.friendly_error(code)
            assert isinstance(msg, str) and msg

    def test_file_too_large_detail_appended_when_regex_matches(self):
        msg = P.friendly_error(
            "FILE_TOO_LARGE", "File exceeds the maximum size of 5 MB."
        )
        assert "File exceeds the maximum size of 5 MB." in msg

    def test_too_many_pages_detail_appended_when_regex_matches(self):
        msg = P.friendly_error(
            "TOO_MANY_PAGES", "File exceeds the maximum of 50 pages."
        )
        assert "File exceeds the maximum of 50 pages." in msg

    def test_file_too_large_detail_with_html_not_appended(self):
        evil = "File exceeds the maximum size of 5 MB. <script>"
        msg = P.friendly_error("FILE_TOO_LARGE", evil)
        # The non-matching (tampered) detail must NOT be appended at all.
        assert "<script>" not in msg
        assert evil not in msg
        assert msg == P._GENERIC_MESSAGES["FILE_TOO_LARGE"]

    def test_detail_with_path_or_secret_never_surfaced(self):
        for evil in [
            "C:\\secrets\\key.pem",
            "GEMINI_API_KEY=abc123",
            'File exceeds the maximum size of 5 MB.. extra',
            "File exceeds the maximum of 50 pages. and then some",
        ]:
            msg_large = P.friendly_error("FILE_TOO_LARGE", evil)
            msg_pages = P.friendly_error("TOO_MANY_PAGES", evil)
            assert evil not in msg_large
            assert evil not in msg_pages

    def test_other_codes_never_append_detail(self):
        # Even if a detail is provided for a non-whitelisted code, it is dropped.
        msg = P.friendly_error("INVALID_PDF", "some/internal/path.pdf")
        assert "some/internal/path.pdf" not in msg

    def test_reembedding_required_has_no_ids(self):
        msg = P.friendly_error(
            "REEMBEDDING_REQUIRED",
            "Re-embedding is required for: 1234-id, 5678-id",
        )
        assert "1234-id" not in msg
        assert "5678-id" not in msg
        assert "administrator" in msg.lower()

    def test_unknown_code_is_generic(self):
        msg = P.friendly_error("SOME_NEW_CODE", "leaky detail here")
        assert "leaky detail here" not in msg
        assert msg == P._UNKNOWN_CODE_MESSAGE

    def test_non_string_code_is_generic(self):
        assert P.friendly_error(None) == P._UNKNOWN_CODE_MESSAGE
        assert P.friendly_error(500) == P._UNKNOWN_CODE_MESSAGE

    def test_service_busy_returns_busy_message(self):
        msg = P.friendly_error("SERVICE_BUSY")
        assert msg == (
            "The assistant is busy right now. Please wait a moment and try again."
        )

    def test_quota_exceeded_returns_quota_message(self):
        msg = P.friendly_error("QUOTA_EXCEEDED")
        assert msg == (
            "The usage limit for the assistant has been reached. Please check "
            "your quota and try again after it resets. Daily limits reset once "
            "per day; per-minute limits reset within a minute."
        )

    def test_llm_auth_failed_returns_config_message(self):
        msg = P.friendly_error("LLM_AUTH_FAILED")
        assert msg == (
            "The assistant isn't configured correctly. Please contact the "
            "administrator."
        )


# ---------------------------------------------------------------------------
# error_severity — warning (recoverable) vs error (hard failure)
# ---------------------------------------------------------------------------
class TestErrorSeverity:
    def test_recoverable_codes_are_warning(self):
        for code in [
            "SERVICE_BUSY",
            "TIMEOUT",
            "CONNECTION_ERROR",
            "REEMBEDDING_REQUIRED",
            # Quota is recoverable after reset -> amber/warning.
            "QUOTA_EXCEEDED",
        ]:
            assert P.error_severity(code) == "warning"

    def test_hard_failures_are_error(self):
        # LLM_AUTH_FAILED is a hard config error -> red/error (NOT recoverable).
        for code in [
            "LLM_FAILED",
            "INTERNAL_ERROR",
            "INVALID_PDF",
            "LLM_AUTH_FAILED",
        ]:
            assert P.error_severity(code) == "error"

    def test_unknown_and_non_string_are_error(self):
        assert P.error_severity("SOME_NEW_CODE") == "error"
        assert P.error_severity(None) == "error"
        assert P.error_severity(500) == "error"


# ---------------------------------------------------------------------------
# Response-shape validators (Change F)
# ---------------------------------------------------------------------------
class TestValidateQueryResponse:
    def test_valid_with_citations(self):
        body = {
            "answer": "hello",
            "citations": [
                {"document": "a.pdf", "page": 3, "excerpt": "x"},
                {"document": "b.pdf", "page": "Intro", "excerpt": "y"},
            ],
        }
        assert P.validate_query_response(body) is True

    def test_valid_empty_citations(self):
        assert P.validate_query_response({"answer": "a", "citations": []}) is True

    def test_missing_answer_rejected(self):
        assert P.validate_query_response({"citations": []}) is False

    def test_wrong_answer_type_rejected(self):
        assert P.validate_query_response({"answer": 1, "citations": []}) is False

    def test_citations_not_list_rejected(self):
        assert P.validate_query_response({"answer": "a", "citations": {}}) is False

    def test_bad_citation_rejected(self):
        body = {"answer": "a", "citations": [{"document": "a.pdf", "page": 1}]}
        assert P.validate_query_response(body) is False

    def test_bool_page_rejected(self):
        body = {
            "answer": "a",
            "citations": [{"document": "a.pdf", "page": True, "excerpt": "x"}],
        }
        assert P.validate_query_response(body) is False

    def test_non_dict_rejected(self):
        assert P.validate_query_response("nope") is False


class TestValidateUploadResponse:
    def test_valid(self):
        body = {
            "document_id": "id",
            "document_name": "a.pdf",
            "status": "ingested",
            "page_count": 3,
            "chunk_count": 10,
            "message": "ok",
        }
        assert P.validate_upload_response(body) is True

    def test_missing_field_rejected(self):
        body = {
            "document_id": "id",
            "document_name": "a.pdf",
            "status": "ingested",
            "page_count": 3,
            "message": "ok",
        }
        assert P.validate_upload_response(body) is False

    def test_bool_count_rejected(self):
        body = {
            "document_id": "id",
            "document_name": "a.pdf",
            "status": "ingested",
            "page_count": True,
            "chunk_count": 10,
            "message": "ok",
        }
        assert P.validate_upload_response(body) is False


class TestValidateDocumentsResponse:
    def test_valid_populated(self):
        body = {
            "documents": [
                {
                    "document_name": "a.pdf",
                    "page_count": 3,
                    "chunk_count": 10,
                    "ingested_at": "2024-01-01T00:00:00",
                }
            ]
        }
        assert P.validate_documents_response(body) is True

    def test_valid_empty(self):
        assert P.validate_documents_response({"documents": []}) is True

    def test_documents_not_list_rejected(self):
        assert P.validate_documents_response({"documents": {}}) is False

    def test_bad_record_rejected(self):
        body = {"documents": [{"document_name": "a.pdf"}]}
        assert P.validate_documents_response(body) is False

    def test_non_dict_rejected(self):
        assert P.validate_documents_response([]) is False


# ---------------------------------------------------------------------------
# page_badge_text — handles int | str
# ---------------------------------------------------------------------------
class TestPageBadge:
    def test_int_page(self):
        assert P.page_badge_text(7) == "p. 7"

    def test_str_section_page(self):
        assert P.page_badge_text("Introduction") == "Introduction"

    def test_blank_str(self):
        assert P.page_badge_text("   ") == "p. ?"

    def test_bool_defensive(self):
        assert P.page_badge_text(True) == "p. ?"


class TestFormatting:
    def test_citation_title_escaped(self):
        title = P.format_citation_title("![x](http://evil)", 2)
        assert "![x](http://evil)" not in title
        assert "p. 2" in title

    def test_excerpt_escaped(self):
        out = P.format_excerpt("[a](http://evil)")
        assert "[a](http://evil)" not in out


# ---------------------------------------------------------------------------
# Two escapers: strict (verbatim) vs answer (keeps benign formatting).
# ---------------------------------------------------------------------------
# The six adversarial / formatting payloads under test.
_IMG_PAYLOAD = "![x](https://evil.example/a.png)"
_LINK_PAYLOAD = "[click](https://evil.example)"
_REF_PAYLOAD = "![x][r]\n[r]: https://evil.example/a.png"
_HTML_PAYLOAD = "<img src=https://evil.example/a.png>"
_DIRECTIVE_PAYLOAD = ":red[x]"
_BOLD_PAYLOAD = "**bold**\n* item"


class TestEscapeStrictNeutralizesAll:
    """escape_markdown_strict must fully neutralize ALL six payloads."""

    def test_image_neutralized(self):
        out = P.escape_markdown_strict(_IMG_PAYLOAD)
        assert _IMG_PAYLOAD not in out
        assert "\\!" in out and "\\[" in out and "\\]" in out
        assert "\\(" in out and "\\)" in out

    def test_link_neutralized(self):
        out = P.escape_markdown_strict(_LINK_PAYLOAD)
        assert _LINK_PAYLOAD not in out
        assert "\\[" in out and "\\]" in out and "\\(" in out and "\\)" in out

    def test_reference_link_neutralized(self):
        out = P.escape_markdown_strict(_REF_PAYLOAD)
        # Brackets escaped so neither the ref use nor the ref definition work.
        assert "\\[" in out and "\\]" in out and "\\!" in out
        assert "[r]" not in out.replace("\\[", "").replace("\\]", "")

    def test_raw_html_neutralized(self):
        out = P.escape_markdown_strict(_HTML_PAYLOAD)
        assert "\\<" in out and "\\>" in out
        assert "<img" not in out.replace("\\<", "")

    def test_directive_neutralized(self):
        out = P.escape_markdown_strict(_DIRECTIVE_PAYLOAD)
        assert "\\[" in out and "\\]" in out
        assert ":red[x]" not in out

    def test_bold_asterisks_remain_literal(self):
        out = P.escape_markdown_strict(_BOLD_PAYLOAD)
        # Strict escapes '*', so no active bold/list forms (asterisks escaped).
        assert "\\*" in out
        assert "**bold**" not in out


class TestEscapeAnswerKeepsFormatting:
    """escape_markdown_answer neutralizes active elements but keeps emphasis."""

    def test_image_neutralized(self):
        out = P.escape_markdown_answer(_IMG_PAYLOAD)
        assert _IMG_PAYLOAD not in out
        assert "\\[" in out and "\\]" in out  # brackets escaped -> no image

    def test_link_neutralized(self):
        out = P.escape_markdown_answer(_LINK_PAYLOAD)
        assert _LINK_PAYLOAD not in out
        assert "\\[" in out and "\\]" in out  # brackets escaped -> no link

    def test_reference_link_neutralized(self):
        out = P.escape_markdown_answer(_REF_PAYLOAD)
        assert "\\[" in out and "\\]" in out
        assert "[r]" not in out.replace("\\[", "").replace("\\]", "")

    def test_raw_html_neutralized(self):
        out = P.escape_markdown_answer(_HTML_PAYLOAD)
        assert "\\<" in out
        assert "<img" not in out.replace("\\<", "")

    def test_directive_neutralized(self):
        out = P.escape_markdown_answer(_DIRECTIVE_PAYLOAD)
        assert "\\[" in out and "\\]" in out
        assert ":red[x]" not in out

    def test_bold_and_list_preserved(self):
        out = P.escape_markdown_answer(_BOLD_PAYLOAD)
        # '*' is NOT escaped by the answer escaper, so bold/list render.
        assert "\\*" not in out
        assert "**bold**" in out
        assert "* item" in out

    def test_only_brackets_and_lt_escaped(self):
        # The answer escaper escapes '[', ']' and '<' (and backslash), nothing more.
        out = P.escape_markdown_answer("[ ] < * _ # | ( ) ! > ~ `")
        assert "\\[" in out and "\\]" in out and "\\<" in out
        for ch in "*_#|()!>~`":
            assert ("\\" + ch) not in out

    def test_non_string_coerced(self):
        assert P.escape_markdown_answer(123) == "123"


# ---------------------------------------------------------------------------
# format_timestamp
# ---------------------------------------------------------------------------
class TestFormatTimestamp:
    def test_iso_with_z_and_micros(self):
        assert P.format_timestamp("2026-10-18T13:15:28.589634Z") == "18 Oct 2026 (UTC)"

    def test_iso_with_z_no_micros(self):
        assert P.format_timestamp("2026-10-18T13:15:28Z") == "18 Oct 2026 (UTC)"

    def test_iso_without_z_or_micros(self):
        assert P.format_timestamp("2026-10-18T13:15:28") == "18 Oct 2026 (UTC)"

    def test_tz_converted_to_utc(self):
        # 2026-10-18 01:30 at +03:00 == 2026-10-17 22:30 UTC -> day is 17.
        assert P.format_timestamp("2026-10-18T01:30:00+03:00") == "17 Oct 2026 (UTC)"

    def test_malformed_returns_unchanged(self):
        assert P.format_timestamp("not-a-date") == "not-a-date"

    def test_empty_returns_empty(self):
        assert P.format_timestamp("") == ""

    def test_none_returns_empty(self):
        assert P.format_timestamp(None) == ""


# ---------------------------------------------------------------------------
# group_citations
# ---------------------------------------------------------------------------
class TestGroupCitations:
    def test_empty_list(self):
        assert P.group_citations([]) == []

    def test_non_list(self):
        assert P.group_citations(None) == []

    def test_duplicates_collapse_with_passages(self):
        citations = [
            {"document": "a.pdf", "page": 1, "excerpt": "e1"},
            {"document": "a.pdf", "page": 2, "excerpt": "e2"},
            {"document": "a.pdf", "page": 2, "excerpt": "e3"},
        ]
        groups = P.group_citations(citations)
        assert len(groups) == 2
        assert groups[0] == {
            "document": "a.pdf",
            "page": 1,
            "excerpts": ["e1"],
            "passages": [1],
        }
        assert groups[1] == {
            "document": "a.pdf",
            "page": 2,
            "excerpts": ["e2", "e3"],
            "passages": [2, 3],
        }

    def test_first_appearance_order_preserved(self):
        citations = [
            {"document": "b.pdf", "page": 5, "excerpt": "x"},
            {"document": "a.pdf", "page": 1, "excerpt": "y"},
            {"document": "b.pdf", "page": 5, "excerpt": "z"},
        ]
        groups = P.group_citations(citations)
        # b.pdf/5 appears first, so it stays first even though a.pdf/1 is "lower".
        assert [(g["document"], g["page"]) for g in groups] == [
            ("b.pdf", 5),
            ("a.pdf", 1),
        ]
        assert groups[0]["passages"] == [1, 3]
        assert groups[1]["passages"] == [2]

    def test_string_pages(self):
        citations = [
            {"document": "a.pdf", "page": "Intro", "excerpt": "e1"},
            {"document": "a.pdf", "page": "Intro", "excerpt": "e2"},
        ]
        groups = P.group_citations(citations)
        assert len(groups) == 1
        assert groups[0]["page"] == "Intro"
        assert groups[0]["passages"] == [1, 2]


# ---------------------------------------------------------------------------
# grouped_citation_title + sources_label
# ---------------------------------------------------------------------------
class TestGroupedTitleAndSourcesLabel:
    def test_single_passage_suffix_singular(self):
        group = {"document": "a.pdf", "page": 2, "excerpts": ["e"], "passages": [2]}
        title = P.grouped_citation_title(group)
        assert "a.pdf" in title
        assert "p. 2" in title
        assert "· passage 2" in title
        assert "passages" not in title

    def test_multiple_passages_suffix_plural_sorted_unique(self):
        group = {
            "document": "a.pdf",
            "page": 2,
            "excerpts": ["e"],
            "passages": [3, 2, 2],
        }
        title = P.grouped_citation_title(group)
        assert "· passages 2, 3" in title

    def test_title_is_escaped(self):
        group = {
            "document": "![x](http://evil)",
            "page": 2,
            "excerpts": ["e"],
            "passages": [1],
        }
        title = P.grouped_citation_title(group)
        assert "![x](http://evil)" not in title
        assert "\\[" in title

    def test_sources_label(self):
        assert P.sources_label(0) == "Sources (0)"
        assert P.sources_label(3) == "Sources (3)"


# ---------------------------------------------------------------------------
# styles: color constants + WCAG contrast
# ---------------------------------------------------------------------------
class TestStylesContrast:
    def test_muted_on_white_passes_aa(self):
        ratio = S.contrast_ratio(S.MUTED_TEXT_HEX, S.WHITE_BG_HEX)
        assert ratio >= 4.5

    def test_muted_on_sidebar_passes_aa(self):
        ratio = S.contrast_ratio(S.MUTED_TEXT_HEX, S.SIDEBAR_BG_HEX)
        assert ratio >= 4.5

    def test_old_muted_color_failed_on_sidebar(self):
        # Documents WHY the color was darkened: #64748b failed on #f1f5f9.
        assert S.contrast_ratio("#64748b", S.SIDEBAR_BG_HEX) < 4.5

    def test_constants_present(self):
        assert S.MUTED_TEXT_HEX == "#475569"
        assert S.WHITE_BG_HEX == "#ffffff"
        assert S.SIDEBAR_BG_HEX == "#f1f5f9"


# ---------------------------------------------------------------------------
# Dollar escaping — prevents accidental Streamlit $...$ LaTeX/math rendering.
# ---------------------------------------------------------------------------
class TestDollarEscaping:
    def _no_unescaped_dollar(self, out: str) -> bool:
        """True if every '$' in ``out`` is immediately preceded by a backslash."""
        return all(i > 0 and out[i - 1] == "\\" for i, ch in enumerate(out) if ch == "$")

    def test_strict_escapes_every_dollar(self):
        out = P.escape_markdown_strict("fee is $5 and $10")
        assert "\\$5" in out
        assert "\\$10" in out
        # No '$' may start LaTeX: every '$' must be backslash-escaped.
        assert self._no_unescaped_dollar(out)

    def test_answer_escapes_every_dollar(self):
        out = P.escape_markdown_answer("fee is $5 and $10")
        assert "\\$5" in out
        assert "\\$10" in out
        assert self._no_unescaped_dollar(out)

    def test_answer_escapes_dollar_but_keeps_bold(self):
        out = P.escape_markdown_answer("**bold** costs $5")
        # Emphasis stays live...
        assert "**bold**" in out
        assert "\\*" not in out
        # ...while the dollar is escaped.
        assert "\\$5" in out
        assert self._no_unescaped_dollar(out)


# ---------------------------------------------------------------------------
# format_page_count — singular/plural grammar + defensive fallback.
# ---------------------------------------------------------------------------
class TestFormatPageCount:
    def test_one_is_singular(self):
        assert P.format_page_count(1) == "1 page"

    def test_two_is_plural(self):
        assert P.format_page_count(2) == "2 pages"

    def test_zero_is_plural(self):
        assert P.format_page_count(0) == "0 pages"

    def test_non_int_fallback_is_plural(self):
        assert P.format_page_count("?") == "? pages"
