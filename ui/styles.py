"""Static CSS for the UI.

This module holds the ONE and ONLY block of HTML/CSS that the app renders with
``unsafe_allow_html=True``. It is a hard-coded, static string constant: it
contains NO dynamic values, NO interpolation, and NO user/backend data. Because
nothing variable is ever placed inside it, there is no injection surface here.

All dynamic content (answers, excerpts, document names, filenames) is rendered
elsewhere as ESCAPED plain text via normal ``st.write``/``st.markdown`` WITHOUT
``unsafe_allow_html`` (see :mod:`ui.presentation.escape_markdown`).
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Color constants (exposed so tests can recompute contrast from them).
#
# Muted/helper text was darkened from #64748b (slate-500) to #475569 (slate-600)
# for WCAG AA. Measured contrast ratios:
#   #475569 on #ffffff (white main bg)       = 7.58:1  (PASS, >= 4.5:1)
#   #475569 on #f1f5f9 (sidebar/secondary bg) = 6.92:1  (PASS, >= 4.5:1)
# The OLD #64748b FAILED on the sidebar/secondary background:
#   #64748b on #ffffff = 4.76:1 (pass) BUT #64748b on #f1f5f9 = 4.34:1 (FAIL).
# ---------------------------------------------------------------------------
MUTED_TEXT_HEX = "#475569"   # slate-600: muted/helper text (AA on white + sidebar)
WHITE_BG_HEX = "#ffffff"     # white main background
SIDEBAR_BG_HEX = "#f1f5f9"   # light neutral sidebar / secondary background


def _srgb_channel_luminance(channel_8bit: int) -> float:
    """Linearize one 8-bit sRGB channel per the WCAG relative-luminance formula."""
    c = channel_8bit / 255.0
    if c <= 0.03928:
        return c / 12.92
    return ((c + 0.055) / 1.055) ** 2.4


def _relative_luminance(hex_color: str) -> float:
    """Return the WCAG relative luminance (0..1) of a ``#rrggbb`` color."""
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i : i + 2], 16) for i in (0, 2, 4))
    return (
        0.2126 * _srgb_channel_luminance(r)
        + 0.7152 * _srgb_channel_luminance(g)
        + 0.0722 * _srgb_channel_luminance(b)
    )


def contrast_ratio(hex_fg: str, hex_bg: str) -> float:
    """Return the WCAG contrast ratio between a foreground and background color.

    Pure function. Ratio is ``(L_lighter + 0.05) / (L_darker + 0.05)`` using the
    WCAG relative-luminance formula, so the result is in ``[1.0, 21.0]``.
    """
    l1 = _relative_luminance(hex_fg)
    l2 = _relative_luminance(hex_bg)
    lighter, darker = max(l1, l2), min(l1, l2)
    return (lighter + 0.05) / (darker + 0.05)


# ---------------------------------------------------------------------------
# Static CSS. Keep this minimal and commented. No dynamic values may ever be
# concatenated into this string.
# ---------------------------------------------------------------------------
CSS: str = """
<style>
/* Neutral answer card: used for normal answers and the empty-citations case.
   Deliberately NOT styled as an error/warning — just a calm surface. */
.adi-card {
    background-color: #f8fafc;      /* very light neutral */
    border: 1px solid #e2e8f0;      /* subtle border */
    border-left: 4px solid #0f766e; /* teal accent rail */
    border-radius: 8px;
    padding: 1rem 1.15rem;
    margin: 0.5rem 0 0.75rem 0;
}

/* Small pill used to show a citation's page/section badge. */
.adi-page-badge {
    display: inline-block;
    background-color: #0f766e;      /* teal */
    color: #ffffff;
    font-size: 0.75rem;
    font-weight: 600;
    line-height: 1;
    padding: 0.25rem 0.55rem;
    border-radius: 999px;
    margin-left: 0.4rem;
    vertical-align: middle;
}

/* A single row in the sidebar document list. */
.adi-doc-row {
    padding: 0.5rem 0.65rem;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    margin-bottom: 0.5rem;
    background-color: #ffffff;
}

/* Muted / helper text. Darkened to #475569 (slate-600) for WCAG AA:
   #475569 on #ffffff = 7.58:1 and on #f1f5f9 = 6.92:1 (both >= 4.5:1).
   The previous #64748b failed on the #f1f5f9 sidebar background (4.34:1). */
.adi-muted {
    color: #475569;
    font-size: 0.85rem;
}

/* Sidebar meta lines (page count, ingested date), the upload hint, and the
   "Sources (n)" label all use the same darker muted tone so they clear AA on
   both the white main background and the #f1f5f9 sidebar/secondary surfaces.
   Streamlit renders st.caption as a <small> inside the block; target it plus
   our explicit helper classes. */
section[data-testid="stSidebar"] small,
.adi-doc-meta,
.adi-upload-hint,
.adi-sources-label {
    color: #475569;   /* slate-600 — AA on #ffffff (7.58:1) and #f1f5f9 (6.92:1) */
}
</style>
"""


def inject_css() -> None:
    """Inject the static CSS block into the Streamlit page.

    This is the ONLY place in the whole UI that passes
    ``unsafe_allow_html=True``, and it renders the static :data:`CSS` constant
    only — never any dynamic value. Imported lazily inside the function so that
    importing this module does not require Streamlit and has no side effects.
    """
    import streamlit as st

    st.markdown(CSS, unsafe_allow_html=True)
