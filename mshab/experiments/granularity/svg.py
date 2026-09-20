"""The one SVG text primitive the experiment's two renderers share."""

from __future__ import annotations

from xml.sax.saxutils import escape


def text(
    x: float,
    y: float,
    value: object,
    *,
    size: float = 13,
    weight: int = 400,
    fill: str = "#2f3a4d",
    anchor: str = "start",
) -> str:
    """One ``<text>`` element with ``value`` escaped."""

    return (
        '<text x="{x}" y="{y}" font-size="{size}" font-weight="{weight}" '
        'fill="{fill}" text-anchor="{anchor}">{value}</text>'
    ).format(
        x=x, y=y, size=size, weight=weight, fill=fill, anchor=anchor,
        value=escape(str(value)),
    )
