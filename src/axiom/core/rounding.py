"""How many digits of a number a reader is entitled to see.

A posterior mean of ``12.3456789`` beside a 90% interval half a unit wide is
seven digits of which two are knowledge and five are arithmetic. Printing all
seven is not neutral: a reader takes trailing digits as precision, compares two
results on digits that are noise, and quotes a number to a room at a resolution
the data never supported. The interval is the honest statement; the point
estimate should not out-run it.

So every printed number in axiom is rounded to the place its own uncertainty
reaches. The convention is the usual one in measurement reporting: keep two
significant digits of the uncertainty, and round the value to that same
decimal place.

    >>> format_measured(12.3456789, 0.52)
    '12.35'
    >>> format_measured(12.3456789, 5.2)
    '12.3'
    >>> format_measured(12.3456789, 52.0)
    '12'

This is a *display* rule and only a display rule. Nothing here touches a stored
value, a serialized spec, or a ``detail`` dict: `io` writes seventeen digits on
purpose, and a failure's ``detail`` is machine-read provenance rather than
prose. Round on the way to a human, never on the way to a file.

When the uncertainty is unknown there is nothing to round against, and the
functions that accept ``None`` fall back to a plain ``%g`` — an unrounded
number is the honest answer to "how precise is this?" when the answer is "no
one said".
"""

from __future__ import annotations

import math

__all__ = [
    "DIGITS",
    "decimals_for",
    "format_interval",
    "format_measured",
    "round_to",
]

DIGITS = 2
"""Significant digits kept of the uncertainty itself.

Two is the common convention and the one this package uses everywhere. One is
defensible and loses the difference between a half-unit and a unit; three
claims the third digit of a standard error is meaningful, which for a
Monte Carlo estimate of it is rarely true.
"""

#: Below and above these exponents a fixed-point number stops being readable and
#: the exponent form takes over — the same reasoning as ``%g``, with the upper
#: cut at ten million rather than at the digit count, because a budget of
#: 1234000 reads better than 1.234e+06 and 5e+09 does not.
_SMALL_EXPONENT = -5
_LARGE_EXPONENT = 7


def _place(uncertainty: float, digits: int) -> int:
    """The power of ten of the last digit of ``uncertainty`` worth keeping."""
    u = abs(float(uncertainty))
    if not math.isfinite(u):
        raise ValueError(f"uncertainty must be finite, got {uncertainty}")
    if u == 0.0:
        raise ValueError("uncertainty must be non-zero; a value with no spread has no last digit")
    if digits < 1:
        raise ValueError(f"digits must be at least 1, got {digits}")
    exponent = math.floor(math.log10(u))
    place = exponent - (digits - 1)
    # Rounding can carry the uncertainty into the next decade — 0.0999 to two
    # digits is 0.10, whose last kept digit sits one place higher — and keeping
    # the old place would print a third digit of it.
    if math.floor(math.log10(round(u, -place))) > exponent:
        place += 1
    return place


def decimals_for(uncertainty: float, *, digits: int = DIGITS) -> int:
    """Decimal places warranted by ``uncertainty``, for ``round`` and ``format``.

    Negative when the uncertainty is larger than one and the last meaningful
    digit sits to the *left* of the point: an effect of 1234 give or take 500 is
    known to the nearest ten, so this returns ``-1`` and ``round(1234, -1)`` is
    the number to print.

    Raises on a zero or non-finite uncertainty rather than inventing a width for
    it; callers that may not have one should use ``format_measured``, which
    documents its fallback.
    """
    return -_place(uncertainty, digits)


def round_to(value: float, uncertainty: float, *, digits: int = DIGITS) -> float:
    """``value`` rounded to the place ``uncertainty`` reaches.

    The numeric form of ``format_measured``, for the callers that need to keep
    computing — an axis tick, a table column, a threshold comparison — rather
    than to print. Note that a float cannot always *hold* the rounded value
    exactly; the string form is the one that never lies.
    """
    return round(float(value), decimals_for(uncertainty, digits=digits))


def format_measured(
    value: float,
    uncertainty: float | None = None,
    *,
    digits: int = DIGITS,
    fallback: int = 6,
    group: bool = False,
) -> str:
    """``value`` as text, carrying only the digits ``uncertainty`` supports.

    ``uncertainty`` is whatever states the resolution of the value: a standard
    error, a posterior standard deviation, the half-width of an interval. With
    ``None``, a non-finite, or a zero uncertainty there is no resolution to
    apply and the number is formatted at ``fallback`` significant digits, which
    is what it did before this module existed.

    Trailing zeros are kept, because they are the claim: ``12.30`` says the
    hundredths digit was resolved and ``12.3`` says it was not.

    ``group`` puts thousands separators in the fixed form, for the places that
    already had them — a report reads ``1,234,000``, a terminal card does not.
    """
    x = float(value)
    sep = "," if group else ""
    if uncertainty is None or not math.isfinite(float(uncertainty)) or float(uncertainty) == 0.0:
        return f"{x:{sep}.{fallback}g}"
    if not math.isfinite(x):
        return f"{x:g}"
    place = _place(float(uncertainty), digits)
    rounded = round(x, -place) or 0.0  # `or` normalises a rounded-away -0.0
    exponent = 0 if rounded == 0.0 else math.floor(math.log10(abs(rounded)))
    if _SMALL_EXPONENT < exponent < _LARGE_EXPONENT:
        return f"{rounded:{sep}.{max(-place, 0)}f}"
    return f"{rounded:.{max(exponent - place, 0)}e}"


def format_interval(
    lower: float,
    upper: float,
    *,
    uncertainty: float | None = None,
    digits: int = DIGITS,
    fallback: int = 6,
    group: bool = False,
) -> str:
    """``[lower, upper]`` with both bounds at the precision the width supports.

    The uncertainty of an interval is the interval: its half-width is the ``±``
    a reader converts it to, and a digit finer than that distinguishes nothing.
    Both bounds are rounded to the *same* place, so the pair stays readable as a
    pair — one bound at two decimals beside another at four is the artefact this
    avoids.

    ``uncertainty`` overrides that half-width, for a caller showing the interval
    beside a standard error: a 95% band is about two sigma wide, and rounding
    its bounds to the band while rounding the estimate to the sigma prints one
    result at two resolutions.
    """
    half_width = 0.5 * (float(upper) - float(lower)) if uncertainty is None else float(uncertainty)
    if not math.isfinite(half_width) or half_width == 0.0:
        # a degenerate or unusable width states no resolution, so neither do we
        lo = format_measured(lower, fallback=fallback, group=group)
        hi = format_measured(upper, fallback=fallback, group=group)
        return f"[{lo}, {hi}]"
    lo = format_measured(lower, half_width, digits=digits, group=group)
    hi = format_measured(upper, half_width, digits=digits, group=group)
    return f"[{lo}, {hi}]"
