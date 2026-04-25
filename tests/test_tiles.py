"""Tests for ``mosaic.tiles``.

Only the core algorithmic primitive lives here: ``average_rgb`` is what
feeds the matcher, so getting it wrong would silently degrade every
mosaic. File-system discovery and decoding are validated transitively
by the end-to-end tests in ``test_core.py``.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from mosaic.tiles import average_rgb


def test_average_rgb_two_halves() -> None:
    # Left half white, right half black -> mean is mid-gray. This catches
    # both axis confusion (rows vs cols) and any off-by-one in the reduction,
    # which a solid-color test would not.
    image = Image.new("RGB", (10, 10), (0, 0, 0))
    white = Image.new("RGB", (5, 10), (255, 255, 255))
    image.paste(white, (0, 0))

    avg = average_rgb(image)

    np.testing.assert_allclose(avg, [127.5, 127.5, 127.5])
