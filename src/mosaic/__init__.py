"""Photo mosaic generator package.

Public API:
    generate_mosaic_photo: top-level entry point that builds a mosaic image.
    MosaicConfig: dataclass holding generation parameters.
"""

from mosaic.core import MosaicConfig, generate_mosaic_photo

__all__ = ["MosaicConfig", "generate_mosaic_photo"]
__version__ = "0.2.0"
