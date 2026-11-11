"""Ground-truth-free structured metadata derived from the query images.

These features are computed from pixels and EXIF only. They never read folder
names, file names, case IDs, or answer keys, so they are safe inputs to Jev and
the routing policy.
"""

from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Any, Union

from PIL import ExifTags, Image, ImageStat

ImageSource = Union[str, Path, bytes]

_JEV_MAX_EDGE = 1024


def _open(source: ImageSource) -> Image.Image:
    if isinstance(source, (str, Path)):
        return Image.open(source)
    return Image.open(io.BytesIO(source))


def describe_image(source: ImageSource) -> dict[str, Any]:
    """Return coarse, answer-free properties of one image."""
    with _open(source) as img:
        width, height = img.size
        exif_year = None
        try:
            exif = img.getexif()
            tags = {ExifTags.TAGS.get(k, k): v for k, v in exif.items()}
            stamp = tags.get("DateTimeOriginal") or tags.get("DateTime")
            if isinstance(stamp, str) and stamp[:4].isdigit():
                exif_year = int(stamp[:4])
        except Exception:  # EXIF is optional and often stripped
            pass
        rgb = img.convert("RGB")
        rgb.thumbnail((256, 256))
        hsv = rgb.convert("HSV")
        mean_saturation = ImageStat.Stat(hsv).mean[1] / 255.0
        spread = chroma_spread(rgb)
    return {
        "width": width,
        "height": height,
        "aspect_ratio": round(width / height, 3) if height else None,
        "mean_saturation": round(mean_saturation, 3),
        "chroma_spread": round(spread, 1),
        # Sepia prints are tinted (non-zero saturation) but nearly uniform in
        # hue, so the spread of opponent-colour channels is the better signal.
        "likely_monochrome": spread < MONOCHROME_SPREAD_THRESHOLD,
        "exif_year": exif_year,
    }


# Note: chosen after inspecting the six pilot images (the 1918 photo scores
# 12.4; every colour photo scores >= 26). Treat as provisional.
MONOCHROME_SPREAD_THRESHOLD = 18.0


def chroma_spread(rgb: Image.Image) -> float:
    """Std. dev. of opponent colour channels (rg, yb), Hasler-Suesstrunk style."""
    getter = getattr(rgb, "get_flattened_data", None) or rgb.getdata
    pixels = list(getter())
    n = len(pixels) or 1
    rg = [r - g for r, g, _ in pixels]
    yb = [0.5 * (r + g) - b for r, g, b in pixels]

    def var(xs: list[float]) -> float:
        m = sum(xs) / n
        return sum((x - m) ** 2 for x in xs) / n

    return (var(rg) + var(yb)) ** 0.5


def jev_image_block(source: ImageSource) -> dict[str, Any]:
    """Encode an image as a LangChain image content block for Jev.

    TypeSafe accepts PNG and JPEG only, so everything is re-encoded as a
    downscaled JPEG.
    """
    with _open(source) as img:
        rgb = img.convert("RGB")
        rgb.thumbnail((_JEV_MAX_EDGE, _JEV_MAX_EDGE))
        buffer = io.BytesIO()
        rgb.save(buffer, format="JPEG", quality=88)
    return {
        "type": "image",
        "base64": base64.b64encode(buffer.getvalue()).decode("ascii"),
        "mime_type": "image/jpeg",
    }
