import colorsys
import re
from typing import Optional, Tuple

# ----- 색상 관련 유틸 -----
COLOR_LABELS = ("red", "green", "yellow", "purple", "white")
VALID_COLORS = {label: label for label in COLOR_LABELS}
COLOR_HEX_MAP = {
    "red": "#f52629",
    "green": "#48ad0d",
    "white": "#ffffff",
    "yellow": "#ffdd00",
    "purple": "#781de7",
}
HEX_TO_COLOR_MAP = {hex_val: label for label, hex_val in COLOR_HEX_MAP.items()}


def normalize_color_hex(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if not text.startswith("#"):
        text = f"#{text}"
    if re.match(r"^#[0-9a-fA-F]{6}$", text):
        return text.lower()
    return None


def normalize_color_label(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    color = str(value).strip().lower()
    if not color or color == "none":
        return None
    return color if color in VALID_COLORS else None


def color_label_to_hex(color: Optional[str]) -> Optional[str]:
    if not color:
        return None
    return COLOR_HEX_MAP.get(color)


def _hex_to_rgb(hex_value: str) -> Tuple[int, int, int]:
    return tuple(int(hex_value[idx:idx + 2], 16) for idx in (1, 3, 5))


def hex_to_color_label(value: Optional[str]) -> Optional[str]:
    hex_value = normalize_color_hex(value)
    if not hex_value:
        return None
    if hex_value in HEX_TO_COLOR_MAP:
        return HEX_TO_COLOR_MAP[hex_value]

    r, g, b = _hex_to_rgb(hex_value)
    h, s, v = colorsys.rgb_to_hsv(r / 255.0, g / 255.0, b / 255.0)
    hue = h * 360.0

    # Keep ranges simple; adjust as needed for your palette.
    if s < 0.12:
        return "white" if v > 0.5 else "purple"

    if (0 <= hue < 30) or (330 <= hue < 360):
        return "red"
    if 30 <= hue < 80:
        return "yellow"
    if 80 <= hue < 170:
        return "green"
    # remaining hues map to purple
    return "purple"
