"""Central visual language for the KSP-inspired interface."""
from dataclasses import dataclass
from pathlib import Path


Color = tuple[float, float, float, float]


@dataclass(frozen=True)
class UiTheme:
    panel: Color = (0.025, 0.045, 0.070, 0.92)
    panel_alt: Color = (0.055, 0.085, 0.115, 0.96)
    panel_edge: Color = (0.18, 0.36, 0.48, 1.0)
    control: Color = (0.08, 0.13, 0.17, 0.98)
    control_hover: Color = (0.12, 0.22, 0.28, 1.0)
    control_pressed: Color = (0.04, 0.36, 0.46, 1.0)
    disabled: Color = (0.17, 0.19, 0.21, 0.70)
    text: Color = (0.90, 0.96, 1.0, 1.0)
    text_muted: Color = (0.55, 0.67, 0.73, 1.0)
    cyan: Color = (0.20, 0.90, 1.0, 1.0)
    green: Color = (0.35, 1.0, 0.55, 1.0)
    amber: Color = (1.0, 0.72, 0.24, 1.0)
    magenta: Color = (1.0, 0.34, 0.92, 1.0)
    danger: Color = (1.0, 0.28, 0.22, 1.0)
    padding: float = 0.025
    row_height: float = 0.058
    title_scale: float = 0.048
    body_scale: float = 0.040


THEME = UiTheme()


def font_path(semibold: bool = False) -> str:
    name = "Rajdhani-SemiBold.ttf" if semibold else "Rajdhani-Regular.ttf"
    return str(Path(__file__).resolve().parents[1] / "assets" / "fonts" / name)


def install_default_font(base) -> None:
    """Install the bundled OFL UI face for every subsequently-created TextNode."""
    from panda3d.core import Filename, TextNode

    panda_path = Filename.from_os_specific(font_path()).get_fullpath()
    font = base.loader.load_font(panda_path)
    if font is not None:
        TextNode.set_default_font(font)


def button_options(theme: UiTheme = THEME) -> dict:
    """DirectButton options shared by every interactive surface."""
    return {
        "frameColor": (
            theme.control,
            theme.control_pressed,
            theme.control_hover,
            theme.disabled,
        ),
        "text_fg": theme.text,
        "text_scale": 0.82,
        "borderWidth": (0.012, 0.012),
        "relief": 1,
    }
