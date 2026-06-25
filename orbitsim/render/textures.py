"""Download + cache real surface texture maps; offline-safe (flat-color fallback
is the caller's job when this returns None)."""
import os
import urllib.request

_BASE = "https://raw.githubusercontent.com/mrdoob/three.js/dev/examples/textures/planets/"
TEXTURE_URLS = {
    "earth_day": _BASE + "earth_atmos_2048.jpg",
    "earth_night": _BASE + "earth_lights_2048.png",
    "stars": "https://raw.githubusercontent.com/jeromeetienne/threex.planets/master/images/galaxy_starfield.png",
}

_DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "textures"
)
# Image magic numbers: JPEG starts FF D8, PNG starts 89 50 4E 47.
_MAGICS = (b"\xff\xd8", b"\x89PNG")


def _ext_for(url: str) -> str:
    return ".png" if url.lower().endswith(".png") else ".jpg"


def _looks_like_image(data: bytes) -> bool:
    return any(data.startswith(m) for m in _MAGICS)


def _fetch(url: str, timeout: int = 30) -> bytes:
    """Download raw bytes with a browser-like User-Agent."""
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    return urllib.request.urlopen(req, timeout=timeout).read()


def texture_path(name: str, cache_dir: str | None = None):
    """Return a cached local path for `name`, downloading on first use.

    Returns None if the name is unknown, the download fails, or the bytes are not
    a valid image (e.g. a CAPTCHA HTML page). The caller falls back to a flat color.
    """
    url = TEXTURE_URLS.get(name)
    if url is None:
        return None
    cache_dir = cache_dir or _DATA_DIR
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f"{name}{_ext_for(url)}")
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return path
    try:
        data = _fetch(url)
    except Exception:
        return None
    if not _looks_like_image(data):
        return None
    tmp = path + ".part"
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, path)
    return path
