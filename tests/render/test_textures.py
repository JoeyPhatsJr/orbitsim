"""Tests for the texture download/cache (offline-safe)."""
import os
from orbitsim.render import textures


def test_registry_has_earth_keys():
    assert "earth_day" in textures.TEXTURE_URLS
    assert "earth_night" in textures.TEXTURE_URLS


def test_registry_has_stars_key():
    assert "stars" in textures.TEXTURE_URLS
    assert textures.TEXTURE_URLS["stars"].endswith(".png")


def test_cache_hit_returns_existing_valid_file(tmp_path):
    # Pre-place a file with a valid JPEG magic; no network should be needed.
    p = tmp_path / "earth_day.jpg"
    p.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 16)
    got = textures.texture_path("earth_day", cache_dir=str(tmp_path))
    assert got == str(p)


def test_unknown_name_returns_none(tmp_path):
    assert textures.texture_path("not_a_planet", cache_dir=str(tmp_path)) is None


def test_bad_magic_is_rejected(tmp_path, monkeypatch):
    # Simulate a download that returns an HTML CAPTCHA page (no image magic).
    def fake_fetch(url, timeout=30):
        return b"<html>captcha</html>"
    monkeypatch.setattr(textures, "_fetch", fake_fetch)
    assert textures.texture_path("earth_day", cache_dir=str(tmp_path)) is None
