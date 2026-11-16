"""The logo files the napari manifest and the README point at are really shipped."""

import re
from importlib import resources
from pathlib import Path

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def test_manifest_icon_is_a_shipped_png():
    manifest = resources.files("pyclem3d.napari").joinpath("napari.yaml").read_text("utf-8")
    m = re.search(r"^icon:\s*(\S+):(\S+)\s*$", manifest, flags=re.MULTILINE)
    assert m, "napari.yaml has no `icon: package:resource` line"
    icon = resources.files(m.group(1)).joinpath(m.group(2))
    assert icon.is_file(), f"{m.group(0)!r} does not resolve to a file"
    assert icon.read_bytes().startswith(PNG_MAGIC)


def test_readme_images_exist():
    readme = Path(__file__).resolve().parents[1] / "README.md"
    refs = re.findall(r'(?:src|srcset)="([^"]+)"', readme.read_text("utf-8"))
    local = [r for r in refs if not r.startswith(("http://", "https://"))]
    assert any("pyclem3d_logo" in r for r in local), "README.md no longer shows the logo"
    assert any("how_it_works" in r for r in local), "README.md no longer shows the infographic"
    for ref in local:
        assert (readme.parent / ref).is_file(), ref
