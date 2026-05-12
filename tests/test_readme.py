from pathlib import Path

import pytest


def test_readme_renders():
    """Verify README.rst renders correctly for PyPI."""
    try:
        from readme_renderer.rst import render
    except ImportError:
        pytest.skip("readme_renderer not installed")

    readme_path = Path(__file__).parent.parent.parent / "README.rst"
    with open(readme_path, "r") as f:
        readme_content = f.read()

    rendered = render(readme_content)
    assert rendered is not None, "README.rst failed to render as valid RST"
