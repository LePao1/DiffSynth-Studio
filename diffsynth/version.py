"""Version information for diffsynth."""

import re
import sys


def _get_version():
    """Get version from installed package metadata or pyproject.toml."""
    try:
        # Try installed package metadata (when installed from wheel)
        from importlib.metadata import version

        return version("diffsynth")
    except Exception:
        pass

    try:
        # Try parsing pyproject.toml (when running from source tree)
        import os

        pyproject_path = os.path.join(os.path.dirname(__file__), "..", "pyproject.toml")

        if sys.version_info >= (3, 11):  # noqa: UP036 – project supports Python 3.10 which lacks tomllib
            import tomllib

            with open(pyproject_path, "rb") as f:
                data = tomllib.load(f)
                return data.get("project", {}).get("version", "0.0.0")
        else:
            # For Python < 3.11, read and parse manually
            with open(pyproject_path, encoding="utf-8") as f:
                content = f.read()
                # Simple regex to extract version from [project] section
                match = re.search(r'^version\s*=\s*"([^"]+)"', content, re.MULTILINE)
                if match:
                    return match.group(1)
                return "0.0.0"
    except Exception:
        pass

    return "0.0.0"


__version__ = _get_version()
