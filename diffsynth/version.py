"""Version information for diffsynth."""



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

        import tomllib

        with open(pyproject_path, "rb") as f:
            data = tomllib.load(f)
            return data.get("project", {}).get("version", "0.0.0")
    except Exception:
        pass

    return "0.0.0"


__version__ = _get_version()
