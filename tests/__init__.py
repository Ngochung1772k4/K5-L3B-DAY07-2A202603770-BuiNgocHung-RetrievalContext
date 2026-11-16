"""Ensure UTF-8 default for Path.read_text/write_text on Windows cp1252."""
from pathlib import Path

_orig_read_text = Path.read_text
_orig_write_text = Path.write_text


def _utf8_read_text(self, encoding=None, errors=None, *args, **kwargs):
    return _orig_read_text(self, encoding=encoding or "utf-8", errors=errors, *args, **kwargs)


def _utf8_write_text(self, data, encoding=None, errors=None, *args, **kwargs):
    return _orig_write_text(
        self, data, encoding=encoding or "utf-8", errors=errors, *args, **kwargs
    )


Path.read_text = _utf8_read_text  # type: ignore[method-assign]
Path.write_text = _utf8_write_text  # type: ignore[method-assign]
