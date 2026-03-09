"""Canonical data gateway import path for all tabs.

This module intentionally re-exports the governed gateway implementation
from app.data_gateway to keep backward compatibility while enforcing
`app.data.data_gateway` as the canonical import location.
"""
from __future__ import annotations

from app.data_gateway import *  # noqa: F401,F403

