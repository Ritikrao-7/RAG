"""Vercel entry point. Vercel serves the ASGI app named `app` from this file."""

from app.main import app

__all__ = ["app"]
