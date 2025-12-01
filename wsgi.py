"""WSGI entry point for Vercel deployment."""
from app.server import app

__all__ = ["app"]
