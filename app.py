"""Vercel / WSGI entry shim: Flask app is defined in ``web_ui.app``."""

import web_ui.app as web_ui_application

app = web_ui_application.app
