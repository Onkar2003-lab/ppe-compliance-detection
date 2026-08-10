"""The demonstration's dashboard: a local web console for the zone compliance monitor (O6).

Run it with ``python -m src.dashboard`` and open the printed address.
"""

from src.dashboard.app import create_app, main

__all__ = ["create_app", "main"]
