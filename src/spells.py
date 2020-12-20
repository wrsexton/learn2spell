import typing as T
import requests as R
import os

API_URL="https://www.dnd5eapi.co/api"
SPELLS_INDEX_URL=f"{API_URL}/spells"

class Spells():
    """An interface to the dnd5e API"""

    def __init__(self):
        self._s = R.Session()

    def getAllSpellIndexes(self):
        """TODO Document"""
        r = self._s.get(SPELLS_INDEX_URL)
        indexes = [s["index"] for s in r.json()["results"]]
        return indexes
