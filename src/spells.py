import typing as T
import requests as R
import json
import os

API="https://www.dnd5eapi.co/api"

class Spells():
    """An interface to the dnd5e API"""

    def __init__(self):
        self._s = R.Session()

    def getAllSpellIndexes(self):
        """TODO Document"""
        r = self._s.get(f"{API}/spells")
        j = r.json()
        return j
