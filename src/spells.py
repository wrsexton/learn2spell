import typing as T
import requests as R
import backoff as BO
import os

API_URL="https://www.dnd5eapi.co/api"
SPELLS_INDEX_URL=f"{API_URL}/spells"

class Spells():
    """An interface to the dnd5e API"""

    def __init__(self):
        self._s = R.Session()

    @BO.on_exception(BO.expo,
                     R.exceptions.RequestException,
                     max_tries=10,
                     jitter=None)
    def getAllSpellIndexes(self) -> T.Iterable[str]:
        """TODO Document"""
        r = self._s.get(SPELLS_INDEX_URL)

        if(r.status_code != 200):
            raise Exception(f"({r.status_code}) An error occurred when accessing {SPELLS_INDEX_URL}")
        
        indexes = [s["index"] for s in r.json()["results"]]
        return indexes

    @BO.on_exception(BO.expo,
                     R.exceptions.RequestException,
                     max_tries=10,
                     jitter=None)
    def getSpell(self,
                 index: str) -> dict:
        """TODO Document"""
        url = self.getSpellURL(index)
        r = self._s.get(url)

        if(r.status_code != 200):
            raise Exception(f"({r.status_code}) An error occurred when accessing {url}")

        return r.json()

    def getAllSpells(self) -> T.Iterable[dict]:
        indexes = self.getAllSpellIndexes
    
    @staticmethod
    def getSpellURL(index):
        """TODO Document"""
        return f"{SPELLS_INDEX_URL}/{index}"
