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
        
        indexes = self.spellsToSpellKeys(r.json()["results"], "index")
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
        """TODO Document"""
        spells = []
        indexes = self.getAllSpellIndexes()
        for index in indexes:
            spells.append(self.getSpell(index))
        return spells

    def loadJSONData(self,
                     filepath: str) -> T.Iterable[dict]:
        """TODO Document"""
        # Create data file if it does not yet exist
        if not path.exists(filepath):
            spells_data = {"spells":self.getAllSpells()}
            with open(filepath, "w") as f:
                J.dump(spells_data, f)
                return spells_data["spells"]
        # Load data file if it does currently exist
        with open(filepath, "r") as f:
            spells_data = J.load(f)
        return spells_data["spells"]

    @staticmethod
    def spellsToSpellKeys(spell_list_json: T.Iterable[dict],
                          key: str) -> T.Union[T.Iterable[str],
                                               T.Iterable[dict]]:
        """TODO Document"""
        return [s[key] for s in spell_list_json]
    
    @staticmethod
    def getSpellURL(index: str) -> str:
        """TODO Document"""
        return f"{SPELLS_INDEX_URL}/{index}"
