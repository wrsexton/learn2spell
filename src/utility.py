import json as J
import pickle as P

from os import path

import spells as S

def loadJSONData(filepath):
    """TODO Document"""
    # Create data file if it does not yet exist
    if not path.exists(filepath):
        spells_data = {"spells":S.Spells().getAllSpells()}
        with open(filepath, "w") as f:
            J.dump(spells_data, f)
        return spells_data["spells"]
    # Load data file if it does currently exist
    with open(filepath, "r") as f:
        spells_data = J.load(f)
    return spells_data["spells"]
