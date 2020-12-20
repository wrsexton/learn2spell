import typing as T
import tensorflow as TF
import numpy as NP
import json as J

from collections import Counter

import utility as U
import spells as S

def createLookupTables(data: str) -> T.Iterable[dict]:
    """Build vocabulary for machine learning
    
    :param data: The string from which to build the lookup table.
    :return: A tuple containing two maps associating vocab and integer values"""
    counts = Counter(data)
    vocab = sorted(counts, key=counts.get, reverse=True)
    vocab_to_int = {word: ii for ii, word in enumerate(vocab)}
    int_to_vocab = {v:k for k, v in vocab_to_int.items()}
    return vocab_to_int, int_to_vocab

def main():
    spell_list_json = U.loadJSONData("spell_data.json")
    descs = S.Spells.spellsToSpellKeys(spell_list_json, "desc")
    vocab_to_int, int_to_vocab = createLookupTables(
        "\n\n".join(["\n".join(d) for d in descs]))
    
    
