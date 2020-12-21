import typing as T
import tensorflow as TF
import numpy as NP
import json as J
import warnings as W

from collections import Counter
from os import path

import utility as U
import spells as S

def createLookupTable(data: str) -> dict:
    """Build vocabulary for machine learning
    
    :param data: The string from which to build the lookup table.
    :return: A tuple containing two maps associating vocab and integer values"""
    counts = Counter(data)
    vocab = sorted(counts, key=counts.get, reverse=True)
    vocab_to_int = {word: ii for ii, word in enumerate(vocab)}
    return vocab_to_int

def main():
    # PREPROCESSING
    preprocessed_path = "preprocessed.p"
    if not path.exists(preprocessed_path):
        spell_list_json = U.loadJSONData("spell_data.json")
        descs = S.Spells.spellsToSpellKeys(spell_list_json, "desc")
        data_descs = "\n\n".join(["\n".join(d) for d in descs])
        U.preprocessAndSaveData(data_descs, createLookupTable, preprocessed_path)
    int_data, vocab_to_int, int_to_vocab, tokens = U.loadPickle(preprocessed_path)

    # BUILDING THE NEURAL NETWORK
    # Check for a GPU
    if not TF.test.gpu_device_name():
        W.warn('No GPU found. Please use a GPU to train the neural network.')
    else:
        print(f"Default GPU Device: {TF.test.gpu_device_name()}")

    
