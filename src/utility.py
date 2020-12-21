import typing as T
import json as J
import pickle as P

from os import path

import spells as S

# Punctuation Tokens
TOKENS={
    "." : "||period||",
    "," : "||comma||",
    ";" : "||semicolon||",
    "\"" : "||quotation_mark||",
    "?" : "||question_mark||",
    ";" : "||semicolon||",
    "(" : "||left_parantheses||",
    ")" : "||right_parantheses||",
    "-" : "||dash||",
    "!" : "||exclamation_mark||",
    "\n" : "||return||"
}

def loadJSONData(filepath: str) -> T.Iterable[dict]:
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

def preprocessAndSaveData(data: str,
                          lookup_creation_func: T.Any,
                          filepath: str,
                          tokens: T.Optional[dict]=TOKENS):
    for key, token in TOKENS.items():
        data = data.replace(key, f" {token} ")
    data = data.lower().split()
    vocab_to_int = lookup_creation_func(data)
    int_to_vocab = {v:k for k, v in vocab_to_int.items()}
    int_data = [vocab_to_int[word] for word in data]
    with open(filepath, "wb") as f:
        P.dump((int_data, vocab_to_int, int_to_vocab, TOKENS), f)

def loadPickle(filepath: str):
    with open(filepath, "rb") as f:
        return P.load(f)

