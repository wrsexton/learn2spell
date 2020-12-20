"""A collection of static fixtures"""

FAKE_SPELLS_INDEX = {
    "count": 3,
    "results": [
        {
            "index": "fake-spell-1",
            "name": "Fake Spell 1",
            "url": "/api/spells/fake-spell-1"
        },
        {
            "index": "fake-spell-2",
            "name": "Fake Spell 2",
            "url": "/api/spells/fake-spell-2"
        },
        {
            "index": "fake-spell-3",
            "name": "Fake Spell 3",
            "url": "/api/spells/fake-spell-3"
        }
    ]
}

FAKE_SPELLS_INDEX_NAMES = [s["index"] for s in FAKE_SPELLS_INDEX["results"]]

FAKE_SPELL = {
    "index": "fake-spell-1",
    "name": "Fake Spell 1",
    "desc": [
    "Ability damage ability score animal type armor class base land speed battle grid calling subschool change shape cleric concentrate on a spell earth domain environment flank halfling domain healing subschool material plane ranged weapon sacred bonus transmutation."
    ],
    "higher_level": [
        "When you cast this spell using a spell slot of 3rd level or higher, the damage (both initial and later) increases by 1d4 for each slot level above 2nd."
    ],
    "range": "90 feet",
    "components": [
        "V",
        "S",
        "M"
    ],
    "material": "Powdered rhubarb leaf and an adder's stomach.",
    "ritual": False,
    "duration": "Instantaneous",
    "concentration": False,
    "casting_time": "1 action",
    "level": 2,
    "attack_type": "ranged",
    "damage": {
        "damage_type": {
            "index": "acid",
            "name": "Acid",
            "url": "/api/damage-types/acid"
        },
        "damage_at_slot_level": {
            "2": "4d4",
            "3": "5d4",
            "4": "6d4",
            "5": "7d4",
            "6": "8d4",
            "7": "9d4",
            "8": "10d4",
            "9": "11d4"
        }
    },
    "school": {
        "index": "evocation",
        "name": "Evocation",
        "url": "/api/magic-schools/evocation"
    },
    "classes": [
        {
            "index": "wizard",
            "name": "Wizard",
            "url": "/api/classes/wizard"
        }
    ],
    "subclasses": [
        {
            "index": "lore",
            "name": "Lore",
            "url": "/api/subclasses/lore"
        },
        {
            "index": "land",
            "name": "Land",
            "url": "/api/subclasses/land"
        }
    ],
    "url": "/api/spells/fake-spell-1"
}
