import typing as T
import pytest as PT
import json as J

class TestLearn2Spell():

    def test_sanity(self):
        assert True

    def test_spells(self,
                    spells):
        j = spells.getAllSpellIndexes()
        print(J.dumps(j, indent=2, sort_keys=True))
        assert False
