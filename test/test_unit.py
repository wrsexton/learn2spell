import typing as T
import pytest as PT
import requests_mock as RM
import spells as S

from . import fixture as F

class TestLearn2Spell():

    def test_sanity(self):
        assert True

    @PT.mark.parametrize(
        ('success', 'code' ),
        (
            ( True, 200 ),
            ( False, 404 )
        )
    )
    def test_getAllSpellIndexess(self,
                                 requests_mock: RM.Mocker,
                                 spells: S.Spells,
                                 success: bool,
                                 code: int):
        requests_mock.register_uri(RM.GET,
                                   S.SPELLS_INDEX_URL,
                                   status_code=code,
                                   json=F.FAKE_SPELLS_INDEX)
        if(success):
            result = spells.getAllSpellIndexes()
            assert result == [s["index"] for s in F.FAKE_SPELLS_INDEX["results"]]
        else:
            with PT.raises(Exception):
                result = spells.getAllSpellIndexes()

    @PT.mark.parametrize(
        ('success', 'code' ),
        (
            ( True, 200 ),
            ( False, 404 )
        )
    )
    def test_getSpell(self,
                      requests_mock: RM.Mocker,
                      spells: S.Spells,
                      success: bool,
                      code: int):
        index = "fake-spell-1"
        requests_mock.register_uri(RM.GET,
                                   spells.getSpellURL(index),
                                   status_code=code,
                                   json=F.FAKE_SPELL)
        if(success):
            result = spells.getSpell(index)
            assert result == F.FAKE_SPELL
        else:
            with PT.raises(Exception):
                result = spells.getAllSpellIndexes()
