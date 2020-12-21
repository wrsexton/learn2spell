import typing as T
import pytest as PT
import requests_mock as RM

import spells as S
import learning as L

from . import fixture as F

class TestLearn2Spell():

    def test_sanity(self):
        L.main()
        assert False

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
            assert result == F.FAKE_SPELLS_INDEX_NAMES
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

    def test_getAllSpells(self,
                          requests_mock: RM.Mocker,
                          spells: S.Spells):
        code = 200
        indexes = F.FAKE_SPELLS_INDEX_NAMES
        allSpells = []
        for index in indexes:
            requests_mock.register_uri(RM.GET,
                                       spells.getSpellURL(index),
                                       status_code=code,
                                       json=F.FAKE_SPELL)
            allSpells.append(F.FAKE_SPELL)
        requests_mock.register_uri(RM.GET,
                                   S.SPELLS_INDEX_URL,
                                   status_code=code,
                                   json=F.FAKE_SPELLS_INDEX)
        result = spells.getAllSpells()
        assert result == allSpells
