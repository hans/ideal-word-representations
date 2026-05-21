"""Tests for ``src.datasets.french_morph``.

Each test pairs UD features as spaCy's ``fr_core_news_md`` would emit them
with the CSV cell string expected by
``data/french_morphology/suffix_pairs_top10.csv``. Inputs are
authentic — they were captured from spaCy probes on real French sentences —
so passing tests indicate the mapping is wired correctly for the typical
outputs of the tagger.
"""

import pytest

from src.datasets.french_morph import ud_features_to_csv_cell

# ---------------------------------------------------------------------------
# Verbs — indicative present (1/2/3 sg & pl)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "person,number,expected",
    [
        ("1", "Sing", "first-person|indicative|present|singular"),
        ("2", "Sing", "indicative|present|second-person|singular"),
        ("3", "Sing", "indicative|present|singular|third-person"),
        ("1", "Plur", "first-person|indicative|plural|present"),
        ("2", "Plur", "indicative|plural|present|second-person"),
        ("3", "Plur", "indicative|plural|present|third-person"),
    ],
)
def test_indicative_present(person, number, expected):
    morph = {
        "Mood": "Ind",
        "Number": number,
        "Person": person,
        "Tense": "Pres",
        "VerbForm": "Fin",
    }
    assert ud_features_to_csv_cell("VERB", morph) == expected


# ---------------------------------------------------------------------------
# Verbs — indicative imperfect / future / historic (passé simple)
# ---------------------------------------------------------------------------


def test_indicative_imperfect_3sg():
    # parlait → 3sg imperfect indicative
    morph = {
        "Mood": "Ind",
        "Number": "Sing",
        "Person": "3",
        "Tense": "Imp",
        "VerbForm": "Fin",
    }
    assert (
        ud_features_to_csv_cell("VERB", morph)
        == "imperfect|indicative|singular|third-person"
    )


def test_indicative_future_3pl():
    morph = {
        "Mood": "Ind",
        "Number": "Plur",
        "Person": "3",
        "Tense": "Fut",
        "VerbForm": "Fin",
    }
    assert (
        ud_features_to_csv_cell("VERB", morph)
        == "future|indicative|plural|third-person"
    )


def test_indicative_historic_3sg():
    # envoya / fut → passé simple, UD Tense=Past
    morph = {
        "Mood": "Ind",
        "Number": "Sing",
        "Person": "3",
        "Tense": "Past",
        "VerbForm": "Fin",
    }
    assert (
        ud_features_to_csv_cell("AUX", morph)
        == "historic|indicative|past|singular|third-person"
    )


# ---------------------------------------------------------------------------
# Verbs — past participle (masc/fem × sg/pl), plus underspecified variants
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "gender,number,expected",
    [
        ("Masc", "Sing", "masculine|participle|past|singular"),
        ("Masc", "Plur", "masculine|participle|past|plural"),
        ("Fem", "Sing", "feminine|participle|past|singular"),
        ("Fem", "Plur", "feminine|participle|past|plural"),
    ],
)
def test_past_participle_gender_number(gender, number, expected):
    morph = {"VerbForm": "Part", "Tense": "Past", "Gender": gender, "Number": number}
    assert ud_features_to_csv_cell("VERB", morph) == expected


def test_past_participle_underspecified():
    # If Gender/Number are not emitted, fall back to less-specific cells.
    assert (
        ud_features_to_csv_cell("VERB", {"VerbForm": "Part", "Tense": "Past"})
        == "participle|past"
    )
    assert (
        ud_features_to_csv_cell(
            "VERB", {"VerbForm": "Part", "Tense": "Past", "Number": "Plur"}
        )
        == "participle|past|plural"
    )


# ---------------------------------------------------------------------------
# Verbs — gerund / present participle
# ---------------------------------------------------------------------------


def test_present_participle():
    # spaCy may emit VerbForm=Part|Tense=Pres or VerbForm=Ger; both should map
    # to the same gerund cell.
    morph_part = {"VerbForm": "Part", "Tense": "Pres"}
    morph_ger = {"VerbForm": "Ger"}
    assert ud_features_to_csv_cell("VERB", morph_part) == "gerund|participle|present"
    assert ud_features_to_csv_cell("VERB", morph_ger) == "gerund|participle|present"


# ---------------------------------------------------------------------------
# Verbs — infinitive
# ---------------------------------------------------------------------------


def test_infinitive_plain():
    assert ud_features_to_csv_cell("VERB", {"VerbForm": "Inf"}) == "infinitive"


def test_infinitive_with_number():
    # The CSV's odd ``infinitive|plural`` / ``infinitive|singular`` cells:
    # if spaCy ever attaches Number to an infinitive, surface it.
    assert (
        ud_features_to_csv_cell("VERB", {"VerbForm": "Inf", "Number": "Sing"})
        == "infinitive|singular"
    )
    assert (
        ud_features_to_csv_cell("VERB", {"VerbForm": "Inf", "Number": "Plur"})
        == "infinitive|plural"
    )


# ---------------------------------------------------------------------------
# Verbs — imperative
# ---------------------------------------------------------------------------


def test_imperative_2sg():
    morph = {
        "Mood": "Imp",
        "Number": "Sing",
        "Person": "2",
        "Tense": "Pres",
        "VerbForm": "Fin",
    }
    assert ud_features_to_csv_cell("VERB", morph) == "imperative|second-person|singular"


def test_imperative_2pl():
    # Parlez → spaCy emits Mood=Imp|Number=Plur|Person=2|Tense=Pres|VerbForm=Fin
    morph = {
        "Mood": "Imp",
        "Number": "Plur",
        "Person": "2",
        "Tense": "Pres",
        "VerbForm": "Fin",
    }
    assert ud_features_to_csv_cell("VERB", morph) == "imperative|plural|second-person"


# ---------------------------------------------------------------------------
# Verbs — subjunctive
# ---------------------------------------------------------------------------


def test_subjunctive_present_3pl():
    # soient → Mood=Sub|Tense=Pres|Person=3|Number=Plur
    morph = {
        "Mood": "Sub",
        "Number": "Plur",
        "Person": "3",
        "Tense": "Pres",
        "VerbForm": "Fin",
    }
    assert (
        ud_features_to_csv_cell("AUX", morph)
        == "plural|present|subjunctive|third-person"
    )


def test_subjunctive_present_3sg():
    morph = {
        "Mood": "Sub",
        "Number": "Sing",
        "Person": "3",
        "Tense": "Pres",
        "VerbForm": "Fin",
    }
    assert (
        ud_features_to_csv_cell("VERB", morph)
        == "present|singular|subjunctive|third-person"
    )


def test_subjunctive_imperfect_3sg():
    morph = {
        "Mood": "Sub",
        "Number": "Sing",
        "Person": "3",
        "Tense": "Imp",
        "VerbForm": "Fin",
    }
    assert (
        ud_features_to_csv_cell("VERB", morph)
        == "imperfect|singular|subjunctive|third-person"
    )


# ---------------------------------------------------------------------------
# Verbs — conditional
# ---------------------------------------------------------------------------


def test_conditional_3sg():
    morph = {
        "Mood": "Cnd",
        "Number": "Sing",
        "Person": "3",
        "Tense": "Pres",
        "VerbForm": "Fin",
    }
    assert (
        ud_features_to_csv_cell("VERB", morph)
        == "conditional|present|singular|third-person"
    )


# ---------------------------------------------------------------------------
# Nouns
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "gender,number,expected",
    [
        ("Masc", "Sing", "masculine|noun|singular"),
        ("Masc", "Plur", "masculine|noun|plural"),
        ("Fem", "Sing", "feminine|noun|singular"),
        ("Fem", "Plur", "feminine|noun|plural"),
    ],
)
def test_noun_gender_number(gender, number, expected):
    morph = {"Gender": gender, "Number": number}
    assert ud_features_to_csv_cell("NOUN", morph) == expected


def test_noun_underspecified():
    assert ud_features_to_csv_cell("NOUN", {}) == "noun"
    assert ud_features_to_csv_cell("NOUN", {"Number": "Plur"}) == "noun|plural"
    assert ud_features_to_csv_cell("NOUN", {"Gender": "Fem"}) == "feminine|noun"


# ---------------------------------------------------------------------------
# Adverbs, adjectives, and POSes with no CSV cell
# ---------------------------------------------------------------------------


def test_adverb():
    assert ud_features_to_csv_cell("ADV", {}) == "adverb"


def test_adjective_returns_none():
    # The CSV has no `adjective` cell, by design. Adjectives still appear in
    # the corpus; we tag them with csv_cell=None so the analogy task simply
    # ignores them.
    assert ud_features_to_csv_cell("ADJ", {"Gender": "Masc", "Number": "Plur"}) is None


@pytest.mark.parametrize(
    "upos",
    [
        "DET",
        "PRON",
        "ADP",
        "CCONJ",
        "SCONJ",
        "PUNCT",
        "NUM",
        "PART",
        "INTJ",
        "PROPN",
        "SYM",
        "X",
        "SPACE",
    ],
)
def test_other_pos_returns_none(upos):
    assert ud_features_to_csv_cell(upos, {}) is None


def test_none_inputs():
    assert ud_features_to_csv_cell(None, None) is None
    assert ud_features_to_csv_cell("VERB", None) is None
