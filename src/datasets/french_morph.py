"""UD → CSV-cell mapping for French morphology.

The French inflection CSV (`data/french_morphology/suffix_pairs_top10.csv`)
labels each (base_orth, derived_orth) pair with morphological cells using a
pipe-joined, alphabetically-sorted feature string, e.g.
``"indicative|present|singular|third-person"``.

Per audio token, we need a single best-guess cell to disambiguate
morphologically-ambiguous orthographic forms (e.g., *parle* → 1sg ind pres vs.
3sg ind pres vs. 2sg imp vs. 3sg subj pres). spaCy's ``fr_core_news_md`` emits
UD-style features (Mood/Tense/Number/Person/Gender/VerbForm) per token; this
module converts those features to the CSV's cell format.

Coverage notes for the ~90 distinct cells observed in the CSV:

- ``adverb``: UPOS=ADV.
- ``noun``, ``feminine|noun``, ``masculine|noun|plural`` etc.: UPOS=NOUN with
  optional Gender / Number.
- ``infinitive`` (and bizarre ``infinitive|singular``/``infinitive|plural``):
  VerbForm=Inf with optional Number.
- ``participle|past|...``, ``feminine|participle|past|singular`` etc.:
  VerbForm=Part with Tense=Past, optional Gender/Number.
- ``gerund|participle|present`` (+ optional Number): VerbForm=Part with
  Tense=Pres, or VerbForm=Ger.
- ``imperative|...``: Mood=Imp.
- ``conditional|...|present``: Mood=Cnd. (fr_core_news_md often mistags
  conditional as ``Tense=Imp`` — we trust whatever spaCy gives us.)
- ``...|subjunctive`` (present or imperfect): Mood=Sub.
- ``indicative|present|...``, ``imperfect|indicative|...``,
  ``future|indicative|...``, ``historic|indicative|past|...``: Mood=Ind with
  Tense=Pres/Imp/Fut/Past respectively.

Cells that exist in the CSV but cannot be reached from spaCy's outputs
(notably the third-person imperatives like ``imperative|singular|third-person``,
which are surface subjunctives) are intentionally unreachable — see the
"Out of scope" section of issue #2.
"""

from __future__ import annotations

from typing import Mapping

__all__ = ["ud_features_to_csv_cell"]


_PERSON_TAG = {"1": "first-person", "2": "second-person", "3": "third-person"}
_NUMBER_TAG = {"Sing": "singular", "Plur": "plural"}
_GENDER_TAG = {"Masc": "masculine", "Fem": "feminine"}


def _join(parts: list[str]) -> str:
    return "|".join(sorted(parts))


def ud_features_to_csv_cell(
    upos: str | None,
    morph: Mapping[str, str] | None,
) -> str | None:
    """Convert a spaCy / UD ``(upos, morph)`` pair to a CSV cell string.

    Returns ``None`` for tokens whose POS has no corresponding CSV cell
    (function words, punctuation, adjectives, etc.) — downstream analogy code
    treats these as untagged.

    The returned cell's parts are alphabetically sorted to match the CSV's
    convention.
    """
    if upos is None:
        return None
    morph = morph or {}

    if upos == "ADV":
        return "adverb"

    if upos == "NOUN":
        parts = ["noun"]
        if (g := _GENDER_TAG.get(morph.get("Gender", ""))) is not None:
            parts.append(g)
        if (n := _NUMBER_TAG.get(morph.get("Number", ""))) is not None:
            parts.append(n)
        return _join(parts)

    if upos not in ("VERB", "AUX"):
        return None

    verbform = morph.get("VerbForm")
    mood = morph.get("Mood")
    tense = morph.get("Tense")
    number = _NUMBER_TAG.get(morph.get("Number", ""))
    person = _PERSON_TAG.get(morph.get("Person", ""))
    gender = _GENDER_TAG.get(morph.get("Gender", ""))

    if verbform == "Inf":
        parts = ["infinitive"]
        if number is not None:
            parts.append(number)
        return _join(parts)

    if verbform == "Part":
        if tense == "Past":
            parts = ["participle", "past"]
            if gender is not None:
                parts.append(gender)
            if number is not None:
                parts.append(number)
            return _join(parts)
        if tense == "Pres":
            parts = ["gerund", "participle", "present"]
            if number is not None:
                parts.append(number)
            return _join(parts)

    if verbform == "Ger":
        parts = ["gerund", "participle", "present"]
        if number is not None:
            parts.append(number)
        return _join(parts)

    if mood == "Imp":
        parts = ["imperative"]
        if person is not None:
            parts.append(person)
        if number is not None:
            parts.append(number)
        return _join(parts)

    if mood == "Cnd":
        parts = ["conditional", "present"]
        if person is not None:
            parts.append(person)
        if number is not None:
            parts.append(number)
        return _join(parts)

    if mood == "Sub":
        if tense == "Imp":
            parts = ["imperfect", "subjunctive"]
        else:
            parts = ["present", "subjunctive"]
        if person is not None:
            parts.append(person)
        if number is not None:
            parts.append(number)
        return _join(parts)

    if mood == "Ind":
        if tense == "Pres":
            parts = ["indicative", "present"]
        elif tense == "Imp":
            parts = ["imperfect", "indicative"]
        elif tense == "Fut":
            parts = ["future", "indicative"]
        elif tense == "Past":
            parts = ["historic", "indicative", "past"]
        else:
            return None
        if person is not None:
            parts.append(person)
        if number is not None:
            parts.append(number)
        return _join(parts)

    return None
