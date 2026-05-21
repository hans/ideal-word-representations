"""Add ``word_morph_detail`` (per-token morphological tags) to an MLS French
preprocessed dataset.

For each utterance, this script runs spaCy ``fr_core_news_md`` over the
``text`` field, aligns spaCy tokens back to MFA word tokens by character offset
(handling French elision/clitic splits like ``l'heure`` → ``l'`` + ``heure``
and ``dit-il`` → ``dit`` + ``-il``), and writes a per-word morphological dict
parallel to ``word_phonemic_detail``.

The augmented dataset replaces the input dataset in place (atomic swap via a
temporary directory). A JSON report with token / UPOS / alignment counts is
also emitted.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

import datasets
import spacy
from spacy.language import Language
from spacy.tokens import Doc

from src.datasets.french_morph import ud_features_to_csv_cell

L = logging.getLogger(__name__)


# spaCy tokens whose surface form is one of these elided clitics are NOT the
# head of a multi-token MFA word (``l'``, ``j'``, ``qu'`` etc.) and should be
# skipped when picking which spaCy token represents an MFA word.
ELIDED_CLITICS_PREFIXES = (
    "l'",
    "j'",
    "n'",
    "m'",
    "t'",
    "s'",
    "d'",
    "c'",
    "qu'",
    "lorsqu'",
    "puisqu'",
    "jusqu'",
    "quoiqu'",
)


def _is_elided_clitic(text: str) -> bool:
    """True for tokens like ``l'``, ``qu'`` (lowercased to match)."""
    lower = text.lower()
    return any(lower == prefix for prefix in ELIDED_CLITICS_PREFIXES)


def _strip_leading_hyphen(text: str) -> str:
    return text.lstrip("-")


def _locate_mfa_word_spans(text: str, words: list[str]) -> list[tuple[int, int] | None]:
    """Find character offsets in ``text`` for each non-empty MFA word.

    Walks ``text`` left-to-right with a cursor; for each word, advances past
    any whitespace / hyphens, then verifies that the next ``len(word)``
    characters match. NFC-normalizes both sides defensively.

    Empty MFA tokens (silence markers) get ``None``.

    Returns ``None`` for a word if no match could be found at the expected
    position — alignment is missing for that token and downstream callers
    will mark ``alignment_confidence == "missing"``.
    """
    text_nfc = unicodedata.normalize("NFC", text)
    spans: list[tuple[int, int] | None] = []
    cursor = 0
    for word in words:
        if word == "":
            spans.append(None)
            continue
        word_nfc = unicodedata.normalize("NFC", word)
        # Look from the cursor forward for the next occurrence of word.
        idx = text_nfc.find(word_nfc, cursor)
        if idx < 0:
            # Try a case-insensitive search as a fallback.
            idx = text_nfc.lower().find(word_nfc.lower(), cursor)
        if idx < 0:
            L.warning(
                "Could not locate MFA word %r in text from offset %d", word, cursor
            )
            spans.append(None)
            continue
        end = idx + len(word_nfc)
        spans.append((idx, end))
        cursor = end
    return spans


def _pick_spacy_token(
    doc: Doc, span: tuple[int, int], mfa_word: str
) -> tuple[Any, str] | None:
    """Return ``(token, alignment_confidence)`` for the spaCy token that best
    represents an MFA word at ``span = (char_start, char_end)``.

    Strategy:
    1. Try ``Doc.char_span(...)`` with ``alignment_mode='expand'`` to get
       overlapping spaCy tokens.
    2. From those tokens, drop any that are elided clitics (``l'``, ``qu'``,
       …) when there's a non-elided alternative.
    3. Prefer a token whose stripped surface matches the MFA word exactly
       (``"exact"`` confidence); otherwise pick the rightmost remaining
       candidate (``"fuzzy"``).

    Returns ``None`` if no overlap exists at all.
    """
    char_start, char_end = span
    spacy_span = doc.char_span(char_start, char_end, alignment_mode="expand")
    if spacy_span is None or len(spacy_span) == 0:
        return None

    candidates = list(spacy_span)
    non_elided = [t for t in candidates if not _is_elided_clitic(t.text)]
    if non_elided:
        candidates = non_elided

    mfa_lower = mfa_word.lower()
    for t in candidates:
        if _strip_leading_hyphen(t.text).lower() == mfa_lower:
            return t, "exact"

    # Fall back to the rightmost candidate (typically the lexical head in
    # French elision: ``l'heure`` → ``heure``).
    return candidates[-1], "fuzzy"


def _build_word_morph_detail(
    item: dict[str, Any], doc: Doc
) -> list[dict[str, Any] | None]:
    """Construct the ``word_morph_detail`` list for one dataset item."""
    text = item["text"]
    words: list[str] = item["word_detail"]["utterance"]
    spans = _locate_mfa_word_spans(text, words)

    out: list[dict[str, Any] | None] = []
    for word, span in zip(words, spans):
        if word == "":
            # Silence markers carry no morphology.
            out.append(None)
            continue
        if span is None:
            out.append(
                {
                    "upos": None,
                    "morph": {},
                    "csv_cell": None,
                    "lemma": None,
                    "alignment_confidence": "missing",
                }
            )
            continue
        picked = _pick_spacy_token(doc, span, word)
        if picked is None:
            out.append(
                {
                    "upos": None,
                    "morph": {},
                    "csv_cell": None,
                    "lemma": None,
                    "alignment_confidence": "missing",
                }
            )
            continue
        tok, confidence = picked
        morph = dict(tok.morph.to_dict())
        out.append(
            {
                "upos": tok.pos_,
                "morph": morph,
                "csv_cell": ud_features_to_csv_cell(tok.pos_, morph),
                "lemma": tok.lemma_,
                "alignment_confidence": confidence,
            }
        )
    return out


def _make_mapper(nlp: Language):
    """HuggingFace ``Dataset.map`` callable that adds the morph column."""

    def _map(item):
        doc = nlp(item["text"])
        item["word_morph_detail"] = _build_word_morph_detail(item, doc)
        return item

    return _map


def _summarize(ds: datasets.Dataset) -> dict[str, Any]:
    """Compute token-level counts for the report."""
    n_total = 0
    n_silence = 0
    n_missing = 0
    upos_counts: Counter[str] = Counter()
    confidence_counts: Counter[str] = Counter()
    csv_cell_counts: Counter[str] = Counter()
    for item in ds:
        for entry in item["word_morph_detail"]:
            n_total += 1
            if entry is None:
                n_silence += 1
                continue
            conf = entry["alignment_confidence"]
            confidence_counts[conf] += 1
            if conf == "missing":
                n_missing += 1
                continue
            upos = entry["upos"] or "UNKNOWN"
            upos_counts[upos] += 1
            if entry["csv_cell"] is not None:
                csv_cell_counts[entry["csv_cell"]] += 1
    return {
        "tokens_total": n_total,
        "tokens_silence": n_silence,
        "tokens_aligned": n_total - n_silence - n_missing,
        "tokens_missing_alignment": n_missing,
        "by_alignment_confidence": dict(confidence_counts),
        "by_upos": dict(sorted(upos_counts.items(), key=lambda kv: (-kv[1], kv[0]))),
        "top_csv_cells": dict(csv_cell_counts.most_common(30)),
        "n_distinct_csv_cells": len(csv_cell_counts),
    }


def _atomic_save(ds: datasets.Dataset, dataset_path: Path) -> None:
    """Save ``ds`` to ``dataset_path``, replacing any existing dataset there.

    Writes to a sibling ``.tmp`` directory first, then swaps. Cleans up the
    temp directory on failure.
    """
    tmp_path = dataset_path.with_name(dataset_path.name + ".tmp")
    if tmp_path.exists():
        shutil.rmtree(tmp_path)
    try:
        ds.save_to_disk(str(tmp_path))
    except Exception:
        if tmp_path.exists():
            shutil.rmtree(tmp_path)
        raise
    backup_path = dataset_path.with_name(dataset_path.name + ".prev")
    if backup_path.exists():
        shutil.rmtree(backup_path)
    if dataset_path.exists():
        dataset_path.rename(backup_path)
    tmp_path.rename(dataset_path)
    if backup_path.exists():
        shutil.rmtree(backup_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        required=True,
        type=Path,
        help="Path to preprocessed dataset (mutated in place).",
    )
    parser.add_argument(
        "--report", required=True, type=Path, help="Path to write JSON summary report."
    )
    parser.add_argument(
        "--model", default="fr_core_news_md", help="spaCy model to load."
    )
    parser.add_argument(
        "--num-proc",
        type=int,
        default=1,
        help="HuggingFace map num_proc (spaCy is single-threaded; "
        "multiprocess only helps if you have many cores).",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    L.info("Loading spaCy model: %s", args.model)
    nlp = spacy.load(args.model, disable=["parser", "ner"])

    L.info("Loading dataset from %s", args.dataset)
    ds = datasets.load_from_disk(str(args.dataset))

    L.info("Tagging %d examples", len(ds))
    ds = ds.map(_make_mapper(nlp), num_proc=args.num_proc, desc="spaCy tagging")

    L.info("Saving augmented dataset to %s (atomic swap)", args.dataset)
    _atomic_save(ds, args.dataset)

    summary = _summarize(ds)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    L.info("Report written to %s", args.report)
    L.info(
        "Summary: %d aligned / %d total tokens; %d missing",
        summary["tokens_aligned"],
        summary["tokens_total"],
        summary["tokens_missing_alignment"],
    )


if __name__ == "__main__":
    main()
