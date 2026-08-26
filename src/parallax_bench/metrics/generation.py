"""Generation metrics.

Two families:

- **Mechanical** (offline, deterministic): citation accuracy against the
  document-level ground truth, and response language correctness via a
  stopword/diacritics profile detector.  These need no models and no network.
- **LLM-judge** (faithfulness, answer relevancy): pluggable through the
  ``Judge`` protocol.  The judge model must differ from the generator model
  (self-preference bias) — enforce that in run configuration, not here.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

# -- citation accuracy ------------------------------------------------------


@dataclass(frozen=True)
class CitationScore:
    n_cited: int
    n_valid: int        # cited doc actually appears in the retrieved set
    n_relevant: int     # cited doc is relevant per qrels
    valid_rate: float
    relevant_rate: float


def citation_accuracy(
    cited_doc_ids: Sequence[str],
    retrieved_doc_ids: Sequence[str],
    rels: Mapping[str, int],
) -> CitationScore:
    retrieved = set(retrieved_doc_ids)
    n_cited = len(cited_doc_ids)
    n_valid = sum(1 for d in cited_doc_ids if d in retrieved)
    n_relevant = sum(1 for d in cited_doc_ids if rels.get(d, 0) > 0)
    return CitationScore(
        n_cited=n_cited,
        n_valid=n_valid,
        n_relevant=n_relevant,
        valid_rate=n_valid / n_cited if n_cited else 0.0,
        relevant_rate=n_relevant / n_cited if n_cited else 0.0,
    )


# -- response language correctness ------------------------------------------

# Minimal offline language identification for the benchmark languages.
# Deliberately simple and transparent: top function words + strongly
# language-specific characters.  Good enough for "did the model answer in the
# language of the question", which involves whole paragraphs, not tweets.

_STOPWORDS: dict[str, set[str]] = {
    "cs": {"a", "se", "na", "je", "že", "v", "s", "do", "pro", "podle", "nebo", "které", "být",
           "jsou", "za", "od", "této", "však", "musí", "při", "pokud"},
    "sk": {"a", "sa", "na", "je", "že", "v", "s", "do", "pre", "podľa", "alebo", "ktoré", "byť",
           "sú", "za", "od", "tejto", "však", "musí", "pri", "ak"},
    "en": {"the", "of", "and", "to", "in", "is", "that", "for", "on", "with", "as", "by", "are",
           "be", "this", "shall", "or", "which", "not", "from"},
    "de": {"der", "die", "das", "und", "in", "von", "zu", "den", "mit", "für", "auf", "ist",
           "des", "im", "nicht", "eine", "werden", "oder", "dem", "nach"},
    "pl": {"i", "w", "na", "się", "z", "do", "nie", "że", "jest", "oraz", "dla", "przez", "które",
           "być", "są", "od", "lub", "tej", "jednak", "przy"},
    "fr": {"le", "la", "les", "de", "des", "et", "à", "en", "un", "une", "du", "est", "pour",
           "que", "qui", "dans", "par", "sur", "ne", "pas"},
    "es": {"el", "la", "los", "las", "de", "y", "a", "en", "un", "una", "del", "es", "para",
           "que", "por", "con", "no", "se", "su", "como"},
    "it": {"il", "la", "i", "le", "di", "e", "a", "in", "un", "una", "del", "è", "per", "che",
           "non", "si", "con", "dei", "delle", "sono"},
    "hu": {"a", "az", "és", "hogy", "nem", "is", "egy", "van", "meg", "de", "el", "kell", "vagy",
           "már", "csak", "ha", "való", "mint", "által", "szerint"},
    "nl": {"de", "het", "een", "van", "en", "in", "is", "dat", "op", "te", "voor", "met", "zijn",
           "niet", "aan", "of", "die", "worden", "door", "bij"},
}

_CHAR_MARKERS: dict[str, str] = {
    "cs": "ěřůťďň",
    "sk": "ľĺŕäô",
    "pl": "łńśźż",
    "de": "ß",
    "hu": "őű",
    "fr": "êâîôûëïç",
    "es": "ñ¿¡",
}

_WORD_RE = re.compile(r"\w+", re.UNICODE)


def detect_language(text: str, candidates: Sequence[str] | None = None) -> str | None:
    """Best-guess language among ``candidates`` (default: all profiles).

    Returns None when the text carries no signal (empty or too short).
    """
    langs = list(candidates) if candidates else list(_STOPWORDS)
    words = [w.lower() for w in _WORD_RE.findall(text)]
    if len(words) < 3:
        return None
    lower = text.lower()
    scores: dict[str, float] = {}
    for lang in langs:
        sw = _STOPWORDS.get(lang, set())
        hit = sum(1 for w in words if w in sw)
        score = hit / len(words)
        for ch in _CHAR_MARKERS.get(lang, ""):
            if ch in lower:
                score += 0.05
        scores[lang] = score
    best = max(scores, key=lambda code: scores[code])
    return best if scores[best] > 0 else None


def language_correct(answer: str, expected_lang: str,
                     candidates: Sequence[str] | None = None) -> bool | None:
    """Did the system answer in the language of the question?

    Returns None for undecidable answers (empty/too short) so callers can
    report them separately rather than counting them as wrong.
    """
    detected = detect_language(answer, candidates)
    if detected is None:
        return None
    return detected == expected_lang


# -- LLM-judge metrics -------------------------------------------------------


class Judge(Protocol):
    """A reference-free judge for faithfulness / answer relevancy.

    Implementations call an LLM that is a *different* model from the one that
    generated the answers.  Kept out of the core dependency set on purpose.
    """

    def faithfulness(self, question: str, answer: str, chunks: Sequence[str]) -> float:
        """0..1 — is the answer supported by the retrieved chunks?"""
        ...

    def answer_relevancy(self, question: str, answer: str) -> float:
        """0..1 — does the answer address the question?"""
        ...
