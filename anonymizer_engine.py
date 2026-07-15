"""Orchestrates entity detection -> stable token assignment -> replacement.

`AnonymizerSession` is the core object: it holds one Presidio AnalyzerEngine
(cached per language, since loading NLP models is expensive) plus a running
`value -> token` map. Calling `analyze_paragraph()` / `anonymize_plain_text()`
repeatedly on the same session guarantees that a repeated entity value (e.g.
the same person's name appearing 10 times) always maps to the same token
(e.g. `[ФИО_1]`), and accumulates all mapping rows for persistence in SQLite.
"""

from __future__ import annotations

from collections import defaultdict
from functools import lru_cache
from typing import Dict, List, Tuple

from presidio_analyzer import AnalyzerEngine, RecognizerResult

from entities import ENTITY_LABELS, REGEX_ENTITY_TYPES
from recognizers import build_analyzer_engine

LANGUAGE_LABELS = {"ru": "Русский", "kk": "Қазақша"}


@lru_cache(maxsize=4)
def get_analyzer_engine(language: str) -> AnalyzerEngine:
    """Build (once per process) and cache the AnalyzerEngine for a language.

    Model loading (spaCy / transformers) can take from a few seconds up to
    ~1 minute on first use; subsequent calls reuse the cached engine.
    """
    return build_analyzer_engine(language)


def _resolve_overlaps(results: List[RecognizerResult]) -> List[RecognizerResult]:
    """Greedily pick a non-overlapping subset of detections.

    Regex-based entities (IIN/BIN, IBAN, phone, email, document id) always
    win over NER-based entities (PERSON/ORGANIZATION/LOCATION) when spans
    overlap, since they are deterministic and effectively 100% precise.
    Within the same priority tier, higher score and longer spans win.
    """

    def sort_key(r: RecognizerResult):
        priority = 0 if r.entity_type in REGEX_ENTITY_TYPES else 1
        return (priority, -r.score, -(r.end - r.start))

    ordered = sorted(results, key=sort_key)
    selected: List[RecognizerResult] = []
    covered: List[Tuple[int, int]] = []
    for r in ordered:
        if any(not (r.end <= s or r.start >= e) for s, e in covered):
            continue
        covered.append((r.start, r.end))
        selected.append(r)
    return sorted(selected, key=lambda r: r.start)


class AnonymizerSession:
    """Stateful helper accumulating a consistent value->token map.

    Use one instance per document (or per raw-text submission). After
    processing, `mapping_records` holds every (token, original_value,
    entity_type) triple ready to be persisted via `database.save_mapping`.
    """

    def __init__(self, language: str):
        if language not in ("ru", "kk"):
            raise ValueError(f"Unsupported language: {language!r}")
        self.language = language
        self.engine = get_analyzer_engine(language)
        self._value_to_token: Dict[Tuple[str, str], str] = {}
        self._counters: Dict[str, int] = defaultdict(int)
        self.mapping_records: List[Tuple[str, str, str]] = []

    def _token_for(self, entity_type: str, value: str) -> str:
        key = (entity_type, value.strip())
        token = self._value_to_token.get(key)
        if token is None:
            self._counters[entity_type] += 1
            label = ENTITY_LABELS.get(entity_type, entity_type)
            token = f"[{label}_{self._counters[entity_type]}]"
            self._value_to_token[key] = token
            self.mapping_records.append((token, value, entity_type))
        return token

    def analyze_paragraph(self, text: str) -> List[Tuple[int, int, str]]:
        """Detect entities in `text` and return (start, end, token) matches
        (offsets relative to `text`, sorted ascending, non-overlapping).
        Does not modify `text` itself — used for docx run-level replacement.
        """
        if not text or not text.strip():
            return []
        try:
            raw_results = self.engine.analyze(text=text, language=self.language)
        except Exception as exc:  # model inference failure shouldn't crash the whole doc
            raise RuntimeError(f"Ошибка анализа текста NLP-движком: {exc}") from exc

        results = _resolve_overlaps(raw_results)
        matches = []
        for r in results:
            value = text[r.start : r.end]
            token = self._token_for(r.entity_type, value)
            matches.append((r.start, r.end, token))
        return sorted(matches, key=lambda m: m[0])

    def anonymize_plain_text(self, text: str) -> str:
        """Analyze + replace in one shot for a plain string (no docx runs)."""
        matches = self.analyze_paragraph(text)
        if not matches:
            return text
        out = []
        cursor = 0
        for start, end, token in matches:
            out.append(text[cursor:start])
            out.append(token)
            cursor = end
        out.append(text[cursor:])
        return "".join(out)


def restore_plain_text(text: str, mapping: Dict[str, str]) -> str:
    """Replace every known token in `text` with its original value."""
    restored = text
    for token, value in mapping.items():
        restored = restored.replace(token, value)
    return restored
