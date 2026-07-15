"""Builds the Presidio AnalyzerEngine used for entity detection.

Two families of recognizers are combined:

1. Deterministic regex `PatternRecognizer`s for RU/KZ-specific identifiers
   (ИИН/БИН, KZ IBAN, phone numbers, email, document/passport numbers).
   These are close to 100% precise and are always preferred over NER when
   spans overlap (see `anonymizer_engine._resolve_overlaps`).

2. NER-based recognizers for PERSON / ORGANIZATION / LOCATION:
   - Russian: spaCy's `ru_core_news_lg` pipeline, used via Presidio's
     built-in `SpacyRecognizer` (auto-registered by
     `RecognizerRegistry.load_predefined_recognizers`).
   - Kazakh: spaCy has no native Kazakh NER model, so a custom
     `TransformersNerRecognizer` wraps a HuggingFace token-classification
     pipeline instead. It is a proper `presidio_analyzer.EntityRecognizer`
     subclass, registered into the same `RecognizerRegistry`, so it
     participates in `AnalyzerEngine.analyze()` exactly like a built-in
     recognizer.

Set `RU_NER_BACKEND = "transformer"` to use a HuggingFace model
(`RU_TRANSFORMER_MODEL`) for Russian instead of spaCy.
"""

from __future__ import annotations

import logging
import re
from typing import List, Optional

import spacy
from presidio_analyzer import (
    EntityRecognizer,
    Pattern,
    PatternRecognizer,
    RecognizerRegistry,
    RecognizerResult,
)
from presidio_analyzer import AnalyzerEngine
from presidio_analyzer.nlp_engine import NlpArtifacts, NlpEngineProvider, SpacyNlpEngine

from entities import ENTITY_LABELS

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------

RU_SPACY_MODEL = "ru_core_news_lg"
RU_TRANSFORMER_MODEL = "Burenko/rubert-base-ner"
# Fine-tuned on KazNERD (LREC 2022, https://aclanthology.org/2022.lrec-1.44),
# the reference Kazakh NER dataset from ISSAI/Nazarbayev University. The
# previously configured "Davron/xlm-roberta-large-ner-kazakh" no longer
# resolves on the HF Hub (404/private) - this is a maintained replacement.
KK_TRANSFORMER_MODEL = "yeshpanovrustem/xlm-roberta-large-kaznerd"

# "spacy" (default, uses ru_core_news_lg) or "transformer" (uses RU_TRANSFORMER_MODEL)
RU_NER_BACKEND = "spacy"

# Below this confidence, a transformer NER hit is discarded. The small
# multilingual/Kazakh models are prone to tagging generic Cyrillic words
# (document boilerplate like "ПРИКАЗ", dates, common nouns) as entities with
# low confidence; this threshold trades a bit of recall for much less noise.
MIN_TRANSFORMER_NER_SCORE = 0.6


# ---------------------------------------------------------------------------
# Regex patterns (RU/KZ specific, high-precision entities)
# ---------------------------------------------------------------------------

IIN_BIN_PATTERN = Pattern(name="iin_bin_pattern", regex=r"\b\d{12}\b", score=0.9)
KZ_IBAN_PATTERN = Pattern(name="kz_iban_pattern", regex=r"\bKZ[0-9A-Z]{18}\b", score=0.95)
PHONE_PATTERN = Pattern(
    name="phone_pattern",
    regex=r"(?:\+7|8)[\s\-]?\(?\d{3}\)?[\s\-]?\d{3}[\s\-]?\d{2}[\s\-]?\d{2}\b",
    score=0.85,
)
EMAIL_PATTERN = Pattern(
    name="email_pattern",
    regex=r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
    score=0.9,
)
# Passport / ID document numbers: e.g. "N01234567", "№ 034521678"
DOCUMENT_ID_PATTERN = Pattern(
    name="document_id_pattern",
    regex=r"\b(?:№\s?)?[A-ZА-Я]{1,2}\s?\d{6,9}\b",
    score=0.55,
)

# Signature-line names in RU/KZ official documents, e.g. "Төлеміс А.Ә.",
# "Иванов И.И.", or the reverse order "Д.Х. Қалекес", "И.И. Иванов". NER
# models (especially the small Kazakh transformer) routinely miss these
# abbreviated-initials formats, so a deterministic pattern catches them.
_KZ_CYR_UPPER = "А-ЯЁӘҒҚҢӨҰҮІҺ"
_KZ_CYR_LOWER = "а-яёәғқңөұүіһ"
SIGNATURE_NAME_SURNAME_FIRST_PATTERN = Pattern(
    name="signature_name_surname_first_pattern",
    regex=rf"\b[{_KZ_CYR_UPPER}][{_KZ_CYR_LOWER}]+\s+[{_KZ_CYR_UPPER}]\.\s?[{_KZ_CYR_UPPER}]\.",
    # Kept just below the max so it reliably wins ties against fragmented
    # NER hits over the same span (NER models routinely split these
    # abbreviated-initials names into several sub-word PERSON pieces with
    # scores up to ~0.999; overlap resolution picks the highest score
    # within the same entity type, so this must outscore that).
    score=0.99,
)
SIGNATURE_NAME_INITIALS_FIRST_PATTERN = Pattern(
    name="signature_name_initials_first_pattern",
    regex=rf"\b[{_KZ_CYR_UPPER}]\.\s?[{_KZ_CYR_UPPER}]\.\s+[{_KZ_CYR_UPPER}][{_KZ_CYR_LOWER}]+\b",
    score=0.99,
)


def build_regex_recognizers(language: str) -> List[PatternRecognizer]:
    """Return fresh PatternRecognizer instances scoped to `language`."""
    return [
        PatternRecognizer(
            supported_entity="IIN_BIN",
            patterns=[IIN_BIN_PATTERN],
            supported_language=language,
            context=["иин", "бин", "иин/бин", "жеке сәйкестендіру нөмірі"],
        ),
        PatternRecognizer(
            supported_entity="KZ_IBAN",
            patterns=[KZ_IBAN_PATTERN],
            supported_language=language,
            context=["iban", "счет", "счёт", "шот"],
        ),
        PatternRecognizer(
            supported_entity="PHONE_NUMBER",
            patterns=[PHONE_PATTERN],
            supported_language=language,
        ),
        PatternRecognizer(
            supported_entity="EMAIL_ADDRESS",
            patterns=[EMAIL_PATTERN],
            supported_language=language,
        ),
        PatternRecognizer(
            supported_entity="DOCUMENT_ID",
            patterns=[DOCUMENT_ID_PATTERN],
            supported_language=language,
            context=["паспорт", "удостоверение", "№", "куәлік"],
        ),
        PatternRecognizer(
            supported_entity="PERSON",
            patterns=[
                SIGNATURE_NAME_SURNAME_FIRST_PATTERN,
                SIGNATURE_NAME_INITIALS_FIRST_PATTERN,
            ],
            supported_language=language,
            context=["директор", "подпись", "таныстым", "қолы", "исполнитель"],
            # Presidio's PatternRecognizer defaults to
            # re.IGNORECASE | re.DOTALL | re.MULTILINE. IGNORECASE would
            # defeat the whole point of this pattern (it relies on
            # Uppercase/lowercase to shape-match "Surname I.I." vs.
            # arbitrary lowercase words), so it's turned off here.
            global_regex_flags=re.DOTALL | re.MULTILINE,
        ),
    ]


# ---------------------------------------------------------------------------
# Custom NER recognizer backed by a HuggingFace transformers pipeline
# ---------------------------------------------------------------------------


def _merge_adjacent_same_type(
    results: List[RecognizerResult], text: str
) -> List[RecognizerResult]:
    """Merge consecutive same-type hits separated only by whitespace/nothing.

    `aggregation_strategy="simple"` still splits a single name into several
    sub-word groups when the tokenizer re-starts a "B-" tag mid-word (common
    for Kazakh proper names with a multilingual SentencePiece vocab), e.g.
    "Қалекес Дана Хайроллақызы" -> ("Қа", "лек", "ес Дана Хайроллақызы").
    Since these fragments are always contiguous in the source text, merging
    them back into one span is safe and recovers a single clean entity.
    """
    if not results:
        return results
    ordered = sorted(results, key=lambda r: r.start)
    merged = [ordered[0]]
    for r in ordered[1:]:
        last = merged[-1]
        gap = text[last.end : r.start]
        if r.entity_type == last.entity_type and gap.strip() == "" and len(gap) <= 1:
            merged[-1] = RecognizerResult(
                entity_type=last.entity_type,
                start=last.start,
                end=r.end,
                score=max(last.score, r.score),
            )
        else:
            merged.append(r)
    return merged


class TransformersNerRecognizer(EntityRecognizer):
    """Presidio EntityRecognizer wrapping a HuggingFace token-classification
    pipeline. Used for languages without a dedicated spaCy NER model (Kazakh),
    and optionally for Russian if `RU_NER_BACKEND == "transformer"`.
    """

    LABEL_TO_ENTITY = {
        # 2-3 letter CoNLL-style tags (e.g. RU_TRANSFORMER_MODEL / Burenko-rubert-base-ner)
        "PER": "PERSON",
        "ORG": "ORGANIZATION",
        "LOC": "LOCATION",
        "GPE": "LOCATION",
        # KazNERD label set (KK_TRANSFORMER_MODEL / yeshpanovrustem-xlm-roberta-large-kaznerd)
        "PERSON": "PERSON",
        "ORGANISATION": "ORGANIZATION",
        "ORGANIZATION": "ORGANIZATION",
        "LOCATION": "LOCATION",
        "FACILITY": "LOCATION",
    }

    def __init__(self, model_name: str, supported_language: str):
        self.model_name = model_name
        self._pipeline = None
        super().__init__(
            supported_entities=sorted(set(self.LABEL_TO_ENTITY.values())),
            supported_language=supported_language,
            name=f"TransformersNER[{model_name}]",
        )

    def load(self) -> None:
        """Lazily download/load the HF model. Called once by Presidio."""
        from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline

        logger.info("Loading transformer NER model '%s'...", self.model_name)
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModelForTokenClassification.from_pretrained(self.model_name)
        self._pipeline = pipeline(
            "ner",
            model=model,
            tokenizer=tokenizer,
            aggregation_strategy="simple",
        )

    def analyze(
        self,
        text: str,
        entities: List[str],
        nlp_artifacts: Optional[NlpArtifacts] = None,
    ) -> List[RecognizerResult]:
        if not text or not text.strip():
            return []
        if self._pipeline is None:
            self.load()

        results: List[RecognizerResult] = []
        try:
            ner_output = self._pipeline(text)
        except Exception:
            logger.exception("Transformer NER inference failed for model '%s'", self.model_name)
            return results

        for item in ner_output:
            raw_label = str(item.get("entity_group", item.get("entity", ""))).upper()
            raw_label = raw_label.replace("B-", "").replace("I-", "")
            entity_type = self.LABEL_TO_ENTITY.get(raw_label)
            if entity_type is None:
                continue
            if entities and entity_type not in entities:
                continue
            score = float(item.get("score", 0.75))
            if score < MIN_TRANSFORMER_NER_SCORE:
                continue
            results.append(
                RecognizerResult(
                    entity_type=entity_type,
                    start=int(item["start"]),
                    end=int(item["end"]),
                    score=score,
                )
            )
        return _merge_adjacent_same_type(results, text)


# ---------------------------------------------------------------------------
# NLP engine helpers
# ---------------------------------------------------------------------------


class BlankTokenizerNlpEngine(SpacyNlpEngine):
    """An NlpEngine that provides tokenization only (no NER).

    Presidio's AnalyzerEngine always needs an NlpEngine to produce
    NlpArtifacts (tokens) for its recognizers, even when the actual entity
    detection is fully delegated to a custom recognizer (as is the case for
    Kazakh, where no spaCy NER model exists). We use `spacy.blank(lang_code)`
    which only builds a tokenizer, with no downloadable model package
    required.
    """

    def __init__(self, lang_code: str):
        super().__init__(models=[{"lang_code": lang_code, "model_name": lang_code}])
        self.lang_code = lang_code

    def load(self) -> None:
        try:
            self.nlp = {self.lang_code: spacy.blank(self.lang_code)}
        except Exception:
            logger.warning(
                "spaCy has no blank tokenizer for language '%s'; "
                "falling back to the multilingual blank pipeline 'xx'.",
                self.lang_code,
            )
            self.nlp = {self.lang_code: spacy.blank("xx")}


def _build_ru_spacy_nlp_engine():
    configuration = {
        "nlp_engine_name": "spacy",
        "models": [{"lang_code": "ru", "model_name": RU_SPACY_MODEL}],
    }
    provider = NlpEngineProvider(nlp_configuration=configuration)
    return provider.create_engine()


def build_analyzer_engine(language: str) -> AnalyzerEngine:
    """Construct a fully wired Presidio AnalyzerEngine for "ru" or "kk"."""
    if language not in ("ru", "kk"):
        raise ValueError(f"Unsupported language: {language!r}")

    registry = RecognizerRegistry(supported_languages=[language])

    if language == "ru" and RU_NER_BACKEND == "spacy":
        nlp_engine = _build_ru_spacy_nlp_engine()
    else:
        nlp_engine = BlankTokenizerNlpEngine(lang_code=language)

    if not getattr(nlp_engine, "nlp", None):
        nlp_engine.load()

    registry.load_predefined_recognizers(nlp_engine=nlp_engine, languages=[language])

    # NB: `EntityRecognizer.__init__` calls `self.load()` eagerly, so a missing
    # `transformers`/`torch` install (or a HF download failure) surfaces right
    # here, not on first analyze() call.
    try:
        if language == "ru" and RU_NER_BACKEND == "transformer":
            registry.add_recognizer(TransformersNerRecognizer(RU_TRANSFORMER_MODEL, "ru"))
        elif language == "kk":
            registry.add_recognizer(TransformersNerRecognizer(KK_TRANSFORMER_MODEL, "kk"))
    except ImportError as exc:
        raise ImportError(
            "Не установлены пакеты 'transformers'/'torch', необходимые для "
            "распознавания ФИО/организаций. Выполните: pip install -r requirements.txt"
        ) from exc
    except OSError as exc:
        raise OSError(
            "Не удалось загрузить NER-модель с HuggingFace Hub. Проверьте "
            "подключение к интернету или доступность модели по имени "
            f"'{KK_TRANSFORMER_MODEL if language == 'kk' else RU_TRANSFORMER_MODEL}'."
        ) from exc

    for recognizer in build_regex_recognizers(language):
        registry.add_recognizer(recognizer)

    return AnalyzerEngine(registry=registry, nlp_engine=nlp_engine, supported_languages=[language])
