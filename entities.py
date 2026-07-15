"""Central definitions of the custom PII entity types used across the app.

Keeping these in one place avoids circular imports between `recognizers.py`
(builds the Presidio engine) and `anonymizer_engine.py` (turns detections into
stable, reversible tokens).
"""

# Presidio entity_type -> human-readable Russian token label used inside
# generated tokens, e.g. entity_type="PERSON" -> token "[ФИО_1]".
ENTITY_LABELS = {
    "PERSON": "ФИО",
    "ORGANIZATION": "КОМПАНИЯ",
    "LOCATION": "АДРЕС",
    "IIN_BIN": "ИИН_БИН",
    "KZ_IBAN": "IBAN",
    "PHONE_NUMBER": "ТЕЛЕФОН",
    "EMAIL_ADDRESS": "EMAIL",
    "DOCUMENT_ID": "ДОКУМЕНТ",
}

# Entities produced by deterministic regex recognizers. These are given
# priority over NLP/NER-based entities when spans overlap, since regex
# matches on IIN/BIN, IBAN, phone, email are effectively 100% precise.
#
# NB: PERSON is included here too. It is normally an NER-only entity type,
# but a deterministic "signature name" pattern (Фамилия И.О. / И.О. Фамилия,
# see recognizers.SIGNATURE_NAME_*_PATTERN) also emits PERSON, to catch
# abbreviated-initials names that NER models routinely miss. Prioritizing
# PERSON on overlap only matters when a span is ambiguous between it and
# another entity type, which is the desired tie-break either way.
REGEX_ENTITY_TYPES = {
    "IIN_BIN",
    "KZ_IBAN",
    "PHONE_NUMBER",
    "EMAIL_ADDRESS",
    "DOCUMENT_ID",
    "PERSON",
}

# Entities produced by NLP/NER models (spaCy or transformers).
NER_ENTITY_TYPES = {"PERSON", "ORGANIZATION", "LOCATION"}
