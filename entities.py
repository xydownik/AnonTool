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
# NB: PERSON and LOCATION are included here too. They are normally
# NER-only entity types, but each also has a deterministic regex fallback:
# PERSON gets a "signature name" pattern (Фамилия И.О. / И.О. Фамилия, see
# recognizers.SIGNATURE_NAME_*_PATTERN) to catch abbreviated-initials names
# NER routinely misses, and LOCATION gets a street-address pattern (see
# recognizers.STREET_ADDRESS_PATTERN) since spaCy's ru_core_news_lg rarely
# tags street/house-number spans at all. Prioritizing them on overlap only
# matters when a span is ambiguous between the regex and another entity
# type, which is the desired tie-break either way.
REGEX_ENTITY_TYPES = {
    "IIN_BIN",
    "KZ_IBAN",
    "PHONE_NUMBER",
    "EMAIL_ADDRESS",
    "DOCUMENT_ID",
    "PERSON",
    "LOCATION",
}

# Entities produced by NLP/NER models (spaCy or transformers).
NER_ENTITY_TYPES = {"PERSON", "ORGANIZATION", "LOCATION"}
