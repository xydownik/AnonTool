"""Reverse Masking — обезличивание юридических документов для отправки в LLM.

Запуск:
    streamlit run app.py

См. requirements.txt и README-инструкции по установке языковых моделей.
"""

from __future__ import annotations

import io
import re

import streamlit as st
from docx import Document
from docx.opc.exceptions import PackageNotFoundError

import database as db
import docx_processor as docx_utils
from anonymizer_engine import AnonymizerSession, restore_plain_text

db.init_db()

st.set_page_config(
    page_title="Reverse Masking | Обезличивание для юристов",
    page_icon="🛡️",
    layout="wide",
)

# Recognizes leftover tokens like [ФИО_1], [IBAN_2], [ИИН_БИН_3]
TOKEN_PATTERN = re.compile(r"\[[A-ZА-ЯЁ0-9_]+_\d+\]")

SYSTEM_PROMPT_HINT = (
    "Ты — юридический ассистент. Тебе передан текст документа, в котором часть "
    "данных заменена на технические токены в квадратных скобках, например "
    "[ФИО_1], [КОМПАНИЯ_2], [ИИН_БИН_1], [IBAN_1].\n\n"
    "Отредактируй формулировки согласно моему запросу, но СТРОГО ЗАПРЕЩЕНО:\n"
    "— удалять, переводить, изменять или перефразировать сами токены вида "
    "[ТИП_НОМЕР];\n"
    "— придумывать новые токены в квадратных скобках;\n"
    "— менять регистр, порядок символов или пробелы внутри токена.\n\n"
    "Каждый токен должен остаться в тексте ровно в исходном виде, столько раз, "
    "сколько это грамматически необходимо для связности текста."
)

DOCX_MIME = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"


def _init_state() -> None:
    defaults = {
        "anon_preview_text": "",
        "anon_docx_bytes": None,
        "anon_docx_name": None,
        "anon_mapping_count": 0,
        "last_session_id": "",
        "restored_preview_text": "",
        "restored_docx_bytes": None,
        "restored_docx_name": None,
        "restore_success": False,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def _render_header() -> None:
    st.title("🛡️ Reverse Masking — обезличивание юридических документов")
    st.caption(
        "Обратимая псевдонимизация персональных и корпоративных данных перед "
        "отправкой текста во внешние LLM (ChatGPT, Claude и т.д.)."
    )
    with st.expander("📋 Инструкция: как безопасно работать с ChatGPT / Claude", expanded=False):
        st.markdown(
            "1. Обезличьте документ на вкладке **«1. Обезличивание»** и сохраните "
            "**ключ восстановления**.\n"
            "2. Скопируйте обезличенный текст **вместе с системным промптом ниже** "
            "в ChatGPT/Claude и попросите отредактировать формулировки.\n"
            "3. Вставьте изменённый нейросетью текст и ключ на вкладке "
            "**«2. Восстановление»**, чтобы вернуть реальные данные на места."
        )
        st.markdown("**Системный промпт для LLM (скопируйте вместе с текстом):**")
        st.code(SYSTEM_PROMPT_HINT, language="text")
        st.info(
            "⚠️ Ключ восстановления действителен 24 часа. Без него восстановить "
            "оригинальные данные будет невозможно — сохраните его в надёжном месте.",
            icon="⚠️",
        )


def _find_leftover_tokens(text: str, known_tokens) -> list:
    found = set(TOKEN_PATTERN.findall(text))
    return sorted(found - set(known_tokens))


_init_state()
_render_header()

tab1, tab2 = st.tabs(
    ["1. Обезличивание (прямой ход)", "2. Восстановление оригинала (обратный ход)"]
)

# ---------------------------------------------------------------------------
# Tab 1 — Anonymization
# ---------------------------------------------------------------------------
with tab1:
    st.subheader("Шаг 1 — загрузите документ или вставьте текст")

    col_lang, col_mode = st.columns(2)
    with col_lang:
        language_label = st.radio(
            "Язык документа", options=["Русский", "Қазақша"], horizontal=True, key="anon_lang"
        )
        language = "ru" if language_label == "Русский" else "kk"
    with col_mode:
        input_mode = st.radio(
            "Способ ввода", ["Загрузить .docx", "Вставить текст"], horizontal=True, key="anon_mode"
        )

    uploaded_file = None
    raw_text = ""
    if input_mode == "Загрузить .docx":
        uploaded_file = st.file_uploader("Файл .docx", type=["docx"], key="anon_upload")
    else:
        raw_text = st.text_area("Текст документа", height=300, key="anon_text_input")

    if st.button("🔒 Обезличить документ", type="primary", use_container_width=True):
        if not uploaded_file and not raw_text.strip():
            st.warning("Загрузите файл .docx или введите текст для обезличивания.")
        else:
            with st.spinner("Анализируем текст, определяем сущности... Это может занять до минуты при первом запуске."):
                try:
                    session = AnonymizerSession(language=language)

                    if uploaded_file is not None:
                        try:
                            document = Document(io.BytesIO(uploaded_file.read()))
                        except PackageNotFoundError as exc:
                            raise ValueError(
                                "Файл повреждён или не является корректным .docx документом."
                            ) from exc

                        for paragraph in docx_utils.iter_all_paragraphs(document):
                            matches = session.analyze_paragraph(paragraph.text)
                            docx_utils.apply_replacements_to_paragraph(paragraph, matches)

                        buffer = io.BytesIO()
                        document.save(buffer)
                        buffer.seek(0)

                        st.session_state["anon_docx_bytes"] = buffer.getvalue()
                        st.session_state["anon_docx_name"] = f"anonymized_{uploaded_file.name}"
                        st.session_state["anon_preview_text"] = docx_utils.extract_docx_text(document)
                    else:
                        st.session_state["anon_preview_text"] = session.anonymize_plain_text(raw_text)
                        st.session_state["anon_docx_bytes"] = None
                        st.session_state["anon_docx_name"] = None

                    if not session.mapping_records:
                        st.warning(
                            "В тексте не найдено ни одной сущности для обезличивания. "
                            "Проверьте выбранный язык или содержимое документа."
                        )

                    session_id = db.generate_unique_session_key()
                    db.save_mapping(session_id, session.mapping_records)

                    st.session_state["last_session_id"] = session_id
                    st.session_state["anon_mapping_count"] = len(session.mapping_records)

                except Exception as exc:
                    st.error(f"Не удалось обработать документ: {exc}")
                    st.session_state["anon_preview_text"] = ""

    if st.session_state.get("anon_preview_text"):
        st.success(
            f"Готово! Найдено и заменено уникальных сущностей: "
            f"{st.session_state['anon_mapping_count']}."
        )

        st.markdown("### 🔑 Ключ восстановления — сохраните его!")
        st.code(st.session_state["last_session_id"], language=None)
        st.caption("Ключ действителен 24 часа. Без него восстановить оригинальные данные будет невозможно.")

        st.markdown("### Обезличенный текст")
        st.caption("Скопируйте этот текст вместе с системным промптом (см. инструкцию выше) в ChatGPT / Claude.")
        st.text_area(
            "Обезличенный текст",
            value=st.session_state["anon_preview_text"],
            height=350,
            label_visibility="collapsed",
        )

        if st.session_state.get("anon_docx_bytes"):
            st.download_button(
                "⬇️ Скачать обезличенный .docx",
                data=st.session_state["anon_docx_bytes"],
                file_name=st.session_state.get("anon_docx_name") or "anonymized.docx",
                mime=DOCX_MIME,
            )

# ---------------------------------------------------------------------------
# Tab 2 — Restoration
# ---------------------------------------------------------------------------
with tab2:
    st.subheader("Шаг 2 — восстановите оригинальные данные")

    restore_key = st.text_input(
        "Ключ восстановления (например, KZ-74A1)", key="restore_key_input"
    )
    restore_mode = st.radio(
        "Способ ввода изменённого текста",
        ["Вставить текст", "Загрузить .docx"],
        horizontal=True,
        key="restore_mode",
    )

    restore_upload = None
    restore_text = ""
    if restore_mode == "Загрузить .docx":
        restore_upload = st.file_uploader("Файл .docx с токенами", type=["docx"], key="restore_upload")
    else:
        restore_text = st.text_area(
            "Текст с токенами (ответ, полученный от нейросети)", height=300, key="restore_text_input"
        )

    if st.button("🔓 Восстановить оригинал", type="primary", use_container_width=True):
        if not restore_key.strip():
            st.warning("Введите ключ восстановления.")
        elif not restore_upload and not restore_text.strip():
            st.warning("Вставьте текст или загрузите файл с токенами.")
        else:
            try:
                mapping = db.load_mapping(restore_key.strip())

                if restore_upload is not None:
                    try:
                        document = Document(io.BytesIO(restore_upload.read()))
                    except PackageNotFoundError as exc:
                        raise ValueError(
                            "Файл повреждён или не является корректным .docx документом."
                        ) from exc

                    for paragraph in docx_utils.iter_all_paragraphs(document):
                        matches = docx_utils.find_literal_matches(paragraph.text, mapping)
                        docx_utils.apply_replacements_to_paragraph(paragraph, matches)

                    buffer = io.BytesIO()
                    document.save(buffer)
                    buffer.seek(0)

                    st.session_state["restored_docx_bytes"] = buffer.getvalue()
                    st.session_state["restored_docx_name"] = f"restored_{restore_upload.name}"
                    st.session_state["restored_preview_text"] = docx_utils.extract_docx_text(document)
                else:
                    st.session_state["restored_preview_text"] = restore_plain_text(restore_text, mapping)
                    st.session_state["restored_docx_bytes"] = None
                    st.session_state["restored_docx_name"] = None

                leftover = _find_leftover_tokens(
                    st.session_state["restored_preview_text"], mapping.keys()
                )
                if leftover:
                    st.warning(
                        "⚠️ В тексте остались токены, не найденные в ключе "
                        f"«{restore_key.strip()}»: {', '.join(leftover)}. "
                        "Возможно, нейросеть исказила их написание — проверьте документ вручную."
                    )

                st.session_state["restore_success"] = True

            except KeyError as exc:
                st.error(str(exc))
                st.session_state["restore_success"] = False
            except Exception as exc:
                st.error(f"Ошибка при восстановлении документа: {exc}")
                st.session_state["restore_success"] = False

    if st.session_state.get("restore_success") and st.session_state.get("restored_preview_text"):
        st.success("Данные успешно восстановлены.")
        st.text_area(
            "Итоговый текст",
            value=st.session_state["restored_preview_text"],
            height=350,
            label_visibility="collapsed",
        )
        if st.session_state.get("restored_docx_bytes"):
            st.download_button(
                "⬇️ Скачать восстановленный .docx",
                data=st.session_state["restored_docx_bytes"],
                file_name=st.session_state.get("restored_docx_name") or "restored.docx",
                mime=DOCX_MIME,
            )
