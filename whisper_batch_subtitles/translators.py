from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from deep_translator import GoogleTranslator

if TYPE_CHECKING:
    from whisper_batch_subtitles.config import AppConfig


class TranslationBackend(ABC):
    """A translation backend translates a batch of plain-text strings into a target
    language. It must always return a list the same length as its input -- callers
    treat a length mismatch as a hard failure and fall back to keeping the original
    text rather than silently misaligning segments."""

    @abstractmethod
    def translate_batch(self, texts: list[str], target_language: str) -> list[str]: ...


class GoogleTranslationBackend(TranslationBackend):
    """Wraps deep-translator's unofficial Google Translate endpoint. No API key, no
    guaranteed rate limits -- fine for moderate volume, the long-tail fallback path
    below exists because that endpoint is known to be flaky under load."""

    def __init__(self) -> None:
        self._translators: dict[str, GoogleTranslator] = {}

    def _get_translator(self, target_language: str) -> GoogleTranslator:
        translator = self._translators.get(target_language)
        if translator is None:
            translator = GoogleTranslator(source="auto", target=target_language)
            self._translators[target_language] = translator
        return translator

    def translate_batch(self, texts: list[str], target_language: str) -> list[str]:
        if not texts:
            return []
        translator = self._get_translator(target_language)

        translated_texts: list[str] = []
        if hasattr(translator, "translate_batch"):
            try:
                translated_texts = list(translator.translate_batch(texts))
            except Exception:
                translated_texts = []
            if len(translated_texts) != len(texts):
                translated_texts = []

        if not translated_texts:
            translated_texts = []
            for text in texts:
                try:
                    translated_texts.append(str(translator.translate(text)))
                except Exception:
                    translated_texts.append(text)

        return translated_texts


class DeepLTranslationBackend(TranslationBackend):
    """Optional backend using the official DeepL API. Requires the `deepl` package
    (install the `deepl` extra) and an API key. Untested against a live DeepL account
    in this codebase -- verify manually before relying on it for a real run."""

    def __init__(self, api_key: str) -> None:
        try:
            import deepl  # type: ignore
        except ImportError as error:
            raise RuntimeError(
                "The deepl package is not installed. Install the optional 'deepl' extra to use this backend."
            ) from error
        self._translator = deepl.Translator(api_key)

    def translate_batch(self, texts: list[str], target_language: str) -> list[str]:
        if not texts:
            return []
        try:
            results = self._translator.translate_text(texts, target_lang=_deepl_language_code(target_language))
        except Exception:
            return list(texts)
        if isinstance(results, list):
            translated = [str(result.text) for result in results]
        else:
            translated = [str(results.text)]
        if len(translated) != len(texts):
            return list(texts)
        return translated


def _deepl_language_code(target_language: str) -> str:
    # DeepL wants region-qualified codes for a handful of languages; everything else
    # is just the uppercased ISO code.
    overrides = {"en": "EN-US", "pt": "PT-PT"}
    return overrides.get(target_language.lower(), target_language.upper())


class TranslationEngine:
    def __init__(self, backend: TranslationBackend | None = None) -> None:
        self._backend = backend or GoogleTranslationBackend()

    def translate_segments(
        self, segments: list[dict[str, float | str]], target_language: str
    ) -> list[dict[str, float | str]]:
        texts = [str(segment["text"]).strip() for segment in segments]
        translated_texts = self._backend.translate_batch(texts, target_language) if texts else []
        if len(translated_texts) != len(texts):
            # Backend contract violation safety net -- never silently misalign segments.
            translated_texts = list(texts)

        translated_segments: list[dict[str, float | str]] = []
        for segment, translated_text in zip(segments, translated_texts):
            translated_segment: dict[str, float | str] = {
                "start": float(segment["start"]),
                "end": float(segment["end"]),
                "text": translated_text,
            }
            if "speaker" in segment:
                translated_segment["speaker"] = str(segment["speaker"])
            if "speaker_label" in segment:
                translated_segment["speaker_label"] = str(segment["speaker_label"])
            translated_segments.append(translated_segment)

        return translated_segments


def create_translation_engine(config: "AppConfig") -> TranslationEngine:
    backend_name = (config.translation_backend or "google").lower()
    if backend_name == "deepl":
        api_key = config.deepl_api_key or os.environ.get("DEEPL_API_KEY")
        if not api_key:
            raise RuntimeError(
                "translation_backend is 'deepl' but no API key was provided "
                "(set deepl_api_key in config or the DEEPL_API_KEY environment variable)."
            )
        return TranslationEngine(DeepLTranslationBackend(api_key))
    if backend_name != "google":
        raise ValueError(f"Unknown translation_backend: {backend_name!r} (expected 'google' or 'deepl')")
    return TranslationEngine(GoogleTranslationBackend())
