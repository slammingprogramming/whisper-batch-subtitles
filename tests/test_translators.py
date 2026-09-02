from __future__ import annotations

import sys
import types

import pytest

import whisper_batch_subtitles.translators as translators_module
from whisper_batch_subtitles.config import AppConfig
from whisper_batch_subtitles.translators import (
    DeepLTranslationBackend,
    GoogleTranslationBackend,
    TranslationBackend,
    TranslationEngine,
    _deepl_language_code,
    create_translation_engine,
)


class FakeTranslatorBatchOK:
    def __init__(self, source, target):
        self.target = target

    def translate_batch(self, texts):
        return [f"{text}->{self.target}" for text in texts]

    def translate(self, text):  # pragma: no cover - should not be reached
        raise AssertionError("translate() should not be used when translate_batch succeeds")


class FakeTranslatorBatchRaises:
    def __init__(self, source, target):
        self.target = target

    def translate_batch(self, texts):
        raise RuntimeError("batch endpoint down")

    def translate(self, text):
        return f"{text}~{self.target}"


class FakeTranslatorBatchMismatch:
    def __init__(self, source, target):
        self.target = target

    def translate_batch(self, texts):
        return ["only-one-result"]

    def translate(self, text):
        return f"{text}~{self.target}"


def make_segments():
    return [
        {"start": 0.0, "end": 1.0, "text": "hello"},
        {"start": 1.0, "end": 2.0, "text": "world", "speaker": "me", "speaker_label": "Me"},
    ]


def test_translate_segments_uses_batch_when_lengths_match(monkeypatch):
    monkeypatch.setattr(translators_module, "GoogleTranslator", FakeTranslatorBatchOK)
    engine = TranslationEngine()
    result = engine.translate_segments(make_segments(), "es")
    assert [segment["text"] for segment in result] == ["hello->es", "world->es"]
    assert result[1]["speaker"] == "me"
    assert result[1]["speaker_label"] == "Me"


def test_translate_segments_falls_back_per_segment_on_batch_exception(monkeypatch):
    monkeypatch.setattr(translators_module, "GoogleTranslator", FakeTranslatorBatchRaises)
    engine = TranslationEngine()
    result = engine.translate_segments(make_segments(), "es")
    assert [segment["text"] for segment in result] == ["hello~es", "world~es"]


def test_translate_segments_falls_back_per_segment_on_length_mismatch(monkeypatch):
    # Regression test: a batch call that "succeeds" but returns the wrong
    # number of results must retry per-segment instead of silently keeping
    # the original untranslated text mislabeled as a translation.
    monkeypatch.setattr(translators_module, "GoogleTranslator", FakeTranslatorBatchMismatch)
    engine = TranslationEngine()
    result = engine.translate_segments(make_segments(), "es")
    texts = [segment["text"] for segment in result]
    assert texts == ["hello~es", "world~es"]
    assert "only-one-result" not in texts
    assert "hello" not in texts  # must not silently fall back to the original text


def test_translate_segments_empty_input_returns_empty(monkeypatch):
    monkeypatch.setattr(translators_module, "GoogleTranslator", FakeTranslatorBatchOK)
    engine = TranslationEngine()
    assert engine.translate_segments([], "es") == []


def test_translator_instances_are_cached_per_target_language(monkeypatch):
    created = []

    class TrackingTranslator(FakeTranslatorBatchOK):
        def __init__(self, source, target):
            super().__init__(source, target)
            created.append(target)

    monkeypatch.setattr(translators_module, "GoogleTranslator", TrackingTranslator)
    engine = TranslationEngine()
    engine.translate_segments(make_segments(), "es")
    engine.translate_segments(make_segments(), "es")
    engine.translate_segments(make_segments(), "fr")
    assert created == ["es", "fr"]


class UppercaseBackend(TranslationBackend):
    """A minimal custom backend, used to prove the pluggable interface works end to end."""

    def translate_batch(self, texts, target_language):
        return [f"{text.upper()}[{target_language}]" for text in texts]


def test_translation_engine_accepts_a_custom_backend():
    engine = TranslationEngine(UppercaseBackend())
    result = engine.translate_segments(make_segments(), "es")
    assert [segment["text"] for segment in result] == ["HELLO[es]", "WORLD[es]"]


class BrokenBackend(TranslationBackend):
    """Violates the TranslationBackend contract by returning the wrong length."""

    def translate_batch(self, texts, target_language):
        return ["only one"]


def test_translation_engine_falls_back_to_original_text_on_backend_contract_violation():
    engine = TranslationEngine(BrokenBackend())
    result = engine.translate_segments(make_segments(), "es")
    # engine-level safety net: a misbehaving backend must never misalign segments
    assert [segment["text"] for segment in result] == ["hello", "world"]


def test_google_translation_backend_is_used_by_default(monkeypatch):
    monkeypatch.setattr(translators_module, "GoogleTranslator", FakeTranslatorBatchOK)
    engine = TranslationEngine()
    assert isinstance(engine._backend, GoogleTranslationBackend)
    result = engine.translate_segments(make_segments(), "es")
    assert [segment["text"] for segment in result] == ["hello->es", "world->es"]


def test_deepl_language_code_overrides_known_regional_codes():
    assert _deepl_language_code("en") == "EN-US"
    assert _deepl_language_code("pt") == "PT-PT"
    assert _deepl_language_code("es") == "ES"
    assert _deepl_language_code("JA") == "JA"


def _install_fake_deepl_module(monkeypatch, *, translator_cls):
    fake_module = types.ModuleType("deepl")
    fake_module.Translator = translator_cls  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "deepl", fake_module)


def test_deepl_backend_translates_via_fake_sdk(monkeypatch):
    class FakeResult:
        def __init__(self, text):
            self.text = text

    class FakeDeepLTranslator:
        def __init__(self, api_key):
            self.api_key = api_key

        def translate_text(self, texts, target_lang):
            return [FakeResult(f"{text}|{target_lang}") for text in texts]

    _install_fake_deepl_module(monkeypatch, translator_cls=FakeDeepLTranslator)

    backend = DeepLTranslationBackend("fake-key")
    result = backend.translate_batch(["hello", "world"], "es")
    assert result == ["hello|ES", "world|ES"]


def test_deepl_backend_falls_back_to_original_text_on_api_error(monkeypatch):
    class FakeDeepLTranslator:
        def __init__(self, api_key):
            pass

        def translate_text(self, texts, target_lang):
            raise RuntimeError("DeepL API is down")

    _install_fake_deepl_module(monkeypatch, translator_cls=FakeDeepLTranslator)

    backend = DeepLTranslationBackend("fake-key")
    result = backend.translate_batch(["hello", "world"], "es")
    assert result == ["hello", "world"]


def test_deepl_backend_raises_clear_error_when_package_missing(monkeypatch):
    monkeypatch.setitem(sys.modules, "deepl", None)  # simulate ImportError on `import deepl`
    with pytest.raises(RuntimeError, match="deepl"):
        DeepLTranslationBackend("fake-key")


def test_create_translation_engine_defaults_to_google():
    config = AppConfig(translation_backend="google")
    engine = create_translation_engine(config)
    assert isinstance(engine._backend, GoogleTranslationBackend)


def test_create_translation_engine_deepl_requires_api_key():
    config = AppConfig(translation_backend="deepl", deepl_api_key=None)
    with pytest.raises(RuntimeError, match="API key"):
        create_translation_engine(config)


def test_create_translation_engine_deepl_with_key_builds_backend(monkeypatch):
    class FakeDeepLTranslator:
        def __init__(self, api_key):
            self.api_key = api_key

    _install_fake_deepl_module(monkeypatch, translator_cls=FakeDeepLTranslator)
    config = AppConfig(translation_backend="deepl", deepl_api_key="fake-key")
    engine = create_translation_engine(config)
    assert isinstance(engine._backend, DeepLTranslationBackend)


def test_create_translation_engine_rejects_unknown_backend():
    config = AppConfig(translation_backend="bogus")
    with pytest.raises(ValueError, match="bogus"):
        create_translation_engine(config)
