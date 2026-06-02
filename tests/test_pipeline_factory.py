from rag.pipelines.factory import (
    DEFAULT_CHAT_PIPELINE_MODE,
    build_chat_pipeline,
    get_chat_pipeline_mode,
)


def test_get_chat_pipeline_mode_defaults_to_legacy(monkeypatch):
    monkeypatch.delenv("RAG_CHAT_PIPELINE", raising=False)
    assert get_chat_pipeline_mode() == DEFAULT_CHAT_PIPELINE_MODE


def test_get_chat_pipeline_mode_reads_modular_env(monkeypatch):
    monkeypatch.setenv("RAG_CHAT_PIPELINE", "modular")
    assert get_chat_pipeline_mode() == "modular"


def test_get_chat_pipeline_mode_rejects_unknown_value(monkeypatch):
    monkeypatch.setenv("RAG_CHAT_PIPELINE", "something-else")
    assert get_chat_pipeline_mode() == DEFAULT_CHAT_PIPELINE_MODE


def test_build_chat_pipeline_uses_legacy_by_default(monkeypatch):
    sentinel = object()

    class DummyLegacy:
        def __init__(self, index_dir, **kwargs):
            self.index_dir = index_dir
            self.kwargs = kwargs

    class DummyModular:
        def __init__(self, index_dir, **kwargs):
            raise AssertionError("modular pipeline should not be used by default")

    monkeypatch.setattr("rag.pipelines.factory._get_legacy_pipeline_cls", lambda: DummyLegacy)
    monkeypatch.setattr("rag.pipelines.factory._get_modular_pipeline_cls", lambda: DummyModular)

    pipeline = build_chat_pipeline(index_dir="indexes/default", extra=sentinel)
    assert isinstance(pipeline, DummyLegacy)
    assert pipeline.kwargs["extra"] is sentinel


def test_build_chat_pipeline_uses_modular_when_requested(monkeypatch):
    sentinel = object()

    class DummyLegacy:
        def __init__(self, index_dir, **kwargs):
            raise AssertionError("legacy pipeline should not be used")

    class DummyModular:
        def __init__(self, index_dir, **kwargs):
            self.index_dir = index_dir
            self.kwargs = kwargs

    monkeypatch.setattr("rag.pipelines.factory._get_legacy_pipeline_cls", lambda: DummyLegacy)
    monkeypatch.setattr("rag.pipelines.factory._get_modular_pipeline_cls", lambda: DummyModular)

    pipeline = build_chat_pipeline(
        index_dir="indexes/default",
        pipeline_mode="modular",
        extra=sentinel,
    )
    assert isinstance(pipeline, DummyModular)
    assert pipeline.kwargs["extra"] is sentinel
