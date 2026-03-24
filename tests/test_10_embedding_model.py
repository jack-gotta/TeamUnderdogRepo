"""Tests for embedding model execution and integration."""

import pytest


def test_embedding_model_import() -> None:
    """Verify embedding model can be imported from controlled interface."""
    from llamaindex_models import get_text_embedding_3_large, ModelAccessError
    
    # This should not raise for authorized model
    assert callable(get_text_embedding_3_large)


def test_embedding_model_unauthorized_access() -> None:
    """Verify unauthorized models are blocked."""
    from llamaindex_models import get_chat_model, ModelAccessError
    
    with pytest.raises(ModelAccessError):
        get_chat_model("unauthorized-model-xyz")


def test_embedding_model_registry() -> None:
    """Verify model registry contains text-embedding-3-large."""
    from llamaindex_models import get_available_models
    
    models = get_available_models()
    assert "text-embedding-3-large" in models["embeddings"]
    
    embedding_config = models["embeddings"]["text-embedding-3-large"]
    assert embedding_config["model_name"] == "text-embedding-3-large"
    assert embedding_config["deployment_name"] == "text-embedding-3-large"


def test_embedding_model_isolation() -> None:
    """Verify model isolation prevents direct access."""
    from llamaindex_models import validate_model_access
    
    # Authorized
    assert validate_model_access("embeddings", "text-embedding-3-large")
    
    # Unauthorized
    assert not validate_model_access("embeddings", "gpt-3.5-turbo")
    assert not validate_model_access("chat", "text-embedding-3-large")
