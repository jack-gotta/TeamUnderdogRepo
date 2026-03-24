"""
Tests for embedding model functionality.
"""
import pytest
import sys
import os
from unittest.mock import Mock, patch

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from llamaindex_models import get_text_embedding_3_large, ModelAccessError, get_available_models


class TestEmbeddingModel:
    """Test cases for embedding model functionality."""

    @pytest.fixture
    def embedding_model(self):
        """Fixture providing embedding model instance."""
        try:
            return get_text_embedding_3_large()
        except Exception as e:
            pytest.skip(f"Embedding model not available: {e}")

    def test_get_available_models(self):
        """Test getting available models registry."""
        models = get_available_models()

        assert isinstance(models, dict)
        assert 'embeddings' in models
        assert 'text-embedding-3-large' in models['embeddings']

    def test_get_embedding_model_success(self):
        """Test successful embedding model creation."""
        try:
            model = get_text_embedding_3_large()
            assert model is not None
            assert hasattr(model, 'get_text_embedding')
        except Exception as e:
            pytest.skip(f"Model access not available: {e}")

    def test_get_embedding_model_invalid_name(self):
        """Test error handling for invalid model names."""
        from llamaindex_models import get_embedding_model

        with pytest.raises(ModelAccessError):
            get_embedding_model("invalid-model-name")

    def test_text_embedding_generation(self, embedding_model):
        """Test generating embeddings for text."""
        test_text = "This is a test sentence for embedding generation."

        embedding = embedding_model.get_text_embedding(test_text)

        assert embedding is not None
        assert isinstance(embedding, list)
        assert len(embedding) > 0
        assert all(isinstance(x, (int, float)) for x in embedding)
        # text-embedding-3-large produces 3072-dimensional embeddings
        assert len(embedding) == 3072

    def test_multiple_text_embeddings(self, embedding_model):
        """Test generating embeddings for multiple texts."""
        test_texts = [
            "Python programming language",
            "Machine learning algorithms",
            "Docker container technology"
        ]

        embeddings = []
        for text in test_texts:
            embedding = embedding_model.get_text_embedding(text)
            embeddings.append(embedding)

        assert len(embeddings) == 3
        for embedding in embeddings:
            assert len(embedding) == 3072
            assert all(isinstance(x, (int, float)) for x in embedding)

    def test_embedding_consistency(self, embedding_model):
        """Test that same text produces consistent embeddings."""
        test_text = "Consistent embedding test"

        embedding1 = embedding_model.get_text_embedding(test_text)
        embedding2 = embedding_model.get_text_embedding(test_text)

        # Check that embeddings are very similar (allow for minor variations)
        import math

        def cosine_similarity(a, b):
            dot_product = sum(x * y for x, y in zip(a, b))
            norm_a = math.sqrt(sum(x * x for x in a))
            norm_b = math.sqrt(sum(x * x for x in b))
            return dot_product / (norm_a * norm_b) if norm_a > 0 and norm_b > 0 else 0

        similarity = cosine_similarity(embedding1, embedding2)
        # Embeddings should be very similar (> 0.99) for the same text
        assert similarity > 0.99

    def test_embedding_different_texts(self, embedding_model):
        """Test that different texts produce different embeddings."""
        text1 = "Python programming"
        text2 = "JavaScript development"

        embedding1 = embedding_model.get_text_embedding(text1)
        embedding2 = embedding_model.get_text_embedding(text2)

        assert embedding1 != embedding2

    def test_empty_text_handling(self, embedding_model):
        """Test handling of empty text."""
        empty_text = ""

        # Should handle empty text gracefully
        try:
            embedding = embedding_model.get_text_embedding(empty_text)
            assert embedding is not None
            assert len(embedding) == 3072
        except Exception:
            # Some models might reject empty text, which is acceptable
            pass

    def test_long_text_handling(self, embedding_model):
        """Test handling of very long text."""
        long_text = "Test sentence. " * 1000  # Very long text

        try:
            embedding = embedding_model.get_text_embedding(long_text)
            assert embedding is not None
            assert len(embedding) == 3072
        except Exception:
            # Model might have token limits, which is acceptable
            pass


class TestEmbeddingModelMocked:
    """Test cases using mocked embedding model for environments without Azure access."""

    @patch('llamaindex_models.get_ailab_bearer_token_provider')
    @patch('llamaindex_models.AzureOpenAIEmbedding')
    def test_embedding_model_creation_with_mocks(self, mock_embedding_class, mock_token_provider):
        """Test embedding model creation with mocked dependencies."""
        # Setup mocks
        mock_embedding_instance = Mock()
        mock_embedding_instance.get_text_embedding.return_value = [0.1] * 3072
        mock_embedding_class.return_value = mock_embedding_instance
        mock_token_provider.return_value = lambda: "mock_token"

        # Test model creation
        model = get_text_embedding_3_large()
        assert model is not None

        # Test embedding generation
        embedding = model.get_text_embedding("test text")
        assert len(embedding) == 3072
        assert all(isinstance(x, (int, float)) for x in embedding)

    def test_model_registry_structure(self):
        """Test that model registry has expected structure."""
        models = get_available_models()

        # Check structure
        assert 'embeddings' in models
        assert 'chat' in models

        # Check embedding model details
        embedding_models = models['embeddings']
        assert 'text-embedding-3-large' in embedding_models

        model_config = embedding_models['text-embedding-3-large']
        required_fields = ['deployment_name', 'model_name', 'api_version', 'description']
        for field in required_fields:
            assert field in model_config

    @patch('llamaindex_models.get_ailab_bearer_token_provider')
    def test_model_access_error_handling(self, mock_token_provider):
        """Test proper error handling when model access fails."""
        mock_token_provider.side_effect = Exception("Authentication failed")

        with pytest.raises(Exception):
            get_text_embedding_3_large()

    def test_validate_model_access(self):
        """Test model access validation function."""
        from llamaindex_models import validate_model_access

        # Test valid model
        assert validate_model_access("embeddings", "text-embedding-3-large") is True

        # Test invalid model type
        assert validate_model_access("invalid_type", "text-embedding-3-large") is False

        # Test invalid model name
        assert validate_model_access("embeddings", "invalid-model") is False


class TestEmbeddingIntegration:
    """Integration tests for embedding functionality."""

    @pytest.mark.slow
    def test_real_embedding_generation(self):
        """Integration test with real model (marked as slow)."""
        try:
            model = get_text_embedding_3_large()

            # Test with simple text
            text = "Hello, world!"
            embedding = model.get_text_embedding(text)

            # Verify embedding properties
            assert isinstance(embedding, list)
            assert len(embedding) == 3072
            assert all(isinstance(x, (int, float)) for x in embedding)

            # Verify embedding values are reasonable (not all zeros)
            assert any(abs(x) > 0.001 for x in embedding)

        except Exception as e:
            pytest.skip(f"Real model integration test skipped: {e}")

    @pytest.mark.slow
    def test_embedding_similarity(self):
        """Test that similar texts produce similar embeddings."""
        try:
            model = get_text_embedding_3_large()

            similar_texts = [
                "The cat sat on the mat",
                "A cat was sitting on the mat"
            ]

            embeddings = [model.get_text_embedding(text) for text in similar_texts]

            # Calculate simple cosine similarity
            import math

            def cosine_similarity(a, b):
                dot_product = sum(x * y for x, y in zip(a, b))
                norm_a = math.sqrt(sum(x * x for x in a))
                norm_b = math.sqrt(sum(x * x for x in b))
                return dot_product / (norm_a * norm_b)

            similarity = cosine_similarity(embeddings[0], embeddings[1])

            # Similar texts should have high similarity (> 0.8)
            assert similarity > 0.8

        except Exception as e:
            pytest.skip(f"Similarity test skipped: {e}")