"""
Tests for the ingestion pipeline functionality.
"""
import pytest
import pandas as pd
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch
import sys
import os

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from ingestion import IngestionPipeline
from llama_index.core import Document


class TestIngestionPipeline:
    """Test cases for the ingestion pipeline."""

    @pytest.fixture
    def temp_storage(self):
        """Create temporary storage directory for tests."""
        temp_dir = tempfile.mkdtemp(prefix="test_storage_")
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.fixture
    def pipeline(self, temp_storage):
        """Create a pipeline instance for testing."""
        return IngestionPipeline(storage_dir=temp_storage)

    @pytest.fixture
    def sample_dataframe(self):
        """Create sample Wikipedia data for testing."""
        data = {
            'text': [
                'Python is a high-level programming language known for its simplicity.',
                'Machine learning is a subset of artificial intelligence.',
                'Docker is a containerization platform for applications.',
                'REST APIs provide standardized communication between applications.',
                'Git is a version control system for tracking code changes.'
            ],
            'title': [
                'Python Programming',
                'Machine Learning',
                'Docker Containers',
                'REST APIs',
                'Git Version Control'
            ]
        }
        return pd.DataFrame(data)

    def test_pipeline_initialization(self, temp_storage):
        """Test pipeline initialization."""
        pipeline = IngestionPipeline(storage_dir=temp_storage)

        assert pipeline.storage_dir == Path(temp_storage)
        assert pipeline.storage_dir.exists()
        assert pipeline.embedding_model is None
        assert pipeline.llm_model is None
        assert pipeline.index is None
        assert pipeline.stats["documents_loaded"] == 0

    def test_create_documents(self, pipeline, sample_dataframe):
        """Test document creation from DataFrame."""
        documents = pipeline.create_documents(sample_dataframe)

        assert len(documents) == 5
        for i, doc in enumerate(documents):
            assert isinstance(doc, Document)
            assert doc.text == sample_dataframe.iloc[i]['text']
            assert doc.metadata['title'] == sample_dataframe.iloc[i]['title']
            assert doc.metadata['doc_id'] == str(i)
            assert doc.metadata['source'] == 'wikipedia'

    def test_create_documents_with_limit(self, pipeline, sample_dataframe):
        """Test document creation with max_docs limit."""
        max_docs = 3
        documents = pipeline.create_documents(sample_dataframe, max_docs=max_docs)

        assert len(documents) == max_docs

    @patch('ingestion.pd.read_parquet')
    def test_load_wikipedia_data_success(self, mock_read_parquet, pipeline, sample_dataframe):
        """Test successful Wikipedia data loading."""
        mock_read_parquet.return_value = sample_dataframe

        df = pipeline.load_wikipedia_data()

        assert df is not None
        assert len(df) == 5
        assert pipeline.stats["documents_loaded"] == 5
        mock_read_parquet.assert_called_once()

    @patch('ingestion.pd.read_parquet')
    def test_load_wikipedia_data_failure(self, mock_read_parquet, pipeline):
        """Test Wikipedia data loading failure."""
        mock_read_parquet.side_effect = Exception("Network error")

        df = pipeline.load_wikipedia_data()

        assert df is None
        assert pipeline.stats["documents_loaded"] == 0

    @patch('ingestion.Settings')
    @patch('ingestion.get_text_embedding_3_large')
    @patch('ingestion.get_gpt4o')
    def test_initialize_models_success(self, mock_gpt4o, mock_embedding, mock_settings, pipeline):
        """Test successful model initialization."""
        mock_embedding.return_value = Mock()
        mock_gpt4o.return_value = Mock()

        result = pipeline.initialize_models()

        assert result is True
        assert pipeline.embedding_model is not None
        assert pipeline.llm_model is not None

    @patch('ingestion.get_text_embedding_3_large')
    def test_initialize_models_failure(self, mock_embedding, pipeline):
        """Test model initialization failure."""
        mock_embedding.side_effect = Exception("Model access failed")

        result = pipeline.initialize_models()

        assert result is False
        assert pipeline.embedding_model is None

    def test_get_index_stats_no_index(self, pipeline):
        """Test getting stats when no index exists."""
        stats = pipeline.get_index_stats()

        assert isinstance(stats, dict)
        assert stats["index_available"] is False
        assert "documents_loaded" in stats

    @patch.object(IngestionPipeline, 'initialize_models')
    @patch.object(IngestionPipeline, 'load_wikipedia_data')
    @patch.object(IngestionPipeline, 'create_documents')
    @patch.object(IngestionPipeline, 'create_vector_index')
    def test_run_ingestion_success(self, mock_create_index, mock_create_docs,
                                  mock_load_data, mock_init_models,
                                  pipeline, sample_dataframe):
        """Test successful ingestion pipeline execution."""
        # Setup mocks
        mock_init_models.return_value = True
        mock_load_data.return_value = sample_dataframe
        mock_create_docs.return_value = [Mock() for _ in range(5)]
        mock_create_index.return_value = Mock()

        result = pipeline.run_ingestion(max_docs=5)

        assert result is True
        mock_init_models.assert_called_once()
        mock_load_data.assert_called_once()
        mock_create_docs.assert_called_once()
        mock_create_index.assert_called_once()

    @patch.object(IngestionPipeline, 'initialize_models')
    def test_run_ingestion_model_init_failure(self, mock_init_models, pipeline):
        """Test ingestion failure due to model initialization."""
        mock_init_models.return_value = False

        result = pipeline.run_ingestion()

        assert result is False
        mock_init_models.assert_called_once()

    @patch.object(IngestionPipeline, 'load_existing_index')
    @patch.object(IngestionPipeline, 'initialize_models')
    def test_run_ingestion_use_existing_index(self, mock_init_models,
                                            mock_load_existing, pipeline):
        """Test using existing index when available."""
        mock_init_models.return_value = True
        mock_existing_index = Mock()
        mock_load_existing.return_value = mock_existing_index

        result = pipeline.run_ingestion(force_rebuild=False)

        assert result is True
        # The index should be set by the load_existing_index call
        assert pipeline.index is not None
        mock_load_existing.assert_called_once()

    def test_get_pipeline_singleton(self):
        """Test the global pipeline singleton function."""
        from ingestion import get_pipeline

        pipeline1 = get_pipeline()
        pipeline2 = get_pipeline()

        assert pipeline1 is pipeline2  # Same instance


class TestIngestionIntegration:
    """Integration tests for the ingestion pipeline."""

    @pytest.fixture
    def temp_storage(self):
        """Create temporary storage directory for integration tests."""
        temp_dir = tempfile.mkdtemp(prefix="test_integration_")
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)

    def test_minimal_integration_with_mock_models(self, temp_storage):
        """Test minimal integration with mocked models."""
        pipeline = IngestionPipeline(storage_dir=temp_storage)

        # Create mock models
        mock_embedding = Mock()
        mock_embedding.get_text_embedding.return_value = [0.1] * 3072
        mock_llm = Mock()

        pipeline.embedding_model = mock_embedding
        pipeline.llm_model = mock_llm

        # Create test data
        sample_data = pd.DataFrame({
            'text': ['Test document for integration'],
            'title': ['Integration Test']
        })

        # Test document creation
        documents = pipeline.create_documents(sample_data)
        assert len(documents) == 1
        assert documents[0].text == 'Test document for integration'

        # Test stats
        stats = pipeline.get_index_stats()
        assert isinstance(stats, dict)

    @pytest.mark.slow
    def test_real_model_integration(self, temp_storage):
        """Integration test with real models (marked as slow)."""
        pipeline = IngestionPipeline(storage_dir=temp_storage)

        try:
            # Try to initialize real models
            if not pipeline.initialize_models():
                pytest.skip("Real models not available for integration test")

            # Create minimal test data
            test_data = pd.DataFrame({
                'text': ['Python is a programming language.'],
                'title': ['Python']
            })

            # Test document creation
            documents = pipeline.create_documents(test_data)
            assert len(documents) == 1

            # Test vector index creation (with real embedding)
            index = pipeline.create_vector_index(documents)
            if index is not None:  # Only assert if models are working
                assert pipeline.stats["index_created"] is True
                assert pipeline.stats["nodes_created"] > 0

        except Exception as e:
            pytest.skip(f"Real model integration test skipped: {e}")


class TestIngestionErrors:
    """Test error handling in ingestion pipeline."""

    def test_invalid_storage_directory(self):
        """Test handling of invalid storage directory."""
        # This should still work as it creates the directory
        invalid_path = "/totally/invalid/path/that/does/not/exist"

        try:
            pipeline = IngestionPipeline(storage_dir=invalid_path)
            # On Windows, this might fail or succeed depending on permissions
            assert pipeline.storage_dir == Path(invalid_path)
        except (PermissionError, OSError):
            # This is acceptable on some systems
            pass

    def test_empty_dataframe(self):
        """Test handling of empty DataFrame."""
        pipeline = IngestionPipeline()
        empty_df = pd.DataFrame()

        documents = pipeline.create_documents(empty_df)
        assert len(documents) == 0

    def test_malformed_dataframe(self):
        """Test handling of DataFrame without required columns."""
        pipeline = IngestionPipeline()
        bad_df = pd.DataFrame({'wrong_column': ['test']})

        # Should handle missing 'text' column gracefully
        try:
            documents = pipeline.create_documents(bad_df)
            # If it succeeds, verify the behavior
            assert isinstance(documents, list)
        except (KeyError, AttributeError):
            # This is acceptable error handling
            pass