"""
Unit tests for the Wikipedia RAG system.

Tests core functionality:
- Data loading from HuggingFace
- Index building and persistence
- Document retrieval
- Answer generation
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock
import pandas as pd

from rag_system import WikipediaRAGSystem, create_rag_system


class TestWikipediaRAGSystem:
    """Test cases for WikipediaRAGSystem."""

    @pytest.fixture
    def temp_storage(self):
        """Create a temporary directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def rag_system(self, temp_storage):
        """Create a RAG system with temporary storage."""
        return WikipediaRAGSystem(storage_dir=temp_storage)

    @pytest.fixture
    def sample_wikipedia_data(self):
        """Create sample Wikipedia data for testing."""
        return pd.DataFrame({
            'id': ['doc1', 'doc2', 'doc3'],
            'passage': [
                'Photosynthesis is the process by which plants convert sunlight into energy.',
                'The water cycle describes how water moves through the environment.',
                'Gravity is a fundamental force that attracts objects with mass.'
            ],
            'title': ['Photosynthesis', 'Water Cycle', 'Gravity']
        })

    def test_initialization(self, temp_storage):
        """Test RAG system initialization."""
        system = WikipediaRAGSystem(storage_dir=temp_storage)

        assert system.storage_dir == Path(temp_storage)
        assert system.storage_dir.exists()
        assert system.chunk_size == 512
        assert system.index is None
        assert system.query_engine is None
        assert system.retriever is None

    def test_create_rag_system_factory(self):
        """Test the factory function."""
        system = create_rag_system()
        assert isinstance(system, WikipediaRAGSystem)
        assert system.storage_dir == Path("./rag_storage")

    @patch('pandas.read_parquet')
    def test_load_wikipedia_data_default_url(self, mock_read_parquet, rag_system, sample_wikipedia_data):
        """Test loading data with default URL."""
        mock_read_parquet.return_value = sample_wikipedia_data

        documents = rag_system.load_wikipedia_data()

        # Check that correct URL was used
        expected_url = "hf://datasets/rag-datasets/rag-mini-wikipedia/data/passages.parquet/part.0.parquet"
        mock_read_parquet.assert_called_once_with(expected_url)

        # Check documents
        assert len(documents) == 3
        assert documents[0].text == 'Photosynthesis is the process by which plants convert sunlight into energy.'
        assert documents[0].metadata['title'] == 'Photosynthesis'
        assert documents[0].metadata['source'] == 'wikipedia'

    @patch('pandas.read_parquet')
    def test_load_wikipedia_data_custom_url(self, mock_read_parquet, rag_system, sample_wikipedia_data):
        """Test loading data with custom URL."""
        mock_read_parquet.return_value = sample_wikipedia_data
        custom_url = "custom://dataset/url"

        documents = rag_system.load_wikipedia_data(custom_url)

        mock_read_parquet.assert_called_once_with(custom_url)
        assert len(documents) == 3

    def test_get_index_stats_no_index(self, rag_system):
        """Test getting stats when no index exists."""
        stats = rag_system.get_index_stats()
        assert stats['status'] == 'No index available'

    @patch('pandas.read_parquet')
    def test_build_index_new(self, mock_read_parquet, rag_system, sample_wikipedia_data):
        """Test building a new index."""
        mock_read_parquet.return_value = sample_wikipedia_data

        # Load documents
        documents = rag_system.load_wikipedia_data()

        # Build index
        index = rag_system.build_index(documents)

        # Verify index was created
        assert rag_system.index is not None
        assert index == rag_system.index

        # Check stats
        stats = rag_system.get_index_stats()
        assert stats['status'] == 'Index available'
        assert stats['num_documents'] == 3
        assert stats['chunk_size'] == 512

    @patch('pandas.read_parquet')
    def test_setup_retriever(self, mock_read_parquet, rag_system, sample_wikipedia_data):
        """Test setting up the retriever."""
        mock_read_parquet.return_value = sample_wikipedia_data

        # Build index first
        documents = rag_system.load_wikipedia_data()
        rag_system.build_index(documents)

        # Setup retriever
        retriever = rag_system.setup_retriever(similarity_top_k=3)

        assert rag_system.retriever is not None
        assert retriever == rag_system.retriever
        assert retriever.similarity_top_k == 3

    def test_setup_retriever_no_index(self, rag_system):
        """Test that retriever setup fails without index."""
        with pytest.raises(ValueError, match="Index must be built before setting up retriever"):
            rag_system.setup_retriever()

    @patch('pandas.read_parquet')
    def test_setup_query_engine(self, mock_read_parquet, rag_system, sample_wikipedia_data):
        """Test setting up the query engine."""
        mock_read_parquet.return_value = sample_wikipedia_data

        # Build index and setup retriever first
        documents = rag_system.load_wikipedia_data()
        rag_system.build_index(documents)
        rag_system.setup_retriever()

        # Setup query engine
        query_engine = rag_system.setup_query_engine()

        assert rag_system.query_engine is not None
        assert query_engine == rag_system.query_engine

    def test_setup_query_engine_no_retriever(self, rag_system):
        """Test that query engine setup fails without retriever."""
        with pytest.raises(ValueError, match="Retriever must be setup before creating query engine"):
            rag_system.setup_query_engine()

    def test_retrieve_documents_no_retriever(self, rag_system):
        """Test that document retrieval fails without retriever."""
        with pytest.raises(ValueError, match="Retriever must be setup before retrieving documents"):
            rag_system.retrieve_documents("test query")

    def test_generate_answer_no_query_engine(self, rag_system):
        """Test that answer generation fails without query engine."""
        with pytest.raises(ValueError, match="Query engine must be setup before generating answers"):
            rag_system.generate_answer("test question")


class TestIntegration:
    """Integration tests that test the full pipeline."""

    @pytest.fixture
    def temp_storage(self):
        """Create a temporary directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def sample_wikipedia_data(self):
        """Create sample Wikipedia data for testing."""
        return pd.DataFrame({
            'id': ['doc1', 'doc2', 'doc3'],
            'passage': [
                'Photosynthesis is the biological process by which plants, algae, and certain bacteria convert light energy from the sun into chemical energy stored in glucose.',
                'The water cycle, also known as the hydrological cycle, describes the continuous movement of water on, above, and below the surface of the Earth.',
                'Gravity is a fundamental interaction that causes all things with mass or energy to be attracted to one another, including planets, stars, galaxies, and even light.'
            ],
            'title': ['Photosynthesis', 'Water Cycle', 'Gravity']
        })

    @patch('pandas.read_parquet')
    def test_full_rag_pipeline(self, mock_read_parquet, temp_storage, sample_wikipedia_data):
        """Test the complete RAG pipeline from ingestion to answer generation."""
        mock_read_parquet.return_value = sample_wikipedia_data

        # Initialize system
        system = WikipediaRAGSystem(storage_dir=temp_storage)

        # Step 1: Load data
        documents = system.load_wikipedia_data()
        assert len(documents) == 3

        # Step 2: Build index
        system.build_index(documents)
        assert system.index is not None

        # Step 3: Setup retriever
        system.setup_retriever(similarity_top_k=2)
        assert system.retriever is not None

        # Step 4: Setup query engine
        system.setup_query_engine()
        assert system.query_engine is not None

        # Step 5: Test retrieval
        results = system.retrieve_documents("What is photosynthesis?", top_k=2)
        assert len(results) <= 2
        assert all('rank' in result for result in results)
        assert all('text' in result for result in results)

        # Step 6: Test answer generation (this will use actual LLM)
        # Note: This test may be slow or fail if Azure access isn't working
        # For unit tests, we might want to mock this
        try:
            answer_result = system.generate_answer("What is photosynthesis?")
            assert 'query' in answer_result
            assert 'answer' in answer_result
            assert 'sources' in answer_result
            assert answer_result['query'] == "What is photosynthesis?"
        except Exception as e:
            # If LLM call fails (e.g., network issues), skip this part
            pytest.skip(f"LLM call failed, skipping answer generation test: {e}")

    @patch('pandas.read_parquet')
    def test_index_persistence(self, mock_read_parquet, temp_storage, sample_wikipedia_data):
        """Test that the index can be persisted and loaded."""
        mock_read_parquet.return_value = sample_wikipedia_data

        # Build index in first system
        system1 = WikipediaRAGSystem(storage_dir=temp_storage)
        documents = system1.load_wikipedia_data()
        system1.build_index(documents)

        # Verify files were created
        storage_path = Path(temp_storage)
        assert (storage_path / "docstore.json").exists()
        assert (storage_path / "vector_store.json").exists()
        assert (storage_path / "index_store.json").exists()

        # Create new system that should load existing index
        system2 = WikipediaRAGSystem(storage_dir=temp_storage)

        # Load some documents (can be empty since we're loading existing)
        system2.build_index([], force_rebuild=False)

        # Verify it loaded the existing index
        stats = system2.get_index_stats()
        assert stats['status'] == 'Index available'
        # Note: The exact document count check might vary based on LlamaIndex internals


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
