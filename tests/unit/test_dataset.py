"""Unit tests for dataset handling."""

import pytest
import numpy as np
from pathlib import Path
from valkey_stress_test.core.dataset import Dataset


class TestDataset:
    """Test dataset loading and access."""
    
    def test_load_dataset(self, sample_dataset_path):
        """Test loading HDF5 dataset."""
        # Test successful load
        dataset = Dataset(sample_dataset_path)
        info = dataset.load()
        
        # Test metadata extraction
        assert info.train_size > 0
        assert info.dimensions > 0
        assert info.test_size > 0
        
        # Verify data shapes
        train_vectors = dataset.get_train_vectors()
        assert train_vectors.shape[0] == info.train_size
        assert train_vectors.shape[1] == info.dimensions
        
        test_vectors = dataset.get_test_vectors()
        assert test_vectors.shape[0] == info.test_size
        assert test_vectors.shape[1] == info.dimensions
        
    def test_batch_iteration(self, sample_dataset_path):
        """Test batch iteration."""
        dataset = Dataset(sample_dataset_path)
        dataset.load()  # Load the dataset first
        
        # Test different batch sizes
        batch_size = 50
        batches = list(dataset.iterate_batches(batch_size=batch_size))
        
        assert len(batches) > 0
        
        # Check batch contents
        first_batch_vectors, first_batch_keys = batches[0]
        assert first_batch_vectors.shape[0] <= batch_size
        assert first_batch_vectors.shape[1] == dataset.info.dimensions
        assert len(first_batch_keys) == first_batch_vectors.shape[0]
        
        # Test shuffle option doesn't crash
        shuffled_batches = list(dataset.iterate_batches(batch_size=batch_size, shuffle=True))
        assert len(shuffled_batches) == len(batches)
