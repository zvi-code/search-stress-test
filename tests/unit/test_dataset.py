"""Unit tests for dataset handling."""

import pytest
from pathlib import Path


class TestDataset:
    """Test dataset loading and access."""
    
    def test_load_dataset(self, sample_dataset_path):
        """Test loading HDF5 dataset."""
        # TODO: Test successful load
        # TODO: Test metadata extraction
        raise NotImplementedError()
        
    def test_batch_iteration(self, sample_dataset_path):
        """Test batch iteration."""
        # TODO: Test different batch sizes
        # TODO: Test shuffle option
        raise NotImplementedError()
