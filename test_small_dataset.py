import numpy as np
import tempfile
from pathlib import Path

# Create a small test dataset
test_vectors = np.random.rand(100, 128).astype(np.float32)
test_file = Path(tempfile.gettempdir()) / "test_vectors.npy"
np.save(test_file, test_vectors)
print(f"Created test dataset: {test_file}")
print(f"Shape: {test_vectors.shape}")
