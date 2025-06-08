#!/usr/bin/env python3
"""Debug script to test the fixes"""

import asyncio
import sys
sys.path.insert(0, 'src')

from tests.mocks import MockAsyncRedisPool, MockDataset
from src.valkey_stress_test.workload.query import QueryWorkload
from src.valkey_stress_test.workload.shrink import ShrinkWorkload
from unittest.mock import Mock
import numpy as np

async def test_query_workload():
    print("=== Testing Query Workload ===")
    workload = QueryWorkload()
    pool = MockAsyncRedisPool(Mock())
    await pool.initialize()
    
    client = await pool.get_client()
    print(f"Client created: {type(client)}")
    
    # First, let's test the mock search command directly
    import struct
    query_vector = np.random.random(10).astype(np.float32)
    vector_bytes = struct.pack(f"<{len(query_vector)}f", *query_vector)
    
    print("Testing mock search command directly...")
    try:
        mock_result = await client.execute_command(
            "FT.SEARCH", "test_index",
            "*=>[KNN 5 @vector $vec EF_RUNTIME 100]",
            "PARAMS", "2", "vec", vector_bytes,
            "RETURN", "1", "__vector_score",
            "LIMIT", "0", "5",
            "DIALECT", "2"
        )
        print(f"Mock search result: {mock_result}")
        print(f"Result type: {type(mock_result)}")
        print(f"Result length: {len(mock_result) if mock_result else 0}")
        
        if mock_result and len(mock_result) > 2:
            print(f"First key: {mock_result[1]} (type: {type(mock_result[1])})")
            print(f"First fields: {mock_result[2]} (type: {type(mock_result[2])})")
            if hasattr(mock_result[2], '__len__') and len(mock_result[2]) > 1:
                print(f"Score value: {mock_result[2][1]} (type: {type(mock_result[2][1])})")
    except Exception as e:
        print(f"Mock search failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test a single query execution
    query_vector = np.random.random(10).astype(np.float32)
    
    try:
        print("Executing KNN query...")
        result = await workload._execute_knn_query(
            client, query_vector, k=5, ef_runtime=100, index_name='test_index'
        )
        
        print(f"Query result: {result}")
        if result:
            keys, distances = result
            print(f"Keys: {keys}")
            print(f"Distances: {distances}")
            print("✓ Query execution successful")
        else:
            print("✗ Query returned None")
    except Exception as e:
        print(f"✗ Query failed with exception: {e}")
        import traceback
        traceback.print_exc()

async def test_shrink_workload():
    print("\n=== Testing Shrink Workload ===")
    workload = ShrinkWorkload()
    pool = MockAsyncRedisPool(Mock())
    await pool.initialize()
    
    client = await pool.get_client()
    print(f"Client created: {type(client)}")
    
    # Add some test keys
    print("Adding test keys...")
    for i in range(10):
        key = f"train_{i}"
        await client.hset(key, "vector", b"data")
        client.data[key] = b"data"  # Also add to regular data dict
    
    print("Keys added to mock client")
    
    # Test key scanning
    try:
        all_keys = await workload._get_all_keys(client, pattern="vec:*")
        print(f"Keys found by scan: {all_keys}")
        
        if all_keys:
            # Test deletion of first key
            test_key = all_keys[0]
            print(f"Testing deletion of: {test_key}")
            deleted = await client.delete(test_key)
            print(f"Delete result: {deleted}")
        else:
            print("✗ No keys found by scan")
            
    except Exception as e:
        print(f"✗ Shrink test failed with exception: {e}")
        import traceback
        traceback.print_exc()

async def main():
    await test_query_workload()
    await test_shrink_workload()

if __name__ == "__main__":
    asyncio.run(main())
