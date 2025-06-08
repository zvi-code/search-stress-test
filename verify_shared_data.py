#!/usr/bin/env python3
"""
Verify that the mock clients properly share data.
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from tests.mocks import MockConnectionManager, MockRedisClient
from valkey_stress_test.core import ConnectionConfig

async def test_shared_data():
    """Test that data is shared across pools and clients."""
    print("Testing mock data sharing...")
    
    # Create connection manager with 2 pools
    config = ConnectionConfig()
    manager = MockConnectionManager(config, n_pools=2)
    await manager.initialize()
    
    # Get clients from different pools
    pool1 = manager.get_pool(0)
    pool2 = manager.get_pool(1)
    
    client1 = await pool1.get_client()
    client2 = await pool2.get_client()
    
    print("\n1. Testing HSET/HGET across clients...")
    # Set data with client1
    await client1.hset("vec:test_1", "vector", b"data1")
    await client1.hset("vec:test_2", "vector", b"data2")
    
    # Check if client2 can see it
    data1 = await client2.hget("vec:test_1", "vector")
    data2 = await client2.hget("vec:test_2", "vector")
    
    assert data1 == b"data1", f"Expected b'data1', got {data1}"
    assert data2 == b"data2", f"Expected b'data2', got {data2}"
    print("✅ Data visible across clients")
    
    print("\n2. Testing SCAN across clients...")
    # Scan with client2
    cursor, keys = await client2.scan(0, match="vec:*", count=100)
    
    print(f"Found {len(keys)} keys: {keys}")
    assert len(keys) >= 2, f"Expected at least 2 keys, found {len(keys)}"
    assert b"vec:test_1" in keys or b"vec:test_2" in keys, "Expected keys not found"
    print("✅ SCAN works correctly")
    
    print("\n3. Testing DELETE across clients...")
    # Delete with client1
    deleted = await client1.delete("vec:test_1")
    assert deleted == 1, f"Expected 1 deletion, got {deleted}"
    
    # Verify with client2
    data = await client2.hget("vec:test_1", "vector")
    assert data is None, "Key should be deleted"
    print("✅ DELETE visible across clients")
    
    print("\n✅ All data sharing tests passed!")
    return True

async def main():
    try:
        success = await test_shared_data()
        if success:
            print("\n🎉 Mock data sharing is working correctly!")
            print("\nYou can now run the full test suite:")
            print("  pytest -m unit -v")
            return 0
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))