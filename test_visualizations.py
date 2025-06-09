#!/usr/bin/env python3
"""
Test script to validate scenario configurations and test visualization system
"""
import os
import sys
import json
import yaml
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

from valkey_stress_test.visualization.advanced_visualizer import AdvancedVisualizer

def create_sample_metrics_data():
    """Create sample metrics data that matches our scenario structure"""
    
    # Sample data for 2 hours of testing with phases
    timestamps = []
    start_time = datetime.now() - timedelta(hours=2)
    
    # Create timestamps every 10 seconds
    for i in range(720):  # 2 hours * 3600 seconds / 10 seconds
        timestamps.append(start_time + timedelta(seconds=i * 10))
    
    # Convert to UNIX timestamps for compatibility
    unix_timestamps = [(ts - datetime(1970, 1, 1)).total_seconds() for ts in timestamps]
    
    # Define phases with their characteristics
    phases = [
        {"name": "baseline", "start": 0, "end": 120, "type": "baseline"},
        {"name": "ramp_up", "start": 120, "end": 240, "type": "ramp_up"},
        {"name": "stress_phase_1", "start": 240, "end": 420, "type": "stress"},
        {"name": "cooldown_1", "start": 420, "end": 480, "type": "cooldown"},
        {"name": "stress_phase_2", "start": 480, "end": 600, "type": "stress"},
        {"name": "final_cooldown", "start": 600, "end": 720, "type": "cooldown"}
    ]
    
    # Generate realistic memory usage patterns
    rss_mb = []
    active_mb = []
    resident_mb = []
    allocated_mb = []
    fragmentation_ratio = []
    qps = []  # queries per second
    ips = []  # inserts per second
    dps = []  # deletes per second
    p50_ms = []
    p95_ms = []
    p99_ms = []
    
    for i, ts in enumerate(unix_timestamps):
        # Determine current phase
        current_phase = None
        for phase in phases:
            if phase["start"] <= i < phase["end"]:
                current_phase = phase
                break
        
        if not current_phase:
            current_phase = phases[-1]
        
        # Generate data based on phase type
        if current_phase["type"] == "baseline":
            mem_base = 500 + np.random.normal(0, 10)
            ops_base = 1000 + np.random.normal(0, 50)
            lat_base = 1.0 + np.random.normal(0, 0.1)
        elif current_phase["type"] == "ramp_up":
            progress = (i - current_phase["start"]) / (current_phase["end"] - current_phase["start"])
            mem_base = 500 + progress * 500 + np.random.normal(0, 20)
            ops_base = 1000 + progress * 4000 + np.random.normal(0, 100)
            lat_base = 1.0 + progress * 2.0 + np.random.normal(0, 0.2)
        elif current_phase["type"] == "stress":
            mem_base = 1000 + np.random.normal(0, 50)
            ops_base = 5000 + np.random.normal(0, 200)
            lat_base = 3.0 + np.random.normal(0, 0.5)
        else:  # cooldown
            progress = (i - current_phase["start"]) / (current_phase["end"] - current_phase["start"])
            mem_base = 1000 - progress * 400 + np.random.normal(0, 30)
            ops_base = 5000 - progress * 3000 + np.random.normal(0, 150)
            lat_base = 3.0 - progress * 1.5 + np.random.normal(0, 0.3)
        
        # Memory metrics
        rss_mb.append(max(0, mem_base))
        active_mb.append(max(0, mem_base * 0.8))
        resident_mb.append(max(0, mem_base * 0.9))
        allocated_mb.append(max(0, mem_base * 1.1))
        fragmentation_ratio.append(max(1.0, 1.0 + np.random.normal(0, 0.1)))
        
        # Performance metrics
        total_ops = max(0, ops_base)
        qps.append(total_ops * 0.6)  # 60% queries
        ips.append(total_ops * 0.3)  # 30% inserts
        dps.append(total_ops * 0.1)  # 10% deletes
        
        p50_ms.append(max(0.1, lat_base))
        p95_ms.append(max(0.1, lat_base * 2))
        p99_ms.append(max(0.1, lat_base * 3))
    
    # Create DataFrame with expected column names
    df = pd.DataFrame({
        'timestamp': unix_timestamps,
        'phase': [get_phase_name(i, phases) for i in range(len(unix_timestamps))],
        'rss_mb': rss_mb,
        'active_mb': active_mb,
        'resident_mb': resident_mb,
        'allocated_mb': allocated_mb,
        'fragmentation_ratio': fragmentation_ratio,
        'qps': qps,
        'ips': ips,
        'dps': dps,
        'p50_ms': p50_ms,
        'p95_ms': p95_ms,
        'p99_ms': p99_ms
    })
    
    return df, phases

def get_phase_name(index, phases):
    """Get phase name for a given index"""
    for phase in phases:
        if phase["start"] <= index < phase["end"]:
            return phase["name"]
    return phases[-1]["name"]

def test_scenario_configs():
    """Test that all scenario YAML files are valid"""
    print("🔧 Testing scenario configurations...")
    
    scenarios_dir = project_root / "config" / "scenarios"
    if not scenarios_dir.exists():
        print(f"❌ Scenarios directory not found: {scenarios_dir}")
        return False
    
    valid_scenarios = []
    
    for yaml_file in scenarios_dir.glob("*.yaml"):
        try:
            with open(yaml_file) as f:
                config = yaml.safe_load(f)
            
            # Basic validation
            required_keys = ['name', 'description']
            missing_keys = []
            for key in required_keys:
                if key not in config:
                    missing_keys.append(key)
            
            if missing_keys:
                print(f"❌ {yaml_file.name}: Missing required keys: {missing_keys}")
            else:
                print(f"✅ {yaml_file.name}: Valid configuration")
                valid_scenarios.append(yaml_file.name)
            
        except Exception as e:
            print(f"❌ {yaml_file.name}: Error - {e}")
    
    print(f"\n📊 Valid scenarios: {len(valid_scenarios)}")
    return len(valid_scenarios) > 0

def test_visualizations():
    """Test the visualization system with sample data"""
    print("\n🎨 Testing visualization system...")
    
    # Create sample data
    df, phases = create_sample_metrics_data()
    
    # Create output directory
    output_dir = project_root / "test_outputs"
    output_dir.mkdir(exist_ok=True)
    
    # Initialize visualizer
    visualizer = AdvancedVisualizer()
    
    try:
        # Create MetricsData object
        from valkey_stress_test.visualization.advanced_visualizer import MetricsData
        
        # Create a temporary CSV file for testing
        csv_path = output_dir / "temp_metrics.csv"
        df.to_csv(csv_path, index=False)
        
        # Load data into MetricsData
        metrics_data = MetricsData(csv_file=str(csv_path))
        
        # Debug: Print what was loaded
        print(f"   Memory DataFrame columns: {list(metrics_data.memory_df.columns) if not metrics_data.memory_df.empty else 'Empty'}")
        print(f"   Performance DataFrame columns: {list(metrics_data.performance_df.columns) if not metrics_data.performance_df.empty else 'Empty'}")
        print(f"   Memory shape: {metrics_data.memory_df.shape}")
        print(f"   Performance shape: {metrics_data.performance_df.shape}")
        
        # Test 1: Comprehensive Dashboard
        print("📈 Generating comprehensive dashboard...")
        dashboard_fig = visualizer.create_comprehensive_dashboard(
            metrics_data, 
            "test_scenario"
        )
        print(f"✅ Dashboard generated successfully")
        
        # Test 2: Memory Phase Analysis
        print("🧠 Generating memory phase analysis...")
        memory_fig = visualizer.create_memory_phase_analysis(
            metrics_data,
            "test_scenario"
        )
        print(f"✅ Memory analysis generated successfully")
        
        # Test 3: Performance Correlation
        print("⚡ Generating performance correlation...")
        perf_fig = visualizer.create_performance_correlation_analysis(
            metrics_data,
            "test_scenario"
        )
        print(f"✅ Performance correlation generated successfully")
        
        # Test 4: Phase Comparison
        print("📊 Generating phase comparison...")
        phase_fig = visualizer.create_phase_comparison_heatmap(
            metrics_data,
            "test_scenario"
        )
        print(f"✅ Phase comparison generated successfully")
        
        # Clean up temp file
        csv_path.unlink()
        
        return True
        
    except Exception as e:
        print(f"❌ Visualization test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function"""
    print("🚀 Starting Valkey Stress Test Validation")
    print("=" * 50)
    
    # Test scenario configurations
    scenario_test = test_scenario_configs()
    
    # Test visualizations
    viz_test = test_visualizations()
    
    print("\n" + "=" * 50)
    print("📋 Test Summary:")
    print(f"   Scenario Configs: {'✅ PASS' if scenario_test else '❌ FAIL'}")
    print(f"   Visualizations:   {'✅ PASS' if viz_test else '❌ FAIL'}")
    
    if scenario_test and viz_test:
        print("\n🎉 All tests passed! The system is ready for use.")
        print("\n📁 Check the 'test_outputs' directory for sample visualizations.")
    else:
        print("\n⚠️  Some tests failed. Please check the errors above.")
    
    return scenario_test and viz_test

if __name__ == "__main__":
    sys.exit(0 if main() else 1)
