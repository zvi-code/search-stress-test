#!/usr/bin/env python3
"""
Demo script showing advanced scenario execution and visualization capabilities.

This script demonstrates:
1. Running multiple interesting scenarios
2. Generating comprehensive visualizations
3. Analyzing memory patterns and performance correlations
4. Creating comparative analysis across scenarios

Usage:
    python demo_scenarios_and_visualizations.py [--run-scenarios] [--generate-visuals]
"""

import argparse
import asyncio
import logging
import subprocess
import sys
from pathlib import Path
import time
from typing import List, Dict, Any
import json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ScenarioDemo:
    """Demonstrates advanced scenario execution and visualization."""
    
    def __init__(self, base_dir: str = "."):
        self.base_dir = Path(base_dir)
        self.output_dir = self.base_dir / "output"
        self.viz_dir = self.base_dir / "visualizations"
        self.scenarios_dir = self.base_dir / "config" / "scenarios"
        
        # Ensure directories exist
        self.output_dir.mkdir(exist_ok=True)
        self.viz_dir.mkdir(exist_ok=True)
    
    def get_demo_scenarios(self) -> List[Dict[str, Any]]:
        """Get list of demo scenarios with their configurations."""
        return [
            {
                "name": "memory_fragmentation_analysis",
                "description": "Memory fragmentation testing with multiple grow-shrink cycles",
                "estimated_duration": "45 minutes",
                "memory_required": "16GB",
                "highlights": ["Memory fragmentation patterns", "Cleanup efficiency", "jemalloc behavior"]
            },
            {
                "name": "high_velocity_ingestion", 
                "description": "High-throughput ingestion stress testing",
                "estimated_duration": "30 minutes",
                "memory_required": "32GB",
                "highlights": ["Burst ingestion patterns", "Memory allocation efficiency", "GC pressure"]
            },
            {
                "name": "vector_expansion_memory_impact",
                "description": "Vector expansion feature memory analysis",
                "estimated_duration": "40 minutes", 
                "memory_required": "24GB",
                "highlights": ["Expansion overhead", "Query performance impact", "Memory efficiency"]
            },
            {
                "name": "production_simulation",
                "description": "Realistic production workload patterns",
                "estimated_duration": "60 minutes",
                "memory_required": "12GB", 
                "highlights": ["Daily usage patterns", "Mixed operations", "Business hours simulation"]
            },
            {
                "name": "cluster_memory_distribution",
                "description": "Cluster-wide memory distribution analysis",
                "estimated_duration": "60 minutes",
                "memory_required": "8GB per node",
                "highlights": ["Multi-node patterns", "Load balancing", "Node failure scenarios"]
            }
        ]
    
    def run_scenario(self, scenario_name: str, dry_run: bool = False) -> bool:
        """Run a specific scenario."""
        scenario_file = self.scenarios_dir / f"{scenario_name}.yaml"
        
        if not scenario_file.exists():
            logger.error(f"Scenario file not found: {scenario_file}")
            return False
        
        logger.info(f"{'[DRY RUN] ' if dry_run else ''}Running scenario: {scenario_name}")
        
        cmd = [
            "python", "-m", "valkey_stress_test.cli.main",
            "run", "scenario",
            "--scenario", str(scenario_file),
            "--output-dir", str(self.output_dir)
        ]
        
        if dry_run:
            cmd.append("--dry-run")
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.base_dir)
            
            if result.returncode == 0:
                logger.info(f"✓ Scenario {scenario_name} completed successfully")
                return True
            else:
                logger.error(f"✗ Scenario {scenario_name} failed: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Error running scenario {scenario_name}: {e}")
            return False
    
    def generate_visualizations(self, scenario_name: str) -> bool:
        """Generate visualizations for a scenario."""
        logger.info(f"Generating visualizations for {scenario_name}")
        
        cmd = [
            "python", "-m", "valkey_stress_test.cli.main",
            "visualize", "generate",
            scenario_name,
            "--output-dir", str(self.viz_dir),
            "--type", "all",
            "--format", "both"
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.base_dir)
            
            if result.returncode == 0:
                logger.info(f"✓ Visualizations for {scenario_name} generated successfully")
                return True
            else:
                logger.error(f"✗ Visualization generation failed for {scenario_name}: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Error generating visualizations for {scenario_name}: {e}")
            return False
    
    def create_comparative_analysis(self, scenario_names: List[str]):
        """Create comparative analysis across scenarios."""
        logger.info("Creating comparative analysis...")
        
        # Generate memory usage comparison
        cmd = [
            "python", "-m", "valkey_stress_test.cli.main", 
            "visualize", "compare",
            *scenario_names,
            "--output-dir", str(self.viz_dir),
            "--metric", "rss_mb",
            "--normalize"
        ]
        
        try:
            subprocess.run(cmd, check=True, cwd=self.base_dir)
            logger.info("✓ Comparative analysis generated")
        except Exception as e:
            logger.error(f"Error creating comparative analysis: {e}")
    
    def generate_demo_report(self, scenarios_run: List[str]):
        """Generate a comprehensive demo report."""
        report = {
            "demo_timestamp": time.time(),
            "scenarios_executed": scenarios_run,
            "output_files": {},
            "visualization_files": {},
            "summary": {}
        }
        
        # Collect output files
        for scenario in scenarios_run:
            metrics_file = self.output_dir / f"{scenario}_metrics.csv"
            phases_file = self.output_dir / f"{scenario}_phases.json"
            
            if metrics_file.exists():
                report["output_files"][scenario] = {
                    "metrics": str(metrics_file),
                    "phases": str(phases_file) if phases_file.exists() else None,
                    "metrics_size_mb": metrics_file.stat().st_size / (1024 * 1024)
                }
        
        # Collect visualization files
        for viz_file in self.viz_dir.glob("*.png"):
            scenario = viz_file.stem.split("_")[0]
            if scenario not in report["visualization_files"]:
                report["visualization_files"][scenario] = []
            report["visualization_files"][scenario].append(str(viz_file))
        
        for viz_file in self.viz_dir.glob("*.html"):
            scenario = viz_file.stem.split("_")[0] 
            if scenario not in report["visualization_files"]:
                report["visualization_files"][scenario] = []
            report["visualization_files"][scenario].append(str(viz_file))
        
        # Generate summary
        report["summary"] = {
            "total_scenarios": len(scenarios_run),
            "total_output_files": len(report["output_files"]),
            "total_visualizations": sum(len(files) for files in report["visualization_files"].values()),
            "recommendations": [
                "View interactive dashboards (.html files) for comprehensive analysis",
                "Check memory_fragmentation_analysis for fragmentation patterns",
                "Review high_velocity_ingestion for throughput optimization insights",
                "Analyze production_simulation for realistic workload patterns"
            ]
        }
        
        # Save report
        report_file = self.base_dir / "demo_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"✓ Demo report saved to {report_file}")
        return report
    
    def print_demo_summary(self, scenarios: List[Dict], completed: List[str]):
        """Print a summary of the demo."""
        print("\n" + "="*80)
        print("VALKEY STRESS TEST - ADVANCED SCENARIOS DEMO SUMMARY")
        print("="*80)
        
        print(f"\n📊 SCENARIOS OVERVIEW:")
        for scenario in scenarios:
            status = "✓ COMPLETED" if scenario["name"] in completed else "○ AVAILABLE"
            print(f"  {status} {scenario['name']}")
            print(f"    Description: {scenario['description']}")
            print(f"    Duration: {scenario['estimated_duration']}")
            print(f"    Memory: {scenario['memory_required']}")
            print(f"    Highlights: {', '.join(scenario['highlights'])}")
            print()
        
        if completed:
            print(f"\n📈 VISUALIZATIONS GENERATED:")
            viz_files = list(self.viz_dir.glob("*"))
            print(f"  Total files: {len(viz_files)}")
            print(f"  Interactive dashboards: {len(list(self.viz_dir.glob('*dashboard.html')))}")
            print(f"  Memory analysis: {len(list(self.viz_dir.glob('*memory_analysis.png')))}")
            print(f"  Performance correlations: {len(list(self.viz_dir.glob('*performance_correlation.png')))}")
            print(f"  Phase heatmaps: {len(list(self.viz_dir.glob('*phase_heatmap.png')))}")
            
            print(f"\n📁 OUTPUT LOCATIONS:")
            print(f"  Metrics data: {self.output_dir}")
            print(f"  Visualizations: {self.viz_dir}")
            print(f"  Demo report: {self.base_dir}/demo_report.json")
            
            print(f"\n🚀 NEXT STEPS:")
            print(f"  1. Open interactive dashboards in your browser:")
            for html_file in self.viz_dir.glob("*dashboard.html"):
                print(f"     file://{html_file.absolute()}")
            print(f"  2. Review static visualizations in: {self.viz_dir}")
            print(f"  3. Analyze raw metrics data in: {self.output_dir}")
            print(f"  4. Check demo_report.json for detailed summary")
        
        print("\n" + "="*80)


def main():
    parser = argparse.ArgumentParser(description="Demo advanced scenarios and visualizations")
    parser.add_argument("--run-scenarios", action="store_true", 
                       help="Actually run scenarios (default: dry run only)")
    parser.add_argument("--generate-visuals", action="store_true",
                       help="Generate visualizations for existing data")
    parser.add_argument("--scenarios", nargs="+", 
                       help="Specific scenarios to run (default: all demo scenarios)")
    parser.add_argument("--quick-demo", action="store_true",
                       help="Run quick demo with smaller scenarios")
    
    args = parser.parse_args()
    
    demo = ScenarioDemo()
    all_scenarios = demo.get_demo_scenarios()
    
    # Select scenarios to run
    if args.scenarios:
        scenarios = [s for s in all_scenarios if s["name"] in args.scenarios]
    elif args.quick_demo:
        # Quick demo with smaller/faster scenarios
        scenarios = [s for s in all_scenarios if s["name"] in [
            "production_simulation", 
            "vector_expansion_memory_impact"
        ]]
    else:
        scenarios = all_scenarios
    
    completed_scenarios = []
    
    # Print initial summary
    demo.print_demo_summary(all_scenarios, [])
    
    if args.run_scenarios:
        print(f"\n🚀 RUNNING {len(scenarios)} SCENARIOS...")
        print("This may take several hours depending on your system.")
        
        for scenario in scenarios:
            print(f"\n▶️  Starting {scenario['name']}...")
            if demo.run_scenario(scenario["name"], dry_run=False):
                completed_scenarios.append(scenario["name"])
            else:
                print(f"❌ Failed to run {scenario['name']}")
    
    elif args.generate_visuals:
        # Find existing scenario data
        for scenario in scenarios:
            metrics_file = demo.output_dir / f"{scenario['name']}_metrics.csv"
            if metrics_file.exists():
                completed_scenarios.append(scenario["name"])
    
    else:
        # Dry run mode
        print(f"\n🔍 DRY RUN MODE - Testing {len(scenarios)} scenarios...")
        for scenario in scenarios:
            if demo.run_scenario(scenario["name"], dry_run=True):
                print(f"✓ {scenario['name']} - scenario validated")
    
    # Generate visualizations if requested or scenarios were run
    if (args.run_scenarios or args.generate_visuals) and completed_scenarios:
        print(f"\n📊 GENERATING VISUALIZATIONS for {len(completed_scenarios)} scenarios...")
        
        for scenario_name in completed_scenarios:
            demo.generate_visualizations(scenario_name)
        
        # Create comparative analysis if multiple scenarios
        if len(completed_scenarios) > 1:
            demo.create_comparative_analysis(completed_scenarios)
        
        # Generate demo report
        demo.generate_demo_report(completed_scenarios)
    
    # Print final summary
    demo.print_demo_summary(all_scenarios, completed_scenarios)
    
    if not args.run_scenarios and not args.generate_visuals:
        print(f"\n💡 TIP: Use --run-scenarios to actually execute scenarios")
        print(f"       Use --generate-visuals to create charts from existing data")
        print(f"       Use --quick-demo for a faster demonstration")


if __name__ == "__main__":
    main()
