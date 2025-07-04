#!/usr/bin/env python3
"""
Concurrency Learning Module
Demonstrates threading, race conditions, and pipeline processing in VLSI-AI systems
"""

import sys
import os
import numpy as np
import time

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from hardware.simulation.hardware_sim_concurrent import run_concurrent_hardware_simulation
from hardware.simulation.race_condition_demo import demonstrate_race_condition
from hardware.simulation.pipeline_concurrent import run_pipeline_simulation
from hardware.simulation.hardware_sim import run_hardware_simulation

def main():
    print("CONCURRENCY LEARNING IN VLSI-AI PROJECT")
    print("=" * 50)
    
    # 1. Demonstrate race conditions
    print("\n1. RACE CONDITION DEMONSTRATION:")
    demonstrate_race_condition()
    
    # 2. Test concurrent sample processing
    print("\n2. CONCURRENT SAMPLE PROCESSING:")
    test_data = np.random.randn(8, 28, 28, 1)
    
    # Sequential processing (your original)
    print("\nSequential processing...")
    start_time = time.time()
    sequential_results = run_hardware_simulation(test_data)
    sequential_time = sequential_results['execution_time']
    
    # Concurrent processing
    print("\nConcurrent processing...")
    concurrent_results = run_concurrent_hardware_simulation(test_data, max_workers=4)
    concurrent_time = concurrent_results['execution_time']
    
    print(f"\nCOMPARISON RESULTS:")
    print(f"Sequential time: {sequential_time:.4f}s")
    print(f"Concurrent time: {concurrent_time:.4f}s")
    print(f"Speedup: {sequential_time / concurrent_time:.2f}x")
    print(f"Sequential operations: {sequential_results['operations_count']}")
    print(f"Concurrent operations: {concurrent_results['global_operation_count']}")
    
    # 3. Test pipeline processing
    print("\n3. PIPELINE PROCESSING:")
    pipeline_results = run_pipeline_simulation(test_data[:4])  # Smaller dataset for demo
    print(f"Pipeline processing time: {pipeline_results['execution_time']:.4f}s")
    print(f"MAC operations: {pipeline_results['mac_operations']}")
    print(f"ReLU operations: {pipeline_results['relu_operations']}")
    
    print(f"\nLEARNING SUMMARY:")
    print(f"* Race conditions: Shared variables need synchronization")
    print(f"* Threading: Can improve throughput but adds complexity")
    print(f"* Pipelines: Good for streaming data processing")
    print(f"* Locks: Essential for thread-safe operations")

if __name__ == "__main__":
    main()