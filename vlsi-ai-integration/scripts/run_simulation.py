#!/usr/bin/env python3
import os
import sys
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from hardware.simulation.hardware_sim import run_hardware_simulation
from benchmarking.power_estimator import estimate_power

def main():
    print("Running Hardware Simulation...")
    
    # Check if test data exists
    input_data_path = 'data/mnist/x_test.npy'
    if not os.path.exists(input_data_path):
        print("❌ Test data not found. Please run 'python scripts/train_model.py' first")
        sys.exit(1)
    
    # Load test data
    print("Loading test data...")
    input_data = np.load(input_data_path)
    
    # Use first 5 samples for realistic simulation
    test_samples = input_data[:5]
    print(f"Running simulation on {len(test_samples)} samples...")
    
    # Run hardware simulation
    simulation_results = run_hardware_simulation(test_samples)
    
    # Print detailed results
    print("\n" + "="*60)
    print("DETAILED HARDWARE SIMULATION RESULTS")
    print("="*60)
    
    print(f"\n📊 EXECUTION METRICS:")
    print(f"   Software Simulation Time:    {simulation_results['execution_time']:.4f} seconds")
    print(f"   Theoretical Hardware Time:   {simulation_results.get('theoretical_hw_time', 'N/A'):.6f} seconds")
    print(f"   Simulated Clock Cycles:      {simulation_results['simulated_cycles']:,}")
    print(f"   Clock Frequency:             {simulation_results.get('clock_frequency_mhz', 'N/A'):.1f} MHz")
    
    print(f"\n⚡ OPERATION BREAKDOWN:")
    print(f"   Total Operations:            {simulation_results['operations_count']:,}")
    print(f"   MAC Operations:              {simulation_results.get('mac_operations', 'N/A'):,}")
    print(f"   ReLU Operations:             {simulation_results.get('relu_operations', 'N/A'):,}")
    
    print(f"\n🔋 POWER ANALYSIS:")
    total_power = simulation_results['estimated_power']
    print(f"   Total Estimated Power:       {total_power:.6f} Watts ({total_power*1000:.3f} mW)")
    
    if 'power_breakdown' in simulation_results:
        breakdown = simulation_results['power_breakdown']
        print(f"   Base Power:                  {breakdown['base_power_W']*1000:.3f} mW")
        print(f"   Dynamic Power:               {breakdown['dynamic_power_W']*1000000:.3f} µW")
        print(f"   MAC Power:                   {breakdown['mac_power_W']*1000000:.3f} µW")
        print(f"   ReLU Power:                  {breakdown['relu_power_W']*1000000:.3f} µW")
    
    print(f"\n🚀 PERFORMANCE METRICS:")
    print(f"   Throughput:                  {simulation_results['throughput']:.2f} samples/sec")
    print(f"   Results Shape:               {simulation_results['results'].shape}")
    
    # Hardware vs Software comparison
    if 'theoretical_hw_time' in simulation_results:
        hw_speedup = simulation_results['execution_time'] / simulation_results['theoretical_hw_time']
        print(f"   Hardware Speedup Potential:  {hw_speedup:.1f}x faster")
    
    print("="*60)
    
    print(f"\n💡 INTERPRETATION:")
    if total_power < 0.001:
        print("   ✅ Very low power consumption - suitable for mobile/edge devices")
    elif total_power < 0.01:
        print("   ✅ Low power consumption - good for battery-powered devices")
    elif total_power < 0.1:
        print("   ⚠️  Moderate power consumption")
    else:
        print("   ❗ High power consumption - may need optimization")

if __name__ == "__main__":
    main()