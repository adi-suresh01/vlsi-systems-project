#!/usr/bin/env python3
"""
C++ Hardware Simulation Benchmarking
Compares CPU, TensorFlow Lite, Python hardware simulation, and C++ hardware simulation
"""

import sys
import os
import time
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from ai_models.model_utils import load_model
from ai_models.tflite_converter import run_tflite_inference
from hardware.simulation.hardware_sim import run_hardware_simulation

def main():
    print("VLSI-AI INTEGRATION WITH C++ ACCELERATION")
    print("=" * 60)
    
    # Load test data
    print("Loading test data...")
    test_data_path = 'data/mnist/x_test.npy'
    if not os.path.exists(test_data_path):
        print("ERROR: Test data not found! Run: python scripts/train_model.py")
        return
    
    input_data = np.load(test_data_path)
    test_samples = input_data[:10]
    
    print(f"Processing {len(test_samples)} samples...")
    
    # 1. Python Hardware Simulation
    print("\n1. Python Hardware Simulation:")
    python_results = run_hardware_simulation(test_samples)
    python_time = python_results['execution_time']
    
    # 2. C++ Hardware Simulation  
    print("\n2. C++ Hardware Simulation:")
    try:
        from hardware.cpp.cpp_interface import run_cpp_hardware_simulation
        cpp_results = run_cpp_hardware_simulation(test_samples)
        cpp_time = cpp_results['execution_time']
        cpp_available = True
    except Exception as e:
        print(f"ERROR: C++ simulation failed: {e}")
        print("To fix:")
        print("   cd src/hardware/cpp")
        print("   chmod +x compile.sh")
        print("   ./compile.sh")
        cpp_available = False
        cpp_time = float('inf')
    
    # 3. CPU Inference (for comparison)
    print("\n3. CPU Inference:")
    try:
        model = load_model('data/models/mnist_cnn.h5')
        start_time = time.time()
        cpu_predictions = model.predict(test_samples, verbose=0)
        cpu_time = time.time() - start_time
        cpu_available = True
    except Exception as e:
        print(f"ERROR: CPU inference failed: {e}")
        print("Run: python scripts/train_model.py")
        cpu_available = False
        cpu_time = float('inf')
    
    # 4. TFLite Inference
    print("\n4. TFLite Inference:")
    try:
        start_time = time.time()
        tflite_results = run_tflite_inference(test_samples, 'data/models/mnist_cnn.tflite')
        tflite_time = time.time() - start_time
        tflite_available = True
    except Exception as e:
        print(f"ERROR: TFLite inference failed: {e}")
        tflite_available = False
        tflite_time = float('inf')
    
    # Results
    print(f"\nCOMPREHENSIVE PERFORMANCE COMPARISON:")
    print(f"=" * 50)
    print(f"Python Hardware:     {python_time:.6f}s ({python_time*1000:.2f}ms)")
    
    if cpp_available:
        print(f"C++ Hardware:        {cpp_time:.6f}s ({cpp_time*1000:.2f}ms)")
    else:
        print(f"C++ Hardware:        Not available (compile first)")
    
    if cpu_available:
        print(f"CPU Inference:       {cpu_time:.6f}s ({cpu_time*1000:.2f}ms)")
    else:
        print(f"CPU Inference:       Not available (train model first)")
    
    if tflite_available:
        print(f"TFLite Inference:    {tflite_time:.6f}s ({tflite_time*1000:.2f}ms)")
    else:
        print(f"TFLite Inference:    Not available")
    
    if cpp_available:
        print(f"\nSPEEDUP ANALYSIS:")
        print(f"C++ vs Python:       {python_time/cpp_time:.2f}x faster")
        if cpu_available:
            print(f"C++ vs CPU:          {cpu_time/cpp_time:.2f}x faster") 
        if tflite_available:
            print(f"C++ vs TFLite:       {tflite_time/cpp_time:.2f}x faster")
        
        print(f"\nOPERATION COMPARISON:")
        print(f"Python MAC ops:      {python_results['mac_operations']:,}")
        print(f"C++ MAC ops:         {cpp_results['mac_operations']:,}")
        print(f"Operations match:    {python_results['mac_operations'] == cpp_results['mac_operations']}")
        
        print(f"\nPOWER COMPARISON:")
        print(f"Python power:        {python_results['estimated_power']*1000:.3f} mW")
        print(f"C++ power:           {cpp_results['power_consumption']*1000:.3f} mW")
        
        print(f"\nTHROUGHPUT:")
        print(f"Python throughput:   {python_results['throughput']:.1f} samples/sec")
        print(f"C++ throughput:      {cpp_results['throughput']:.1f} samples/sec")
    
    print(f"\nSETUP INSTRUCTIONS:")
    if not cpp_available:
        print("To enable C++:")
        print("  brew install gcc")
        print("  cd src/hardware/cpp")
        print("  chmod +x compile.sh")
        print("  ./compile.sh")

if __name__ == "__main__":
    main()