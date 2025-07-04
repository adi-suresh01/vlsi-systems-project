"""
C++ Hardware Simulator Interface
Python bindings for high-performance C++ hardware simulation
"""

import ctypes
import numpy as np
import os
from ctypes import POINTER, c_double, c_int, c_uint64, c_void_p

class CPPHardwareSimulator:
    def __init__(self):
        # Load the compiled C++ library
        lib_path = os.path.join(os.path.dirname(__file__), 'libhardware_simulator.so')
        if not os.path.exists(lib_path):
            raise FileNotFoundError(f"C++ library not found at {lib_path}. Run ./compile.sh first!")
        
        self.lib = ctypes.CDLL(lib_path)
        
        # Define function signatures
        self.lib.create_simulator.restype = c_void_p
        self.lib.destroy_simulator.argtypes = [c_void_p]
        
        self.lib.run_cpp_simulation.argtypes = [
            c_void_p,                    # simulator
            POINTER(c_double),           # input_data
            c_int, c_int, c_int,        # samples, height, width
            POINTER(c_double),           # results
            POINTER(c_double),           # execution_time
            POINTER(c_uint64),           # mac_ops
            POINTER(c_uint64),           # relu_ops
            POINTER(c_double)            # power
        ]
        
        # Create simulator instance
        self.simulator = self.lib.create_simulator()
    
    def __del__(self):
        if hasattr(self, 'simulator'):
            self.lib.destroy_simulator(self.simulator)
    
    def run_simulation(self, input_data):
        """Run C++ hardware simulation"""
        # Convert numpy array to C++ format
        if len(input_data.shape) == 4:
            input_data = input_data.squeeze(-1)  # Remove channel dimension
        
        samples, height, width = input_data.shape
        
        # Flatten input data
        flat_data = input_data.flatten().astype(np.float64)
        
        # Prepare output arrays
        results = np.zeros(samples, dtype=np.float64)
        execution_time = c_double()
        mac_ops = c_uint64()
        relu_ops = c_uint64()
        power = c_double()
        
        # Call C++ function
        self.lib.run_cpp_simulation(
            self.simulator,
            flat_data.ctypes.data_as(POINTER(c_double)),
            c_int(samples), c_int(height), c_int(width),
            results.ctypes.data_as(POINTER(c_double)),
            ctypes.byref(execution_time),
            ctypes.byref(mac_ops),
            ctypes.byref(relu_ops),
            ctypes.byref(power)
        )
        
        return {
            'results': results,
            'execution_time': execution_time.value,
            'mac_operations': mac_ops.value,
            'relu_operations': relu_ops.value,
            'power_consumption': power.value,
            'throughput': samples / execution_time.value,
            'operations_count': mac_ops.value + relu_ops.value,
            'estimated_power': power.value  # For compatibility with existing code
        }

def run_cpp_hardware_simulation(input_data):
    """High-performance C++ hardware simulation interface"""
    simulator = CPPHardwareSimulator()
    return simulator.run_simulation(input_data)