import numpy as np

def estimate_power(hardware_results):
    """Estimate power consumption based on hardware simulation results."""
    if isinstance(hardware_results, dict):
        # If hardware_results already contains estimated_power, use it
        if 'estimated_power' in hardware_results:
            return float(hardware_results['estimated_power'])
        
        # Otherwise calculate from cycles and operations
        cycles = hardware_results.get('simulated_cycles', 1000)
        operations = hardware_results.get('operations_count', 784)
    else:
        # Fallback for other input types
        cycles = 1000
        operations = 784
    
    # Power estimation model
    base_power = 0.5  # Base power in Watts
    dynamic_power_per_cycle = 0.0001  # Dynamic power per cycle
    dynamic_power_per_op = 0.0001     # Dynamic power per operation
    
    total_power = base_power + (cycles * dynamic_power_per_cycle) + (operations * dynamic_power_per_op)
    
    return float(total_power)  # Ensure we return a float, not numpy array

def analyze_performance(cpu_time, tflite_time, hardware_results):
    """Analyze performance comparison."""
    if isinstance(hardware_results, dict):
        hw_time = hardware_results.get('execution_time', 0.1)
    else:
        hw_time = 0.1  # Default fallback
    
    cpu_speedup = cpu_time / hw_time if hw_time > 0 else 0
    tflite_speedup = tflite_time / hw_time if hw_time > 0 else 0
    
    return {
        'cpu_vs_hw_speedup': float(cpu_speedup),
        'tflite_vs_hw_speedup': float(tflite_speedup),
        'hw_execution_time': float(hw_time)
    }

def calculate_energy_efficiency(power_watts, execution_time_seconds):
    """Calculate energy efficiency metrics."""
    energy_joules = power_watts * execution_time_seconds
    return {
        'energy_consumption': float(energy_joules),
        'power_efficiency': float(1.0 / power_watts) if power_watts > 0 else 0
    }

def estimate_hardware_metrics(cycles, frequency_mhz=100):
    """Estimate hardware performance metrics."""
    execution_time = cycles / (frequency_mhz * 1e6)  # Convert MHz to Hz
    power_per_cycle = 0.1e-9  # 0.1 nW per cycle
    total_power = cycles * power_per_cycle
    
    return {
        'estimated_execution_time': float(execution_time),
        'estimated_power': float(total_power),
        'frequency_mhz': float(frequency_mhz),
        'cycles': int(cycles)
    }