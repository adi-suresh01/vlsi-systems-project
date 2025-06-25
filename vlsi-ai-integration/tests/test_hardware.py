import unittest
import os
import sys
import numpy as np
import warnings

# Suppress warnings for cleaner test output
warnings.filterwarnings('ignore', category=RuntimeWarning)

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from hardware.simulation.hardware_sim import run_hardware_simulation
except ImportError as e:
    print(f"Warning: Could not import hardware simulation: {e}")
    run_hardware_simulation = None

class TestHardwareIntegration(unittest.TestCase):

    def setUp(self):
        """Create test data"""
        np.random.seed(42)  # For reproducible tests
        self.test_data = np.random.randn(5, 28, 28, 1).astype(np.float32)

    def test_hardware_simulation(self):
        """Test the hardware simulation function"""
        if run_hardware_simulation is None:
            self.skipTest("Hardware simulation not available")
        
        print("Testing hardware simulation...")
        result = run_hardware_simulation(self.test_data)
        self.assertIsNotNone(result, "Simulation should return results")
        
        # Check required keys in result
        required_keys = ['results', 'execution_time', 'simulated_cycles']
        for key in required_keys:
            self.assertIn(key, result, f"Result should contain {key}")
        
        # Check result types
        self.assertIsInstance(result['execution_time'], (int, float))
        self.assertIsInstance(result['simulated_cycles'], (int, float))
        self.assertIsInstance(result['results'], np.ndarray)
        
        # Check result values are reasonable
        self.assertGreater(result['execution_time'], 0, "Execution time should be positive")
        self.assertGreater(result['simulated_cycles'], 0, "Cycles should be positive")
        self.assertEqual(len(result['results']), len(self.test_data), "Results should match input length")

    def test_simulation_with_different_inputs(self):
        """Test with different input shapes"""
        if run_hardware_simulation is None:
            self.skipTest("Hardware simulation not available")
        
        test_cases = [
            np.random.randn(1, 28, 28, 1).astype(np.float32),
            np.random.randn(10, 28, 28, 1).astype(np.float32),
            np.random.randn(3, 28, 28, 1).astype(np.float32)
        ]
        
        for i, test_input in enumerate(test_cases):
            with self.subTest(f"Test case {i+1} - shape {test_input.shape}"):
                result = run_hardware_simulation(test_input)
                self.assertIsNotNone(result)
                self.assertEqual(len(result['results']), len(test_input))

    def test_simulation_performance_metrics(self):
        """Test that simulation returns proper performance metrics"""
        if run_hardware_simulation is None:
            self.skipTest("Hardware simulation not available")
        
        result = run_hardware_simulation(self.test_data)
        
        # Check for optional performance metrics
        optional_keys = ['estimated_power', 'throughput', 'operations_count']
        for key in optional_keys:
            if key in result:
                self.assertIsInstance(result[key], (int, float), f"{key} should be numeric")
                if key == 'throughput':
                    self.assertGreaterEqual(result[key], 0, "Throughput should be non-negative")

if __name__ == '__main__':
    unittest.main(verbosity=2)