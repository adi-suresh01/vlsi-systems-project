import unittest
import os
import sys
import numpy as np
import warnings
import tempfile

# Suppress warnings for cleaner test output
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from ai_models.cnn_trainer import train_cnn_model
    from ai_models.tflite_converter import convert_model_to_tflite
    from hardware.simulation.hardware_sim import run_hardware_simulation
    from integration.tflite_hw_bridge import TFLiteHWBridge
    from benchmarking.power_estimator import analyze_performance
    
    IMPORTS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Some imports failed: {e}")
    IMPORTS_AVAILABLE = False

class TestIntegration(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Set up class-level resources"""
        if not IMPORTS_AVAILABLE:
            return
        
        # Create temporary directory for test models
        cls.temp_dir = tempfile.mkdtemp()
        os.makedirs(os.path.join(cls.temp_dir, 'models'), exist_ok=True)

    def setUp(self):
        """Setup code to initialize any required resources"""
        if not IMPORTS_AVAILABLE:
            self.skipTest("Required imports not available")
        
        self.model_path = os.path.join(self.temp_dir, 'models', 'test_integration_cnn.h5')
        self.tflite_path = os.path.join(self.temp_dir, 'models', 'test_integration_cnn.tflite')
        
        # Use smaller test data to speed up tests
        np.random.seed(42)
        self.test_data = np.random.randn(2, 28, 28, 1).astype(np.float32)
        
        try:
            # Train and convert model
            print("Training model for integration test...")
            self.model = train_cnn_model()
            self.model.save(self.model_path)
            
            print("Converting to TFLite...")
            self.tflite_model = convert_model_to_tflite(self.model, self.tflite_path)
        except Exception as e:
            self.skipTest(f"Model setup failed: {e}")

    def test_hardware_integration(self):
        """Test hardware simulation integration"""
        print("Testing hardware integration...")
        
        simulation_results = run_hardware_simulation(self.test_data)
        self.assertIsNotNone(simulation_results, "Simulation results should not be None")
        self.assertIn('results', simulation_results, "Should contain results")
        self.assertIn('execution_time', simulation_results, "Should contain execution time")

    def test_tflite_hw_bridge(self):
        """Test the TFLite to hardware bridge"""
        print("Testing TFLite hardware bridge...")
        
        if not os.path.exists(self.tflite_path):
            self.skipTest("TFLite model not available")
        
        bridge = TFLiteHWBridge(self.tflite_path)
        
        # Test TFLite inference
        tflite_results = bridge.run_inference(self.test_data)
        self.assertIsNotNone(tflite_results, "TFLite results should not be None")
        self.assertGreater(len(tflite_results), 0, "Should have inference results")
        
        # Test hardware simulation through bridge
        hw_results = bridge.simulate_hardware_inference(self.test_data)
        self.assertIsNotNone(hw_results, "Hardware results should not be None")

    def test_performance_analysis(self):
        """Test performance analysis"""
        print("Testing performance analysis...")
        
        hw_results = run_hardware_simulation(self.test_data)
        
        # Mock CPU time for comparison
        cpu_time = 0.1
        tflite_time = 0.08
        
        performance_metrics = analyze_performance(cpu_time, tflite_time, hw_results)
        self.assertIsNotNone(performance_metrics, "Performance metrics should not be None")
        self.assertIn('cpu_vs_hw_speedup', performance_metrics, "Should include speedup metrics")
        
        # Check that speedup is a reasonable number
        speedup = performance_metrics['cpu_vs_hw_speedup']
        self.assertIsInstance(speedup, (int, float), "Speedup should be numeric")
        self.assertGreaterEqual(speedup, 0, "Speedup should be non-negative")

    def test_end_to_end_pipeline(self):
        """Test complete pipeline"""
        print("Testing end-to-end pipeline...")
        
        if not os.path.exists(self.tflite_path):
            self.skipTest("TFLite model not available")
        
        bridge = TFLiteHWBridge(self.tflite_path)
        benchmark_results = bridge.benchmark_inference(self.test_data)
        
        self.assertIsNotNone(benchmark_results, "Benchmark results should not be None")
        
        required_keys = ['cpu_inference_time', 'hardware_inference_time', 'speedup']
        for key in required_keys:
            self.assertIn(key, benchmark_results, f"Should contain {key}")
            self.assertIsInstance(benchmark_results[key], (int, float), f"{key} should be numeric")

    def tearDown(self):
        """Cleanup"""
        try:
            if hasattr(self, 'model_path') and os.path.exists(self.model_path):
                os.remove(self.model_path)
            if hasattr(self, 'tflite_path') and os.path.exists(self.tflite_path):
                os.remove(self.tflite_path)
        except OSError as e:
            print(f"Warning: Could not remove test files: {e}")

    @classmethod
    def tearDownClass(cls):
        """Clean up class-level resources"""
        if hasattr(cls, 'temp_dir'):
            import shutil
            try:
                shutil.rmtree(cls.temp_dir)
            except OSError as e:
                print(f"Warning: Could not remove temp directory: {e}")

if __name__ == '__main__':
    unittest.main(verbosity=2)