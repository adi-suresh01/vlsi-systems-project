"""
VLSI-AI Integration Pipeline Controller
Orchestrates model training, conversion, hardware simulation, and performance analysis
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from ai_models.cnn_trainer import train_model, load_mnist_data
from ai_models.tflite_converter import convert_to_tflite
from hardware.simulation.hardware_sim import run_hardware_simulation
from benchmarking.performance_analyzer import PerformanceAnalyzer
from benchmarking.power_estimator import estimate_power, analyze_performance
from integration.tflite_hw_bridge import TFLiteHWBridge
import numpy as np

class PipelineController:
    def __init__(self):
        self.model = None
        self.tflite_model_path = None
        self.test_data = None
        self.simulation_results = None
        self.performance_metrics = None
        self.power_estimates = None
        self.analyzer = PerformanceAnalyzer()

    def run_pipeline(self):
        """Execute the complete AI-VLSI integration pipeline"""
        try:
            self.train_model()
            self.convert_model()
            self.simulate_hardware()
            self.analyze_results()
            print("Pipeline completed successfully!")
        except Exception as e:
            print(f"Pipeline failed: {e}")

    def train_model(self):
        """Train CNN model and prepare test data"""
        print("Training CNN model...")
        self.model, (x_test, y_test) = train_model('data/models/pipeline_cnn.h5')
        
        # Save test data for hardware simulation
        self.test_data = x_test[:10]  # Use first 10 samples
        np.save('data/mnist/pipeline_test_data.npy', self.test_data)
        print(f"Model trained and saved. Test data shape: {self.test_data.shape}")

    def convert_model(self):
        """Convert trained model to TensorFlow Lite format"""
        print("Converting model to TensorFlow Lite...")
        self.tflite_model_path = 'data/models/pipeline_cnn.tflite'
        convert_to_tflite('data/models/pipeline_cnn.h5', self.tflite_model_path)
        print(f"Model converted to TFLite: {self.tflite_model_path}")

    def simulate_hardware(self):
        """Execute hardware simulation on test data"""
        print("Running hardware simulation...")
        if self.test_data is not None:
            self.simulation_results = run_hardware_simulation(self.test_data)
            print(f"Hardware simulation completed in {self.simulation_results['execution_time']:.4f}s")
        else:
            print("No test data available for simulation")

    def analyze_results(self):
        """Analyze performance metrics and power consumption"""
        print("Analyzing performance and power...")
        
        if self.simulation_results is not None:
            # Estimate power consumption
            self.power_estimates = estimate_power(self.simulation_results)
            
            # Run CPU inference for comparison
            bridge = TFLiteHWBridge(self.tflite_model_path)
            cpu_start = time.time()
            cpu_results = bridge.run_inference(self.test_data)
            cpu_time = time.time() - cpu_start
            
            # Analyze performance comparison
            self.performance_metrics = analyze_performance(
                cpu_time, 
                cpu_time,  # TFLite time (same as CPU for now)
                self.simulation_results
            )
            
            # Log results using performance analyzer
            self.analyzer.log_performance(
                "Hardware_Simulation", 
                self.simulation_results['execution_time'],
                self.power_estimates
            )
            
            self.analyzer.log_performance(
                "CPU_Inference", 
                cpu_time,
                1.0  # Estimated CPU power
            )
            
            print("\n=== RESULTS ===")
            print(f"Hardware Execution Time: {self.simulation_results['execution_time']:.4f}s")
            print(f"CPU Execution Time: {cpu_time:.4f}s")
            print(f"Speedup: {self.performance_metrics['cpu_vs_hw_speedup']:.2f}x")
            print(f"Hardware Power Estimate: {self.power_estimates:.2f}W")
            print(f"Hardware Throughput: {self.simulation_results['throughput']:.2f} samples/s")
            
            self.analyzer.analyze_performance()
        else:
            print("No simulation results to analyze")

    def get_summary(self):
        """Get pipeline execution summary"""
        return {
            'model_trained': self.model is not None,
            'model_converted': self.tflite_model_path is not None,
            'hardware_simulated': self.simulation_results is not None,
            'performance_analyzed': self.performance_metrics is not None,
            'results': {
                'simulation_results': self.simulation_results,
                'performance_metrics': self.performance_metrics,
                'power_estimates': self.power_estimates
            }
        }

if __name__ == "__main__":
    import time
    
    # Create directories if they don't exist
    os.makedirs('data/models', exist_ok=True)
    os.makedirs('data/mnist', exist_ok=True)
    
    pipeline = PipelineController()
    pipeline.run_pipeline()
    
    # Print summary
    summary = pipeline.get_summary()
    print(f"\n=== PIPELINE SUMMARY ===")
    for key, value in summary.items():
        if key != 'results':
            print(f"{key}: {value}")