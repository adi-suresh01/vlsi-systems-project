from typing import Any, Dict
import tensorflow as tf
import numpy as np
import time
import os
import subprocess

def convert_to_tflite(model_path, tflite_path):
    """Convert Keras model to TensorFlow Lite."""
    model = tf.keras.models.load_model(model_path)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)
    
    print(f"TFLite model saved to {tflite_path}")
    return tflite_model

def run_tflite_inference(tflite_model_path, input_data):
    """Run inference using TensorFlow Lite model."""
    interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Set input tensor
    interpreter.set_tensor(input_details[0]['index'], input_data.astype(np.float32))
    
    # Run inference
    interpreter.invoke()
    
    # Get output
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data

class TFLiteHWBridge:
    def __init__(self, tflite_model_path: str):
        self.tflite_model_path = tflite_model_path
        self.interpreter = self.load_tflite_model()

    def load_tflite_model(self) -> tf.lite.Interpreter:
        """Load TensorFlow Lite model"""
        interpreter = tf.lite.Interpreter(model_path=self.tflite_model_path)
        interpreter.allocate_tensors()
        return interpreter

    def get_input_output_details(self) -> dict:
        """Get model input/output details"""
        input_details = self.interpreter.get_input_details()
        output_details = self.interpreter.get_output_details()
        return {
            "input_details": input_details,
            "output_details": output_details
        }

    def run_inference(self, input_data: np.ndarray) -> np.ndarray:
        """Run TensorFlow Lite inference"""
        input_details = self.interpreter.get_input_details()
        output_details = self.interpreter.get_output_details()

        # Ensure input data has correct shape
        if len(input_data.shape) == 3:
            input_data = np.expand_dims(input_data, axis=0)
        
        self.interpreter.set_tensor(input_details[0]['index'], input_data.astype(np.float32))
        self.interpreter.invoke()
        output_data = self.interpreter.get_tensor(output_details[0]['index'])
        return output_data

    def simulate_hardware_inference(self, input_data: np.ndarray) -> dict:
        """Bridge to hardware simulation"""
        from hardware.simulation.hardware_sim import run_hardware_simulation
        
        # Run hardware simulation
        hw_results = run_hardware_simulation(input_data)
        return hw_results

    def benchmark_inference(self, input_data: np.ndarray) -> dict:
        """Compare CPU vs Hardware inference"""
        # CPU inference
        cpu_start = time.time()
        cpu_output = self.run_inference(input_data)
        cpu_time = time.time() - cpu_start
        
        # Hardware simulation
        hw_results = self.simulate_hardware_inference(input_data)
        
        # Calculate metrics
        speedup = cpu_time / hw_results['execution_time'] if hw_results['execution_time'] > 0 else 0
        
        benchmark_results = {
            'cpu_inference_time': cpu_time,
            'hardware_inference_time': hw_results['execution_time'],
            'speedup': speedup,
            'cpu_output_shape': cpu_output.shape,
            'hardware_results': hw_results,
            'accuracy_comparison': self._compare_outputs(cpu_output, hw_results['results'])
        }
        
        print("=== BENCHMARK RESULTS ===")
        print(f"CPU Inference Time: {cpu_time:.4f}s")
        print(f"Hardware Inference Time: {hw_results['execution_time']:.4f}s")
        print(f"Speedup: {speedup:.2f}x")
        print(f"Hardware Power: {hw_results.get('estimated_power', 'N/A')}W")
        
        return benchmark_results
    
    def _compare_outputs(self, cpu_output, hw_output):
        """Compare CPU and hardware outputs"""
        try:
            # Simple correlation check
            if len(cpu_output.flatten()) == len(hw_output.flatten()):
                correlation = np.corrcoef(cpu_output.flatten(), hw_output.flatten())[0, 1]
                return {'correlation': correlation}
            else:
                return {'correlation': 'N/A - different output shapes'}
        except Exception as e:
            return {'correlation': f'Error: {e}'}

def bridge_tflite_to_hardware(tflite_model_path, input_data):
    """Simple bridge function for compatibility"""
    bridge = TFLiteHWBridge(tflite_model_path)
    return bridge.benchmark_inference(input_data)

if __name__ == "__main__":
    # Test the bridge
    test_data = np.random.randn(5, 28, 28, 1)
    
    # This would require a trained model
    try:
        bridge = TFLiteHWBridge('data/models/mnist_cnn.tflite')
        results = bridge.benchmark_inference(test_data)
        print("Bridge test completed successfully!")
    except Exception as e:
        print(f"Bridge test failed: {e}")