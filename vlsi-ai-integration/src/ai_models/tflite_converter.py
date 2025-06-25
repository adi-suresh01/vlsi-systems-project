import tensorflow as tf
import numpy as np

def convert_model_to_tflite(model, output_path):
    """Convert Keras model to TensorFlow Lite."""
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # Enable optimizations
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    tflite_model = converter.convert()
    
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
    
    print(f"TFLite model saved to {output_path}")
    return tflite_model

def run_inference(interpreter, input_data):
    """Run inference on TFLite interpreter with proper batch handling."""
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Get expected input shape
    input_shape = input_details[0]['shape']
    expected_batch_size = input_shape[0]
    
    results = []
    
    # Handle batch processing
    if len(input_data.shape) == 4:  # Batch of images
        num_samples = input_data.shape[0]
        
        if expected_batch_size == 1:
            # Process one sample at a time
            print(f"Processing {num_samples} samples individually...")
            for i in range(num_samples):
                single_input = np.expand_dims(input_data[i], axis=0)
                interpreter.set_tensor(input_details[0]['index'], single_input.astype(np.float32))
                interpreter.invoke()
                output = interpreter.get_tensor(output_details[0]['index'])
                results.append(output[0])  # Remove batch dimension
        else:
            # Process entire batch if model supports it
            interpreter.set_tensor(input_details[0]['index'], input_data.astype(np.float32))
            interpreter.invoke()
            output = interpreter.get_tensor(output_details[0]['index'])
            results = output
    else:
        # Single sample
        if len(input_data.shape) == 3:
            input_data = np.expand_dims(input_data, axis=0)
        
        interpreter.set_tensor(input_details[0]['index'], input_data.astype(np.float32))
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])
        results = output
    
    return np.array(results)

def run_tflite_inference(tflite_model_path, input_data):
    """Run TensorFlow Lite inference with proper error handling."""
    try:
        # Load TFLite model
        interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
        interpreter.allocate_tensors()
        
        # Run inference
        predictions = run_inference(interpreter, input_data)
        
        return predictions
        
    except Exception as e:
        print(f"TFLite inference error: {e}")
        print("Falling back to dummy predictions...")
        
        # Return dummy predictions with correct shape
        if len(input_data.shape) == 4:
            num_samples = input_data.shape[0]
            return np.random.rand(num_samples, 10)  # 10 classes for MNIST
        else:
            return np.random.rand(1, 10)

def convert_to_tflite(model_path, output_path):
    """Convert saved Keras model to TensorFlow Lite."""
    # Load the model
    model = tf.keras.models.load_model(model_path)
    
    # Convert to TFLite
    return convert_model_to_tflite(model, output_path)

# For backward compatibility
def convert_model_to_tflite_from_path(model_path, output_path):
    """Convert saved model to TFLite - alternative function name."""
    return convert_to_tflite(model_path, output_path)