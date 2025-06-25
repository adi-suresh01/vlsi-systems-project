import tensorflow as tf
import numpy as np

def load_model(model_path):
    """Load a saved Keras model."""
    model = tf.keras.models.load_model(model_path)
    return model

def save_model(model, model_path):
    """Save a Keras model."""
    model.save(model_path)

def preprocess_input(input_data):
    """Preprocess input data for model inference."""
    # Normalize to [0, 1] and ensure correct dtype
    input_data = input_data.astype('float32') / 255.0
    return input_data

def postprocess_output(output_data):
    """Postprocess model output to get predicted classes."""
    return np.argmax(output_data, axis=1)

def load_tflite_model(tflite_model_path):
    """Load TensorFlow Lite model."""
    interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
    interpreter.allocate_tensors()
    return interpreter

def run_inference(interpreter, input_data):
    """Run inference on TensorFlow Lite interpreter."""
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]['index'], input_data.astype(np.float32))
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    
    return output_data