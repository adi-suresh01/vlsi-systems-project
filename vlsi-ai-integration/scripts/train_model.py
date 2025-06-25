#!/usr/bin/env python3
import os
import sys
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from ai_models.cnn_trainer import train_cnn_model
from ai_models.tflite_converter import convert_model_to_tflite

def main():
    print("Starting VLSI-AI Integration Project Training...")
    
    # Create directories if they don't exist - with better error handling
    try:
        # Check if data directory exists and is actually a directory
        if os.path.exists('data') and not os.path.isdir('data'):
            print("Error: 'data' exists but is not a directory. Please remove it.")
            sys.exit(1)
        
        if os.path.exists('data/models') and not os.path.isdir('data/models'):
            print("Error: 'data/models' exists but is not a directory. Please remove it.")
            sys.exit(1)
        
        if os.path.exists('data/mnist') and not os.path.isdir('data/mnist'):
            print("Error: 'data/mnist' exists but is not a directory. Please remove it.")
            sys.exit(1)
        
        # Create directories
        os.makedirs('data', exist_ok=True)
        os.makedirs('data/models', exist_ok=True)
        os.makedirs('data/mnist', exist_ok=True)
        
        print("✅ Directories created successfully")
        
    except Exception as e:
        print(f"❌ Error creating directories: {e}")
        sys.exit(1)
    
    # Train the CNN model on the MNIST dataset
    print("Training CNN model on MNIST dataset...")
    try:
        model = train_cnn_model()
        print("✅ Model training completed")
    except Exception as e:
        print(f"❌ Error training model: {e}")
        sys.exit(1)
    
    # Save the trained model
    model_save_path = 'data/models/mnist_cnn.h5'
    try:
        model.save(model_save_path)
        print(f"✅ Model saved to: {model_save_path}")
    except Exception as e:
        print(f"❌ Error saving model: {e}")
        sys.exit(1)
    
    # Convert the trained model to TensorFlow Lite format
    print("Converting model to TensorFlow Lite...")
    tflite_save_path = 'data/models/mnist_cnn.tflite'
    try:
        convert_model_to_tflite(model, tflite_save_path)
        print(f"✅ TFLite model saved to: {tflite_save_path}")
    except Exception as e:
        print(f"❌ Error converting to TFLite: {e}")
        sys.exit(1)
    
    # Save some test data for benchmarking
    print("Saving test data for benchmarking...")
    try:
        from ai_models.cnn_trainer import load_mnist_data
        (_, _), (x_test, y_test) = load_mnist_data()
        
        # Save first 100 test samples
        test_data = x_test[:100]
        test_labels = y_test[:100]
        
        np.save('data/mnist/x_test.npy', test_data)
        np.save('data/mnist/y_test.npy', test_labels)
        print("✅ Test data saved for benchmarking")
    except Exception as e:
        print(f"❌ Error saving test data: {e}")
        sys.exit(1)
    
    print("\n🎉 Training completed successfully!")
    print("📁 Files created:")
    print(f"   - {model_save_path}")
    print(f"   - {tflite_save_path}")
    print(f"   - data/mnist/x_test.npy")
    print(f"   - data/mnist/y_test.npy")
    print("\n📋 Next steps:")
    print("   1. Run hardware simulation: python scripts/run_simulation.py")
    print("   2. Run full benchmark: python scripts/benchmark.py")

if __name__ == "__main__":
    main()