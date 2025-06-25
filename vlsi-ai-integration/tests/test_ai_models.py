import unittest
import os
import sys
import numpy as np
import warnings

# Suppress TensorFlow warnings for cleaner test output
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from ai_models.cnn_trainer import train_cnn_model, load_model
from ai_models.tflite_converter import convert_model_to_tflite
from ai_models.model_utils import save_model

class TestAIModels(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Set up class-level resources"""
        os.makedirs('data/models', exist_ok=True)

    def setUp(self):
        # Setup code to initialize variables or state before each test
        self.model_path = 'data/models/test_mnist_cnn.h5'
        self.tflite_model_path = 'data/models/test_mnist_cnn.tflite'
        self.model = None

    def test_train_cnn(self):
        """Test the CNN training function"""
        print("Testing CNN training...")
        self.model = train_cnn_model()
        self.assertIsNotNone(self.model, "The model should be trained and not None.")
        
        # Check model structure
        self.assertTrue(hasattr(self.model, 'predict'), "Model should have predict method")
        
        save_model(self.model, self.model_path)
        
        # Verify model was saved
        self.assertTrue(os.path.exists(self.model_path), "Model file should exist after saving")
        
        # Verify model file size is reasonable
        file_size = os.path.getsize(self.model_path)
        self.assertGreater(file_size, 1000, "Model file should be larger than 1KB")

    def test_convert_to_tflite(self):
        """Test the conversion of the trained model to TensorFlow Lite format"""
        print("Testing TFLite conversion...")
        
        if self.model is None:
            if os.path.exists(self.model_path):
                self.model = load_model(self.model_path)
            else:
                self.model = train_cnn_model()
                save_model(self.model, self.model_path)
        
        tflite_model = convert_model_to_tflite(self.model, self.tflite_model_path)
        self.assertIsNotNone(tflite_model, "The TFLite model should be created and not None.")
        
        # Verify TFLite model was saved
        self.assertTrue(os.path.exists(self.tflite_model_path), "TFLite model file should exist")
        
        # Verify TFLite model file size
        file_size = os.path.getsize(self.tflite_model_path)
        self.assertGreater(file_size, 100, "TFLite model file should be larger than 100 bytes")

    def test_model_loading(self):
        """Test model loading functionality"""
        if not os.path.exists(self.model_path):
            model = train_cnn_model()
            save_model(model, self.model_path)
        
        loaded_model = load_model(self.model_path)
        self.assertIsNotNone(loaded_model, "Loaded model should not be None")
        self.assertTrue(hasattr(loaded_model, 'predict'), "Loaded model should have predict method")

    def tearDown(self):
        """Cleanup code after each test"""
        try:
            if os.path.exists(self.model_path):
                os.remove(self.model_path)
            if os.path.exists(self.tflite_model_path):
                os.remove(self.tflite_model_path)
        except OSError as e:
            print(f"Warning: Could not remove test files: {e}")

if __name__ == '__main__':
    unittest.main(verbosity=2)