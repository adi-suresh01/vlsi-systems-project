from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import os

def load_mnist_data():
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)
    return (x_train, y_train), (x_test, y_test)

def create_cnn_model():
    model = keras.Sequential([
        layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def train_cnn_model():
    (x_train, y_train), (x_test, y_test) = load_mnist_data()
    model = create_cnn_model()
    model.fit(x_train, y_train, epochs=5, batch_size=64, validation_data=(x_test, y_test))
    return model

# Add this function that benchmark.py expects
def load_model(model_path):
    """Load a trained model from file path."""
    return keras.models.load_model(model_path)

# Add this function for better integration
def train_model(save_path='data/models/mnist_cnn.h5'):
    """Train CNN on MNIST dataset and save it."""
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Load data and train model
    (x_train, y_train), (x_test, y_test) = load_mnist_data()
    model = create_cnn_model()
    
    print("Training CNN model...")
    history = model.fit(x_train, y_train, epochs=5, batch_size=64, 
                       validation_data=(x_test, y_test), verbose=1)
    
    # Save model
    model.save(save_path)
    print(f"Model saved to {save_path}")
    
    return model, (x_test, y_test)

if __name__ == "__main__":
    trained_model = train_cnn_model()
    trained_model.save('data/models/mnist_cnn_model.h5')