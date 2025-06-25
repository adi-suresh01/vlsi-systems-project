# VLSI-AI Systems Integration

This project explores the integration of deep learning inference pipelines with VLSI architectures for edge deployment. It focuses on simulating convolutional neural networks (CNNs) on Xilinx FPGAs, optimizing hardware components, and benchmarking AI workload offloading.

## Project Structure

```
vlsi-ai-integration
├── src
│   ├── ai_models          # Contains AI model training and conversion scripts
│   ├── hardware           # Contains SystemVerilog files for hardware design
│   ├── benchmarking        # Contains benchmarking utilities
│   ├── integration         # Contains integration scripts for AI and hardware
│   └── dashboard           # Contains the FastAPI dashboard for visualization
├── data                   # Contains datasets and trained models
├── tests                  # Contains unit tests for various components
├── scripts                # Contains scripts for training, simulation, and benchmarking
├── requirements.txt       # Lists Python dependencies
├── setup.py               # Setup script for the project
├── .gitignore             # Git ignore file
└── README.md              # Project documentation
```

## Getting Started

### Prerequisites

- Python 3.x
- TensorFlow
- Vivado (for FPGA simulation)
- SystemVerilog simulator

### Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/vlsi-ai-integration.git
   cd vlsi-ai-integration
   ```

2. Install the required Python packages:
   ```
   pip install -r requirements.txt
   ```

### Usage

1. **Train the CNN Model**:
   Run the training script to train a CNN on the MNIST dataset:
   ```
   python scripts/train_model.py
   ```

2. **Convert to TensorFlow Lite**:
   Convert the trained model to TensorFlow Lite format:
   ```
   python -m src.ai_models.tflite_converter
   ```

3. **Run Hardware Simulation**:
   Execute the simulation script to run the hardware simulation:
   ```
   python scripts/run_simulation.py
   ```

4. **Benchmark Performance**:
   Benchmark the performance and power consumption:
   ```
   python scripts/benchmark.py
   ```

5. **Launch Dashboard** (optional):
   Start the FastAPI dashboard to visualize results:
   ```
   uvicorn src.dashboard.api:app --reload
   ```

### Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or features.

### License

This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgments

- Inspired by the need for efficient AI inference on edge devices.
- Thanks to the open-source community for their contributions to machine learning and hardware design.