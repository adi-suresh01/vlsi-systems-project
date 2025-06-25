# VLSI-AI Integration Project

A complete system demonstrating hardware-accelerated deep learning inference through VLSI design. This project implements a CNN for MNIST digit recognition and compares performance between CPU, TensorFlow Lite, and custom hardware simulation.

## What This Project Does

- Trains a CNN model on MNIST handwritten digits
- Converts the model to TensorFlow Lite for optimization
- Simulates hardware acceleration using custom MAC and ReLU units
- Benchmarks performance across CPU, TFLite, and hardware implementations
- Provides a web dashboard for real-time performance visualization

## Project Structure

```
vlsi-ai-integration/
├── README.md
├── requirements.txt
├── setup.py
│
├── data/                       # Generated models and test data
│   ├── mnist/                  # MNIST test data
│   │   ├── x_test.npy
│   │   └── y_test.npy
│   └── models/                 # Trained models
│       ├── mnist_cnn.h5        # Keras model
│       └── mnist_cnn.tflite    # TensorFlow Lite model
│
├── scripts/                    # Executable scripts
│   ├── train_model.py          # Train and save models
│   ├── run_simulation.py       # Hardware simulation only
│   └── benchmark.py            # Full performance comparison
│
├── src/
│   ├── ai_models/              # CNN training and model conversion
│   │   ├── cnn_trainer.py      # MNIST CNN implementation
│   │   ├── tflite_converter.py # TensorFlow Lite conversion
│   │   └── model_utils.py      # Model loading utilities
│   │
│   ├── benchmarking/           # Performance analysis
│   │   ├── comparison_utils.py # Utility functions
│   │   ├── performance_analyzer.py # Performance analysis
│   │   └── power_estimator.py  # Power consumption calculations
│   │
│   ├── hardware/               # Hardware simulation
│   │   ├── simulation/         # Verilog simulation interface
│   │   ├── components/         # MAC, ReLU, and other units
│   │   └── systemverilog/      # SystemVerilog hardware modules
│   │       ├── mac_unit.sv     # MAC unit implementation
│   │       ├── relu_unit.sv    # ReLU activation unit
│   │       └── pipeline.sv     # Complete pipeline
│   │
│   ├── integration/            # Bridge between AI and hardware
│   │   ├── tflite_hw_bridge.py # TFLite-Hardware interface
│   │   └── pipeline_controller.py # Complete pipeline controller
│   │
│   └── dashboard/              # Web dashboard
│       ├── api.py              # FastAPI backend
│       ├── templates/          # HTML templates
│       │   └── index.html      # Main dashboard page
│       └── static/             # CSS and JavaScript
│           ├── css/
│           │   └── styles.css  # Dashboard styling
│           └── js/
│               └── dashboard.js # Dashboard functionality
│
└── tests/                      # Unit tests
    ├── test_ai_models.py       # AI model tests
    ├── test_hardware.py        # Hardware simulation tests
    └── test_integration.py     # Integration tests
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the Model

This creates the CNN model, converts to TFLite, and saves test data:

```bash
python scripts/train_model.py
```

Expected output:
```
✅ Model training completed
✅ Model saved to: data/models/mnist_cnn.h5
✅ TFLite model saved to: data/models/mnist_cnn.tflite
✅ Test data saved for benchmarking
```

### 3. Run Benchmark Comparison

Compare performance across all three implementations:

```bash
python scripts/benchmark.py
```

This shows detailed timing, speedup analysis, power consumption, and accuracy verification.

### 4. Launch Dashboard (Optional)

For interactive visualization:

```bash
python -m uvicorn src.dashboard.api:app --reload --host 0.0.0.0 --port 8000
```

Open http://localhost:8000 to see real-time performance charts.

## What You'll See

### Benchmark Results
- **Performance**: CPU vs TFLite vs Hardware timing comparison
- **Speedup**: How much faster hardware is compared to software
- **Power**: Estimated power consumption in µW/mW/W
- **Accuracy**: Verification that all implementations give identical results

### Dashboard Features
- **Real-time Charts**: Interactive performance comparison graphs
- **Hardware Metrics**: Clock cycles, MAC operations, ReLU operations
- **Power Analysis**: Total power consumption and energy per inference
- **Accuracy Verification**: Ensures all implementations match
- **Live Logging**: Real-time benchmark progress updates

### Example Output
```
🎯 VLSI-AI INTEGRATION BENCHMARK RESULTS
========================================
📊 PERFORMANCE COMPARISON (10 samples):
   CPU Inference Time:        0.1504 seconds
   TFLite Inference Time:     0.0890 seconds
   Hardware Simulation Time:  0.0320 seconds

⚡ SPEEDUP ANALYSIS:
   CPU vs Hardware:           4.70x
   🎉 Hardware is 4.70x FASTER than CPU!

🔋 POWER ANALYSIS:
   Hardware Power Consumption: 1.234 mW
   Energy per inference:       3.946 µJ
```

## Key Components

### AI Models
- **CNN Architecture**: 2 conv layers + 2 dense layers for MNIST
- **TensorFlow Lite**: Optimized model for mobile/edge deployment
- **Model Utils**: Loading and validation functions

### Hardware Simulation
- **MAC Units**: Multiply-accumulate operations with realistic timing
- **ReLU Units**: Hardware activation function implementation
- **SystemVerilog Modules**: Actual hardware description in `src/hardware/systemverilog/`
- **Pipeline Controller**: Complete integration pipeline management

### Benchmarking & Analysis
- **Power Estimation**: Based on operation counts and hardware characteristics
- **Performance Analysis**: Comprehensive timing and speedup calculations
- **Comparison Utils**: Utilities for cross-platform performance comparison
- **Energy Efficiency**: Per-inference energy consumption metrics

### Web Dashboard
- **FastAPI Backend**: RESTful API for benchmark data
- **Interactive Charts**: Chart.js powered visualizations
- **Real-time Updates**: Live performance monitoring
- **Responsive Design**: Works on desktop and mobile devices

## Understanding the Results

### Performance Metrics
- **CPU Time**: Standard TensorFlow inference on CPU
- **TFLite Time**: Optimized inference using TensorFlow Lite
- **Hardware Time**: Custom VLSI implementation simulation

### Power Analysis
- Values typically range from µW to mW for realistic edge deployment
- Energy per inference shows efficiency for battery-powered devices
- Hardware usually wins on power efficiency vs raw compute power

### Accuracy Verification
- All three implementations should give identical predictions
- This ensures hardware acceleration doesn't compromise accuracy

## Customization

### Changing the Model
Edit `src/ai_models/cnn_trainer.py` to modify:
- Network architecture
- Training parameters
- Dataset (currently MNIST)

### Hardware Parameters
Modify `src/hardware/simulation/hardware_sim.py` for:
- Clock frequency assumptions
- Power consumption models
- Operation timing characteristics

### SystemVerilog Hardware
Actual hardware modules in `src/hardware/systemverilog/`:
- `mac_unit.sv` - Multiply-accumulate unit
- `relu_unit.sv` - ReLU activation function
- `pipeline.sv` - Complete inference pipeline

### Dashboard Appearance
Customize `src/dashboard/static/css/styles.css` for visual changes.

## Technical Details

- **Framework**: TensorFlow 2.x with Keras
- **Hardware**: SystemVerilog modules + Python simulation interface
- **Web**: FastAPI + Chart.js for dashboard
- **Testing**: pytest for comprehensive unit tests

## Troubleshooting

**Missing files error**: Run `python scripts/train_model.py` first

**Import errors**: Ensure all dependencies installed with `pip install -r requirements.txt`

**Dashboard not loading**: Check that uvicorn is installed and port 8000 is available

**Low speedup**: This is simulation - real hardware would show better performance

This project demonstrates the complete pipeline from AI model to hardware implementation, making it easy to understand the benefits of hardware acceleration for edge AI deployment.