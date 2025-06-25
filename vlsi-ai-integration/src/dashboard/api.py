from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request
import sys
import os
import numpy as np
import json
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from ai_models.model_utils import load_model
from ai_models.tflite_converter import run_tflite_inference
from hardware.simulation.hardware_sim import run_hardware_simulation
from benchmarking.power_estimator import analyze_performance, estimate_power

app = FastAPI(title="VLSI-AI Integration Dashboard", version="1.0.0")

# Mount static files
app.mount("/static", StaticFiles(directory="src/dashboard/static"), name="static")

# Set up templates
templates = Jinja2Templates(directory="src/dashboard/templates")

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/api/benchmark")
async def run_benchmark():
    """Run benchmark and return results as JSON"""
    try:
        # Check if required files exist
        model_path = 'data/models/mnist_cnn.h5'
        tflite_model_path = 'data/models/mnist_cnn.tflite'
        input_data_path = 'data/mnist/x_test.npy'
        
        if not all([os.path.exists(f) for f in [model_path, tflite_model_path, input_data_path]]):
            raise HTTPException(status_code=404, detail="Required model files not found. Please run training script first.")
        
        # Load test data
        input_data = np.load(input_data_path)
        test_samples = input_data[:10]  # Use 10 samples for dashboard
        
        # Load models and run benchmarks
        model = load_model(model_path)
        
        # CPU Inference
        import time
        start_time = time.time()
        cpu_predictions = model.predict(test_samples, verbose=0)
        cpu_time = time.time() - start_time
        
        # TFLite Inference
        start_time = time.time()
        tflite_predictions = run_tflite_inference(tflite_model_path, test_samples)
        tflite_time = time.time() - start_time
        
        # Hardware Simulation
        hardware_results = run_hardware_simulation(test_samples)
        hw_time = hardware_results['execution_time']
        
        # Performance Analysis
        performance_metrics = analyze_performance(cpu_time, tflite_time, hardware_results)
        power_consumption = estimate_power(hardware_results)
        
        # Prepare response data
        response_data = {
            "timestamp": datetime.now().isoformat(),
            "sample_count": len(test_samples),
            "performance": {
                "cpu_time": float(cpu_time),
                "tflite_time": float(tflite_time),
                "hardware_time": float(hw_time),
                "cpu_per_sample": float(cpu_time / len(test_samples) * 1000),  # ms
                "tflite_per_sample": float(tflite_time / len(test_samples) * 1000),  # ms
                "hardware_per_sample": float(hw_time / len(test_samples) * 1000)  # ms
            },
            "speedup": {
                "cpu_vs_hardware": float(performance_metrics['cpu_vs_hw_speedup']),
                "tflite_vs_hardware": float(performance_metrics['tflite_vs_hw_speedup'])
            },
            "power": {
                "total_watts": float(power_consumption),
                "total_milliwatts": float(power_consumption * 1000),
                "total_microwatts": float(power_consumption * 1000000),
                "energy_per_inference_uj": float((power_consumption * hw_time / len(test_samples)) * 1000000)
            },
            "hardware_details": {
                "cycles": int(hardware_results.get('simulated_cycles', 0)),
                "operations": int(hardware_results.get('operations_count', 0)),
                "mac_ops": int(hardware_results.get('mac_operations', 0)),
                "relu_ops": int(hardware_results.get('relu_operations', 0)),
                "throughput": float(hardware_results.get('throughput', 0))
            },
            "accuracy": {
                "cpu_sample_prediction": int(np.argmax(cpu_predictions[0])),
                "tflite_sample_prediction": int(np.argmax(tflite_predictions[0])),
                "predictions_match": bool(np.argmax(cpu_predictions[0]) == np.argmax(tflite_predictions[0]))
            }
        }
        
        return JSONResponse(content=response_data)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Benchmark failed: {str(e)}")

@app.get("/api/status")
async def get_status():
    """Check if required files exist"""
    files_to_check = {
        "keras_model": "data/models/mnist_cnn.h5",
        "tflite_model": "data/models/mnist_cnn.tflite",
        "test_data": "data/mnist/x_test.npy"
    }
    
    status = {}
    all_ready = True
    
    for name, path in files_to_check.items():
        exists = os.path.exists(path)
        status[name] = {
            "exists": exists,
            "path": path
        }
        if not exists:
            all_ready = False
    
    return {
        "ready": all_ready,
        "files": status,
        "message": "All files ready!" if all_ready else "Some required files are missing. Please run the training script."
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)