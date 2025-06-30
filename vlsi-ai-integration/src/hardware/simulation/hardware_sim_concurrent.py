import numpy as np
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock, Event
import queue

# Shared resources that could cause race conditions
operation_counter = 0
operation_lock = Lock()
results_queue = queue.Queue()

def simulate_mac_operation_threaded(a, b, c, thread_id):
    """Thread-safe MAC operation with potential race condition"""
    global operation_counter
    
    # Simulate computation time
    time.sleep(0.000001)
    result = a * b + c
    
    # RACE CONDITION EXAMPLE - comment/uncomment to see difference
    # Without lock (race condition):
    # operation_counter += 1
    
    # With lock (thread-safe):
    with operation_lock:
        operation_counter += 1
        print(f"Thread {thread_id}: MAC op #{operation_counter}")
    
    return result

def process_sample_threaded(sample_data, sample_id):
    """Process a single sample in a separate thread"""
    print(f"Thread {threading.current_thread().name}: Processing sample {sample_id}")
    
    if len(sample_data.shape) == 3:
        sample_data = sample_data.squeeze()
    
    local_mac_ops = 0
    local_relu_ops = 0
    conv_results = []
    
    # Simulate convolution with threading
    for kernel_idx in range(8):
        kernel = np.random.randn(3, 3) * 0.1
        conv_sum = 0
        
        for y in range(0, sample_data.shape[0]-2, 4):
            for x in range(0, sample_data.shape[1]-2, 4):
                patch = sample_data[y:y+3, x:x+3]
                
                # Use threaded MAC operation
                mac_result = simulate_mac_operation_threaded(
                    np.sum(patch), np.sum(kernel), conv_sum, 
                    threading.current_thread().name
                )
                conv_sum += mac_result
                local_mac_ops += 9
        
        # ReLU activation
        relu_result = max(0, conv_sum)
        local_relu_ops += 1
        conv_results.append(relu_result)
    
    result = {
        'sample_id': sample_id,
        'result': np.mean(conv_results),
        'mac_ops': local_mac_ops,
        'relu_ops': local_relu_ops,
        'thread_name': threading.current_thread().name
    }
    
    # Thread-safe result storage
    results_queue.put(result)
    
    return result

def run_concurrent_hardware_simulation(input_data, max_workers=4):
    """Hardware simulation with multi-threading"""
    global operation_counter
    operation_counter = 0
    
    print(f"Running concurrent hardware simulation with {max_workers} threads...")
    start_time = time.time()
    
    # Thread pool for parallel processing
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all samples for concurrent processing
        future_to_sample = {
            executor.submit(process_sample_threaded, sample, i): i 
            for i, sample in enumerate(input_data)
        }
        
        results = []
        total_mac_ops = 0
        total_relu_ops = 0
        
        # Collect results as they complete
        for future in as_completed(future_to_sample):
            sample_id = future_to_sample[future]
            try:
                result = future.result()
                results.append((result['sample_id'], result['result']))
                total_mac_ops += result['mac_ops']
                total_relu_ops += result['relu_ops']
                print(f"Completed sample {result['sample_id']} on {result['thread_name']}")
            except Exception as exc:
                print(f"Sample {sample_id} generated exception: {exc}")
    
    # Sort results by sample_id to maintain order
    results.sort(key=lambda x: x[0])
    final_results = [r[1] for r in results]
    
    execution_time = time.time() - start_time
    
    return {
        'results': np.array(final_results),
        'execution_time': execution_time,
        'total_mac_ops': total_mac_ops,
        'total_relu_ops': total_relu_ops,
        'global_operation_count': operation_counter,
        'threads_used': max_workers
    }