import threading
import queue
import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor

class HardwarePipeline:
    """Simulates a hardware pipeline with multiple stages"""
    
    def __init__(self):
        # Pipeline queues
        self.input_queue = queue.Queue(maxsize=10)
        self.mac_queue = queue.Queue(maxsize=10)
        self.relu_queue = queue.Queue(maxsize=10)
        self.output_queue = queue.Queue()
        
        # Pipeline statistics
        self.stats_lock = threading.Lock()
        self.mac_operations = 0
        self.relu_operations = 0
        
        # Control flags
        self.shutdown_event = threading.Event()
    
    def mac_stage_worker(self):
        """MAC unit worker - processes convolution operations"""
        thread_name = threading.current_thread().name
        print(f"{thread_name}: MAC stage started")
        
        while not self.shutdown_event.is_set():
            try:
                # Get data from input queue (timeout to check shutdown)
                data = self.input_queue.get(timeout=1.0)
                
                if data is None:  # Poison pill
                    break
                
                sample_id, sample_data = data
                print(f"{thread_name}: Processing MAC for sample {sample_id}")
                
                # Simulate MAC operations
                conv_results = []
                local_mac_ops = 0
                
                for kernel_idx in range(4):  # Reduced for demo
                    kernel = np.random.randn(3, 3) * 0.1
                    conv_sum = 0
                    
                    # Simulate convolution
                    for y in range(0, sample_data.shape[0]-2, 8):
                        for x in range(0, sample_data.shape[1]-2, 8):
                            patch = sample_data[y:y+3, x:x+3]
                            conv_sum += np.sum(patch * kernel)
                            local_mac_ops += 9
                            time.sleep(0.0001)  # Simulate MAC delay
                    
                    conv_results.append(conv_sum)
                
                # Update statistics thread-safely
                with self.stats_lock:
                    self.mac_operations += local_mac_ops
                
                # Send to next stage
                self.mac_queue.put((sample_id, conv_results))
                self.input_queue.task_done()
                
            except queue.Empty:
                continue  # Check shutdown flag
    
    def relu_stage_worker(self):
        """ReLU activation worker"""
        thread_name = threading.current_thread().name
        print(f"{thread_name}: ReLU stage started")
        
        while not self.shutdown_event.is_set():
            try:
                data = self.mac_queue.get(timeout=1.0)
                
                if data is None:  # Poison pill
                    break
                
                sample_id, conv_results = data
                print(f"{thread_name}: Processing ReLU for sample {sample_id}")
                
                # Apply ReLU activation
                relu_results = []
                local_relu_ops = 0
                
                for conv_val in conv_results:
                    relu_result = max(0, conv_val)
                    relu_results.append(relu_result)
                    local_relu_ops += 1
                    time.sleep(0.00001)  # Simulate ReLU delay
                
                # Update statistics
                with self.stats_lock:
                    self.relu_operations += local_relu_ops
                
                # Send to output
                final_result = np.mean(relu_results)
                self.relu_queue.put((sample_id, final_result))
                self.mac_queue.task_done()
                
            except queue.Empty:
                continue
    
    def output_collector(self):
        """Collect final results"""
        thread_name = threading.current_thread().name
        print(f"{thread_name}: Output collector started")
        
        while not self.shutdown_event.is_set():
            try:
                data = self.relu_queue.get(timeout=1.0)
                
                if data is None:  # Poison pill
                    break
                
                sample_id, result = data
                print(f"{thread_name}: Collected result for sample {sample_id}: {result:.4f}")
                
                self.output_queue.put((sample_id, result))
                self.relu_queue.task_done()
                
            except queue.Empty:
                continue
    
    def process_samples(self, input_data):
        """Process samples through the pipeline"""
        print(f"Starting pipeline with {len(input_data)} samples...")
        
        # Start pipeline workers
        mac_thread = threading.Thread(target=self.mac_stage_worker, name="MAC-Worker")
        relu_thread = threading.Thread(target=self.relu_stage_worker, name="ReLU-Worker")
        output_thread = threading.Thread(target=self.output_collector, name="Output-Collector")
        
        mac_thread.start()
        relu_thread.start()
        output_thread.start()
        
        start_time = time.time()
        
        # Feed samples into pipeline
        for i, sample in enumerate(input_data):
            if len(sample.shape) == 3:
                sample = sample.squeeze()
            self.input_queue.put((i, sample))
            print(f"Fed sample {i} into pipeline")
        
        # Wait for all samples to be processed
        self.input_queue.join()
        self.mac_queue.join()
        self.relu_queue.join()
        
        # Send poison pills to stop workers
        self.input_queue.put(None)
        self.mac_queue.put(None)
        self.relu_queue.put(None)
        
        # Wait for workers to finish
        mac_thread.join()
        relu_thread.join()
        output_thread.join()
        
        execution_time = time.time() - start_time
        
        # Collect results
        results = []
        while not self.output_queue.empty():
            results.append(self.output_queue.get())
        
        # Sort by sample_id
        results.sort(key=lambda x: x[0])
        final_results = [r[1] for r in results]
        
        return {
            'results': np.array(final_results),
            'execution_time': execution_time,
            'mac_operations': self.mac_operations,
            'relu_operations': self.relu_operations,
            'pipeline_stages': 3
        }

def run_pipeline_simulation(input_data):
    """Run the concurrent pipeline simulation"""
    pipeline = HardwarePipeline()
    return pipeline.process_samples(input_data)