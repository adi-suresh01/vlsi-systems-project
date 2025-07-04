"""
Race Condition Demonstration
Shows the importance of thread synchronization in concurrent hardware simulation
"""

import threading
import time
import random

# Shared resource - this will demonstrate race conditions
shared_power_consumption = 0
power_lock = threading.Lock()

def unsafe_power_update(thread_id, iterations=1000):
    """Demonstrates race condition with unsynchronized shared variable access"""
    global shared_power_consumption
    
    for i in range(iterations):
        # RACE CONDITION: Multiple threads reading/writing simultaneously
        current_power = shared_power_consumption
        time.sleep(0.000001)  # Simulate computation time
        new_power = current_power + 0.001  # Add 1mW
        shared_power_consumption = new_power
        
        if i % 100 == 0:
            print(f"Thread {thread_id}: Power = {shared_power_consumption:.3f}W")

def safe_power_update(thread_id, iterations=1000):
    """Thread-safe version using mutex locks"""
    global shared_power_consumption
    
    for i in range(iterations):
        with power_lock:  # CRITICAL SECTION: Only one thread can access at a time
            current_power = shared_power_consumption
            time.sleep(0.000001)
            new_power = current_power + 0.001
            shared_power_consumption = new_power
        
        if i % 100 == 0:
            with power_lock:
                print(f"Thread {thread_id}: Safe Power = {shared_power_consumption:.3f}W")

def demonstrate_race_condition():
    """Demonstrates the difference between unsafe and thread-safe operations"""
    global shared_power_consumption
    
    print("=== RACE CONDITION DEMONSTRATION ===")
    
    # Test 1: Unsafe (race condition)
    print("\n1. UNSAFE - Race Condition:")
    shared_power_consumption = 0
    threads = []
    
    for i in range(3):
        t = threading.Thread(target=unsafe_power_update, args=(i, 500))
        threads.append(t)
        t.start()
    
    for t in threads:
        t.join()
    
    expected = 3 * 500 * 0.001  # 3 threads * 500 iterations * 0.001W
    print(f"Expected: {expected:.3f}W, Actual: {shared_power_consumption:.3f}W")
    print(f"Data loss: {expected - shared_power_consumption:.3f}W")
    
    # Test 2: Safe (with locks)
    print("\n2. SAFE - With Locks:")
    shared_power_consumption = 0
    threads = []
    
    for i in range(3):
        t = threading.Thread(target=safe_power_update, args=(i, 500))
        threads.append(t)
        t.start()
    
    for t in threads:
        t.join()
    
    print(f"Expected: {expected:.3f}W, Actual: {shared_power_consumption:.3f}W")
    print(f"Data loss: {expected - shared_power_consumption:.3f}W")

if __name__ == "__main__":
    demonstrate_race_condition()