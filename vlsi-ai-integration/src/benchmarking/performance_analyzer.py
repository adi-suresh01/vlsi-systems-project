from typing import List, Dict
import time

class PerformanceAnalyzer:
    def __init__(self):
        self.results: List[Dict[str, float]] = []

    def log_performance(self, model_name: str, execution_time: float, power_consumption: float):
        self.results.append({
            'model_name': model_name,
            'execution_time': execution_time,
            'power_consumption': power_consumption
        })

    def analyze_performance(self):
        for result in self.results:
            print(f"Model: {result['model_name']}, Execution Time: {result['execution_time']}s, Power Consumption: {result['power_consumption']}W")

    def compare_performance(self, cpu_time: float, hardware_time: float, cpu_power: float, hardware_power: float):
        speedup = cpu_time / hardware_time
        power_savings = cpu_power - hardware_power
        print(f"Speedup: {speedup}x, Power Savings: {power_savings}W")

    def reset(self):
        self.results.clear()