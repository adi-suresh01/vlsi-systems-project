def compare_inference_performance(cpu_results, hardware_results):
    """
    Compare the performance metrics between CPU and hardware inference.

    Parameters:
    cpu_results (dict): A dictionary containing CPU inference results with keys like 'time', 'accuracy', and 'power'.
    hardware_results (dict): A dictionary containing hardware inference results with keys like 'time', 'accuracy', and 'power'.

    Returns:
    dict: A dictionary containing the comparison results.
    """
    comparison = {
        'time_speedup': cpu_results['time'] / hardware_results['time'],
        'accuracy_difference': cpu_results['accuracy'] - hardware_results['accuracy'],
        'power_difference': cpu_results['power'] - hardware_results['power']
    }
    return comparison

def visualize_comparison(comparison_results):
    """
    Visualize the comparison results using a simple text output.

    Parameters:
    comparison_results (dict): A dictionary containing the comparison results.
    """
    print("Inference Performance Comparison:")
    print(f"Speedup: {comparison_results['time_speedup']}x")
    print(f"Accuracy Difference: {comparison_results['accuracy_difference']:.2f}%")
    print(f"Power Difference: {comparison_results['power_difference']}W")