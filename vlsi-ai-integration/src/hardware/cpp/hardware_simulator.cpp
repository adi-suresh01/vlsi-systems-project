#include <vector>
#include <chrono>
#include <thread>
#include <future>
#include <iostream>
#include <random>
#include <mutex>
#include <algorithm>

class HardwareSimulator {
private:
    std::mutex stats_mutex;
    uint64_t total_mac_ops = 0;
    uint64_t total_relu_ops = 0;
    double total_power = 0.0;
    
public:
    struct SimulationResult {
        std::vector<double> results;
        double execution_time;
        uint64_t mac_operations;
        uint64_t relu_operations;
        double power_consumption;
        double throughput;
    };
    
    // High-performance MAC operation
    inline double simulate_mac_hw(double a, double b, double c) {
        return a * b + c;
    }
    
    // High-performance ReLU
    inline double simulate_relu_hw(double x) {
        return std::max(0.0, x);
    }
    
    // Process single sample with optimized loops
    double process_sample(const std::vector<std::vector<double>>& sample, int sample_id) {
        uint64_t local_mac_ops = 0;
        uint64_t local_relu_ops = 0;
        double local_power = 0.0;
        
        std::vector<double> conv_results;
        conv_results.reserve(8);
        
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<double> dist(0.0, 0.1);
        
        // Simulate 8 feature maps
        for (int kernel_idx = 0; kernel_idx < 8; ++kernel_idx) {
            double conv_sum = 0.0;
            
            // Optimized convolution with stride 4
            for (int y = 0; y < static_cast<int>(sample.size()) - 2; y += 4) {
                for (int x = 0; x < static_cast<int>(sample[0].size()) - 2; x += 4) {
                    // 3x3 convolution
                    for (int ky = 0; ky < 3; ++ky) {
                        for (int kx = 0; kx < 3; ++kx) {
                            double kernel_val = dist(gen);
                            double pixel_val = sample[y + ky][x + kx];
                            conv_sum = simulate_mac_hw(pixel_val, kernel_val, conv_sum);
                            local_mac_ops++;
                            local_power += 1e-9; // 1nW per MAC
                        }
                    }
                }
            }
            
            double relu_result = simulate_relu_hw(conv_sum);
            local_relu_ops++;
            local_power += 1e-10; // 0.1nW per ReLU
            
            conv_results.push_back(relu_result);
        }
        
        // Update global statistics thread-safely
        {
            std::lock_guard<std::mutex> lock(stats_mutex);
            total_mac_ops += local_mac_ops;
            total_relu_ops += local_relu_ops;
            total_power += local_power;
        }
        
        double sum = 0.0;
        for (double val : conv_results) {
            sum += val;
        }
        return sum / conv_results.size();
    }
    
    // Multi-threaded simulation
    SimulationResult run_simulation(const std::vector<std::vector<std::vector<double>>>& input_data, 
                                   int num_threads = std::thread::hardware_concurrency()) {
        
        auto start_time = std::chrono::high_resolution_clock::now();
        
        std::cout << "Running C++ hardware simulation with " << num_threads << " threads..." << std::endl;
        
        total_mac_ops = 0;
        total_relu_ops = 0;
        total_power = 0.0;
        
        std::vector<std::future<double>> futures;
        std::vector<double> results;
        results.resize(input_data.size());
        
        for (size_t i = 0; i < input_data.size(); ++i) {
            futures.push_back(
                std::async(std::launch::async, 
                          [this, &input_data, i]() {
                              return this->process_sample(input_data[i], i);
                          })
            );
        }
        
        for (size_t i = 0; i < futures.size(); ++i) {
            results[i] = futures[i].get();
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        double execution_time = duration.count() / 1000000.0;
        
        return {
            results,
            execution_time,
            total_mac_ops,
            total_relu_ops,
            total_power,
            static_cast<double>(input_data.size()) / execution_time
        };
    }
};

// C interface for Python binding
extern "C" {
    HardwareSimulator* create_simulator() {
        return new HardwareSimulator();
    }
    
    void destroy_simulator(HardwareSimulator* sim) {
        delete sim;
    }
    
    void run_cpp_simulation(HardwareSimulator* sim, 
                           double* input_data, int samples, int height, int width,
                           double* results, double* execution_time, 
                           uint64_t* mac_ops, uint64_t* relu_ops, double* power) {
        
        std::vector<std::vector<std::vector<double>>> data(samples);
        for (int s = 0; s < samples; ++s) {
            data[s].resize(height);
            for (int h = 0; h < height; ++h) {
                data[s][h].resize(width);
                for (int w = 0; w < width; ++w) {
                    data[s][h][w] = input_data[s * height * width + h * width + w];
                }
            }
        }
        
        auto result = sim->run_simulation(data);
        
        for (int i = 0; i < samples; ++i) {
            results[i] = result.results[i];
        }
        *execution_time = result.execution_time;
        *mac_ops = result.mac_operations;
        *relu_ops = result.relu_operations;
        *power = result.power_consumption;
    }
}