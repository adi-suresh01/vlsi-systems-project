# VLSI Systems Project

Power-thermal characterization of sustained AI inference on edge hardware. Combines a Rust simulation framework with real hardware measurements on a Raspberry Pi 5 instrumented with an INA219 current sensor.

## What This Is

This project characterizes what happens when you run sustained neural network inference on a thermally constrained edge device. It includes both a simulation framework (power modeling, RC thermal model, DVFS sweeps, attention scaling analysis) and real hardware experiments measuring temperature, power, and throttling behavior over time.

The core finding: all workloads converge to the same 83 to 85C thermal ceiling regardless of model complexity. A synthetic convolution loop, MobileNetV2 (3.8 inf/s), and SqueezeNet (25.2 inf/s) all draw 5.28 to 5.83W and hit the same steady-state temperature. The thermal envelope is a property of the device, not the workload.

## Hardware Platform

- Raspberry Pi 5 Model B (BCM2712, 4x Cortex-A76, 2.4 GHz stock)
- INA219 I2C current/voltage sensor for system-level power measurement
- No heatsink or fan (intentional, measuring bare thermal behavior)
- RC thermal model calibrated against real data: R_th = 10.67 C/W, C_th = 4.08 J/C, 1.3C MAE

## Project Layout

```
edge-vlsi-monitor/
    Cargo.toml                  Workspace root
    crates/
        vlsi-sim/               Simulation engine (MAC, ReLU, convolution, power, thermal, attention)
        metrics-collector/      Ring buffers, HDR histograms, power tracking
        agent-runtime/          Agent state machine, ONNX inference (tract), model registry
        dashboard/              Axum REST API, WebSocket streaming, static frontend
        cli/                    Command-line interface
    frontend/                   Web dashboard (HTML/CSS/JS, Chart.js)
    rtl/                        SystemVerilog reference modules (MAC, ReLU, pipeline)
plots/                          Experiment analysis and figure generation scripts (Python)
```

## Quick Start

```bash
cd edge-vlsi-monitor
cargo build --release
cargo test --workspace          # 69 tests
```

```bash
cargo run -- sim --samples 10                    # Run a hardware simulation
cargo run -- sim --model mobilenetv2 --samples 5 # Simulate a specific model profile
cargo run -- bench --samples 20 --compare        # Sequential vs parallel comparison
cargo run -- dvfs --samples 5                    # Sweep all DVFS levels
cargo run -- attention-sweep --model tinybert --seq-lengths 64,128,256,512
cargo run -- models                              # List available model profiles
cargo run -- infer --model-path model.onnx --duration 300  # Run real ONNX inference
cargo run -- sysinfo                             # Print host hardware info
cargo run -- dashboard                           # Start the web dashboard on port 8080
```

All commands support `--json` for machine-readable output.

## Key Results

- **RC model validation**: First-order RC thermal model calibrated to 1.3C MAE against INA219 measurements. Default thermal capacitance was 8x too small (0.5 vs 4.08 J/C); thermal resistance was close (10.0 vs 10.67 C/W).
- **Workload-independent thermal ceiling**: Synthetic convolution, MobileNetV2, and SqueezeNet all converge to 83-85C at 5.28-5.83W. Model compression improves throughput but does not reduce thermal burden.
- **Multi-core thermal amplification**: Parallel (4-core) workloads reach the throttle threshold 2.2x faster than sequential (1-core), but both saturate at the same ceiling.
- **Attention scaling barriers**: Transformer attention's O(n^2) scaling creates hard thermal limits. BERT-base exceeds 85C at sequence length 64.

## Documentation

See [edge-vlsi-monitor/README.md](edge-vlsi-monitor/README.md) for the full technical writeup, including the power model equations, thermal model, DVFS operating points, agent lifecycle, and API reference.
## License

MIT

## Author

Aditya Suresh
