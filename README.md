# VLSI Systems Project

Power consumption and thermal simulation for concurrent AI inference agents on edge hardware, built in Rust.

## What This Is

This project simulates what happens when you run multiple AI agents on a shared edge device. It models the power draw, junction temperature, and DVFS throttling behavior that real deployments face but most benchmarks ignore.

The core framework lives in `edge-vlsi-monitor/`. It is a Cargo workspace with five crates covering hardware simulation, metrics collection, agent lifecycle management, a web dashboard, and a CLI.

## Why It Exists

Edge AI benchmarking typically reports latency and throughput for a single model on a cool device. That tells you almost nothing about sustained multi-agent workloads, where thermal coupling between co-located agents causes DVFS throttling that degrades throughput for everyone.

This project models that behavior explicitly. Given N agents sharing a die, it computes steady-state power, thermal trajectory, and the point where throttling kicks in.

## Project Layout

```
edge-vlsi-monitor/
    Cargo.toml                  Workspace root
    crates/
        vlsi-sim/               Simulation engine (MAC, ReLU, convolution, power, thermal)
        metrics-collector/      Lock-free ring buffers, HDR histograms, power tracking
        agent-runtime/          Agent state machine, ONNX inference, async scheduler
        dashboard/              Axum REST API, WebSocket streaming, static frontend
        cli/                    Command-line interface
    frontend/                   Web dashboard (HTML/CSS/JS, Chart.js)
    rtl/                        SystemVerilog reference modules (MAC, ReLU, pipeline)
```

## Quick Start

```bash
cd edge-vlsi-monitor
cargo build --release
cargo test --workspace
```

```bash
cargo run -- sim --samples 10          # Run a hardware simulation
cargo run -- bench --samples 20 --compare  # Sequential vs parallel comparison
cargo run -- dvfs --samples 5          # Sweep all DVFS levels
cargo run -- sysinfo                   # Print host hardware info
cargo run -- dashboard                 # Start the web dashboard on port 8080
```

All commands support `--json` for machine-readable output.

## Documentation

See [edge-vlsi-monitor/README.md](edge-vlsi-monitor/README.md) for the full technical writeup, including the power model equations, thermal model, DVFS operating points, agent lifecycle, experiment guide, and API reference.

## Prior Version

This repository previously contained a Python/C++/SystemVerilog implementation (`vlsi-ai-integration/`). The Rust rewrite replaces all of that with a single binary, compile-time concurrency safety, physics-based power and thermal models, and a WebSocket-based dashboard. The SystemVerilog RTL modules are retained as reference designs in `edge-vlsi-monitor/rtl/`.

## License

MIT

## Author

Aditya Suresh
