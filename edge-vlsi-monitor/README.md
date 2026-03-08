# Edge VLSI AI Monitor

A Rust framework for simulating and profiling the power consumption, thermal behavior, and computational overhead of concurrent AI inference agents on edge hardware.

## Motivation

Most edge AI benchmarking stops at latency and throughput. That tells you how fast your model runs. It does not tell you whether your device can sustain that workload for more than a few minutes before thermal throttling cuts your throughput in half, or whether adding a second agent will push your junction temperature past the point where leakage power dominates your entire power budget.

This project exists because there is a gap between how edge AI systems are evaluated and how they actually behave in deployment. The gap gets wider as the industry moves from single-model inference toward multi-agent orchestration, where several agents share a die, consume context, and compete for the same thermal envelope.

The specific question this framework helps answer: given N agents running inference on shared edge hardware, what is the steady-state power draw, the thermal trajectory, and the point at which DVFS throttling degrades throughput?

## Related Work

This work builds on and differs from several lines of existing research.

**TokenPowerBench** (Ye et al., 2024) benchmarks per-token power consumption of LLMs but focuses on datacenter GPUs, not edge devices, and does not model thermal feedback or multi-agent interaction. **The One-Token Model** (Antarctica.io, 2025) proposes energy-per-token as a standard metric but treats it as a static measurement rather than a dynamic function of temperature and DVFS state.

**Impact of Thermal Throttling on Long-Term Visual Inference** (Rungsuptaweekoon et al., 2020) demonstrates that sustained CNN inference on a Raspberry Pi causes DVFS throttling within minutes, a finding this framework models explicitly through its thermal RC network. Their work is empirical; ours provides a simulation layer that can predict this behavior before deployment.

**CNNParted** (2023) partitions CNN inference across edge nodes and evaluates energy per partition, but does not model the thermal coupling between co-located agents. **DNN+NeuroSim** provides circuit-level power estimation for neural network accelerators but targets ASIC design space exploration rather than runtime agent monitoring.

**ZeroDVFS** (2025) uses LLMs to guide core and frequency allocation on embedded platforms, treating DVFS as an optimization target. Our framework provides the simulation substrate that such an allocator would need: per-agent power breakdowns, thermal state, and operation counts at each DVFS level.

The gap this project fills is the intersection of multi-agent lifecycle management with physics-based power-thermal simulation. Existing tools measure power or simulate hardware, but none provide a unified framework where you can spawn agents, run them through a state machine, and observe the thermal consequences of their combined workload in real time.

## Architecture

The project is a Cargo workspace with five crates. Each crate has a single responsibility, and they compose vertically from simulation primitives up to the web dashboard.

```
vlsi-sim              Simulation engine. MAC, ReLU, convolution, power, thermal.
metrics-collector     Lock-free ring buffers, HDR histograms, power tracking.
agent-runtime         Agent state machine, ONNX inference, async scheduler.
dashboard             Axum REST API, WebSocket streaming, static frontend.
cli                   Command-line interface for all operations.
```

Dependency order: `vlsi-sim` and `metrics-collector` are independent leaf crates. `agent-runtime` depends on both. `dashboard` depends on all three. `cli` depends on everything.

## Power Model

The power estimation follows standard CMOS power equations with DVFS scaling:

```
P_total = P_base + P_dynamic + P_leakage

P_base    = 10 mW * (V / V_nom)
P_dynamic = (N_mac * E_mac + N_relu * E_relu) * V^2 * (f / f_nom)
P_leakage = 1 mW * 2^((T_j - 25) / 10) * (V / V_nom)
```

`E_mac` is 1 nJ per multiply-accumulate operation. `E_relu` is 0.1 nJ per activation. Nominal conditions are 1.0 V at 200 MHz. The leakage term doubles every 10 degrees Celsius above 25C, which is the standard approximation for subthreshold leakage in modern CMOS.

The framework includes five DVFS operating points:

| Mode | Voltage | Frequency | Typical Use |
|------|---------|-----------|-------------|
| Ultra-Low | 0.6 V | 50 MHz | Battery-constrained IoT |
| Low Power | 0.7 V | 100 MHz | Always-on sensing |
| Balanced | 0.8 V | 150 MHz | Mixed workloads |
| Performance | 1.0 V | 200 MHz | Full throughput |
| Turbo | 1.1 V | 250 MHz | Burst inference |

## Thermal Model

Junction temperature evolves according to an RC thermal network:

```
dT_j/dt = (P * R_th - (T_j - T_amb)) / (R_th * C_th)
```

Default parameters: ambient temperature 25C, thermal resistance 10 C/W, thermal capacitance 0.5 J/C, throttle threshold 85C. The steady-state temperature for a given power level is `T_ss = T_amb + P * R_th`. At 11 mW (default DVFS), steady state is 25.11C. At turbo with multiple agents, it can exceed the throttle threshold, at which point the framework signals that DVFS should step down.

## Convolution Simulation

The hardware simulator models a simplified CNN inference pipeline. Each 28x28 input passes through 8 convolutional kernels (3x3, stride 4), followed by ReLU activation and mean pooling. This produces 3,528 MAC operations and 8 ReLU operations per sample. The numbers come from: 8 kernels, 7x7 valid positions per kernel (since `(28-3)/4 + 1 = 7`), and 9 MACs per 3x3 patch.

The convolution is intentionally simplified. The point is not to replicate a production CNN accelerator but to provide a workload whose operation counts are deterministic, verifiable, and directly translatable to power and thermal estimates.

## Agent Lifecycle

Each agent follows a state machine with five states:

```
Idle -> Loading -> Running <-> Paused
                      |            |
                      v            v
                 Terminated   Terminated
```

Transitions are enforced at compile time. An agent cannot skip from Idle to Running, cannot return from Terminated, and cannot reload a model while running. Each agent independently tracks inference latency (HDR histogram with p50/p90/p95/p99), cumulative MAC/ReLU operations, power consumption (1000-sample sliding window), and junction temperature.

## Getting Started

Requirements: Rust 1.75 or later. No other dependencies.

```bash
cd edge-vlsi-monitor
cargo build --release
cargo test --workspace
```

Basic usage:

```bash
# Run a hardware simulation
cargo run -- sim --samples 10

# Compare sequential vs parallel execution
cargo run -- bench --samples 20 --compare

# Sweep all DVFS levels
cargo run -- dvfs --samples 5

# Print system hardware info
cargo run -- sysinfo

# Start the web dashboard on port 8080
cargo run -- dashboard
```

All commands support `--json` for machine-readable output.

## CLI

```
edge-vlsi [--json] [-v] <command>

sim       --samples N  --clock-mhz F  -j THREADS  --height H  --width W
bench     --samples N  --compare
dvfs      --samples N
sysinfo
dashboard --host ADDR  --port PORT  --frontend-dir PATH  --metrics-interval MS
```

Example output from `dvfs`:

```
+-----------+---------+------+------------+--------------+--------+
| Mode      | Voltage | Freq | Power (mW) | HW Time (us) | Temp   |
+===========+=========+======+============+==============+========+
| Ultra-Low | 0.6V    | 50   | 6.60       | 212.16       | 25.0 C |
| Low Power | 0.7V    | 100  | 7.70       | 106.08       | 25.0 C |
| Balanced  | 0.8V    | 150  | 8.81       | 70.72        | 25.0 C |
| Perf      | 1.0V    | 200  | 11.02      | 53.04        | 25.0 C |
| Turbo     | 1.1V    | 250  | 12.12      | 42.43        | 25.0 C |
+-----------+---------+------+------------+--------------+--------+
```

## Dashboard

The web interface runs on Axum with WebSocket-based live updates at 500ms intervals. It exposes 12 REST endpoints and one WebSocket endpoint. The frontend uses Chart.js.

Panels: agent manager (spawn/pause/resume/terminate), performance bar chart, power breakdown doughnut (base/dynamic/leakage), thermal gauge with throttle indicator, hardware metrics table (cycles, MACs, ReLUs, frequency), latency timeline (avg and p99), and an event log.

Agents can be managed through the API:

```bash
# Spawn an agent
curl -X POST localhost:8080/api/agents -H 'Content-Type: application/json' -d '{"name":"agent-1"}'

# Run simulation on it
curl -X POST localhost:8080/api/agents/<id>/simulate -H 'Content-Type: application/json' -d '{"samples":10}'

# Pause, resume, terminate
curl -X POST localhost:8080/api/agents/<id>/pause
curl -X POST localhost:8080/api/agents/<id>/resume
curl -X DELETE localhost:8080/api/agents/<id>
```

## Experiments

**Context scaling.** Run simulations with increasing input dimensions to observe how MAC counts, power, and thermal trajectory scale. With stride-4 convolution, doubling the input side length roughly quadruples the MAC count, which maps directly to dynamic power.

```bash
cargo run -- --json sim --samples 50 --height 14 --width 14
cargo run -- --json sim --samples 50 --height 28 --width 28
cargo run -- --json sim --samples 50 --height 56 --width 56
```

**Multi-agent thermal coupling.** Spawn several agents through the dashboard, trigger concurrent simulations, and observe how the thermal gauge responds. The thermal model accumulates heat from all agents sharing the simulated die.

**DVFS efficiency frontier.** The `dvfs` command sweeps voltage-frequency pairs. Plotting power against throughput (1 / theoretical HW time) reveals the efficiency knee where further frequency increases yield diminishing returns relative to power cost.

**Deployment sizing.** For a given power budget, divide by per-agent power at your target DVFS level to get the maximum agent count. Then verify with the thermal model that the combined workload stays below the throttle threshold.

## Prior Version

This is a rewrite of a Python/C++/SystemVerilog system in the same repository (`vlsi-ai-integration/`). The original used TensorFlow for model training, a Python hardware simulator with per-operation sleep delays, a C++ accelerator with manual mutex locking, and a FastAPI dashboard with HTTP polling. The Rust version replaces all of that with a single 20 MB binary, compile-time concurrency safety, physics-based power and thermal models that the original lacked, and WebSocket streaming for the dashboard.

## Tests

49 tests across all crates. Run with `cargo test --workspace`.

| Crate | Unit | Integration | Covers |
|-------|------|-------------|--------|
| vlsi-sim | 25 | 5 | MAC, ReLU, convolution, pipeline, power, thermal, DVFS |
| metrics-collector | 11 | 0 | Ring buffer, histogram, power tracker |
| agent-runtime | 7 | 3 | State machine, agent lifecycle, metrics |
| dashboard | 0 | 3 | Serialization, WebSocket messages |

## Project Structure

```
edge-vlsi-monitor/
    Cargo.toml
    frontend/
        index.html
        css/styles.css
        js/dashboard.js
    rtl/
        mac_unit.sv
        relu_unit.sv
        pipeline.sv
        testbenches/
    crates/
        vlsi-sim/src/
            mac.rs, relu.rs, conv.rs, pipeline.rs, power.rs, thermal.rs
        metrics-collector/src/
            ring_buffer.rs, histogram.rs, power_tracker.rs
        agent-runtime/src/
            state.rs, agent.rs, scheduler.rs, inference.rs
        dashboard/src/
            rest.rs, ws.rs, state.rs
        cli/src/
            main.rs
```

## License

MIT

## Author

Aditya Suresh
