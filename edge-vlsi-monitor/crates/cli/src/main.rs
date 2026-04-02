use std::net::SocketAddr;
use std::path::PathBuf;

use anyhow::Result;
use clap::{Parser, Subcommand};
use comfy_table::{Table, presets::UTF8_FULL};

use vlsi_sim::{
    generate_test_samples, run_simulation, run_simulation_parallel, run_simulation_from_profile,
    DvfsConfig, PipelineConfig, ThermalConfig,
};
use agent_runtime::ModelRegistry;

#[derive(Parser)]
#[command(name = "edge-vlsi", about = "Edge VLSI AI Monitor - Rust-based hardware simulation & agent monitoring")]
struct Cli {
    #[command(subcommand)]
    command: Commands,

    /// Enable JSON output
    #[arg(long, global = true)]
    json: bool,

    /// Verbosity level (-v, -vv, -vvv)
    #[arg(short, long, action = clap::ArgAction::Count, global = true)]
    verbose: u8,
}

#[derive(Subcommand)]
enum Commands {
    /// Run hardware simulation
    Sim {
        /// Number of samples to simulate
        #[arg(short, long, default_value = "10")]
        samples: usize,

        /// Clock frequency in MHz
        #[arg(long, default_value = "200")]
        clock_mhz: f64,

        /// Number of parallel threads (0 = auto-detect)
        #[arg(short = 'j', long, default_value = "0")]
        threads: usize,

        /// Sample image height
        #[arg(long, default_value = "28")]
        height: usize,

        /// Sample image width
        #[arg(long, default_value = "28")]
        width: usize,

        /// Model name from registry (e.g., tinybert, distilgpt2, mobilenetv2).
        /// When set, uses workload profile instead of synthetic convolution.
        #[arg(long)]
        model: Option<String>,

        /// Sequence length for transformer models
        #[arg(long, default_value = "128")]
        seq_length: usize,
    },

    /// Run full benchmark suite
    Bench {
        /// Number of samples
        #[arg(short, long, default_value = "10")]
        samples: usize,

        /// Compare sequential vs parallel
        #[arg(long)]
        compare: bool,
    },

    /// Launch the web dashboard
    Dashboard {
        /// Bind address
        #[arg(long, default_value = "127.0.0.1")]
        host: String,

        /// Port
        #[arg(short, long, default_value = "8080")]
        port: u16,

        /// Frontend directory
        #[arg(long, default_value = "frontend")]
        frontend_dir: PathBuf,

        /// Metrics broadcast interval (ms)
        #[arg(long, default_value = "500")]
        metrics_interval: u64,
    },

    /// Show system information
    Sysinfo,

    /// Test DVFS power scaling
    Dvfs {
        /// Number of samples for each test
        #[arg(short, long, default_value = "5")]
        samples: usize,
    },

    /// Sweep sequence lengths for transformer attention power analysis
    AttentionSweep {
        /// Transformer model: tinybert, distilgpt2, bert-base
        #[arg(long, default_value = "distilgpt2")]
        model: String,

        /// Comma-separated sequence lengths to sweep
        #[arg(long, value_delimiter = ',', default_value = "128,256,512,1024,2048")]
        seq_lengths: Vec<usize>,

        /// Number of inferences per data point
        #[arg(long, default_value = "1")]
        inferences: u64,

        /// DVFS level: 0=ultra-low, 1=low, 2=balanced, 3=performance, 4=turbo
        #[arg(long)]
        dvfs_level: Option<usize>,

        /// Output as CSV
        #[arg(long)]
        csv: bool,
    },

    /// List available models in the registry
    Models,

    /// Run real ONNX model inference in a loop for power/thermal experiments.
    Infer {
        /// Path to ONNX model file
        #[arg(short, long)]
        model_path: PathBuf,

        /// Duration in seconds
        #[arg(short, long, default_value = "300")]
        duration: u64,

        /// Batch size per inference
        #[arg(short, long, default_value = "1")]
        batch_size: usize,

        /// Path to write per-inference latency CSV (timestamp_epoch_ms,inference_num,latency_ms)
        #[arg(short, long)]
        latency_log: Option<PathBuf>,
    },

    /// Run sustained CPU stress for thermal experiments.
    /// Loops the simulation continuously for a fixed duration.
    Stress {
        /// Duration in seconds
        #[arg(short, long, default_value = "300")]
        duration: u64,

        /// Samples per batch (larger = more work per iteration)
        #[arg(short, long, default_value = "500")]
        samples: usize,

        /// Number of parallel threads (0 = all cores)
        #[arg(short = 'j', long, default_value = "0")]
        threads: usize,

        /// Sample height
        #[arg(long, default_value = "28")]
        height: usize,

        /// Sample width
        #[arg(long, default_value = "28")]
        width: usize,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    let log_level = match cli.verbose {
        0 => "info",
        1 => "debug",
        _ => "trace",
    };

    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new(log_level)),
        )
        .init();

    match cli.command {
        Commands::Sim {
            samples,
            clock_mhz,
            threads,
            height,
            width,
            model,
            seq_length,
        } => cmd_sim(samples, clock_mhz, threads, height, width, model, seq_length, cli.json),

        Commands::Bench { samples, compare } => cmd_bench(samples, compare, cli.json),

        Commands::Dashboard {
            host,
            port,
            frontend_dir,
            metrics_interval,
        } => cmd_dashboard(host, port, frontend_dir, metrics_interval).await,

        Commands::Sysinfo => cmd_sysinfo(cli.json),

        Commands::Dvfs { samples } => cmd_dvfs(samples, cli.json),

        Commands::AttentionSweep {
            model,
            seq_lengths,
            inferences,
            dvfs_level,
            csv,
        } => cmd_attention_sweep(&model, &seq_lengths, inferences, dvfs_level, csv, cli.json),

        Commands::Models => cmd_models(cli.json),

        Commands::Infer {
            model_path,
            duration,
            batch_size,
            latency_log,
        } => cmd_infer(model_path, duration, batch_size, latency_log),

        Commands::Stress {
            duration,
            samples,
            threads,
            height,
            width,
        } => cmd_stress(duration, samples, threads, height, width),
    }
}

fn cmd_sim(
    samples: usize,
    clock_mhz: f64,
    threads: usize,
    height: usize,
    width: usize,
    model: Option<String>,
    seq_length: usize,
    json: bool,
) -> Result<()> {
    let mut dvfs = DvfsConfig::default();
    dvfs.frequency_mhz = clock_mhz;

    if let Some(model_name) = model {
        let registry = ModelRegistry::new();
        let profile = registry
            .workload_profile(&model_name, Some(seq_length), samples as u64)
            .ok_or_else(|| anyhow::anyhow!(
                "Unknown model '{}'. Run 'models' to see available models.", model_name
            ))?;

        println!("Running {} simulation ({} inferences, seq_length={}, {:.0} MHz)...",
            profile.model_name, samples, seq_length, clock_mhz);
        println!("  MACs per inference: {:.2}M", profile.mac_ops as f64 / 1e6);
        println!("  Activations per inference: {:.2}M", profile.activation_ops as f64 / 1e6);

        let thermal_config = ThermalConfig::default();
        let result = run_simulation_from_profile(&profile, &dvfs, &thermal_config);

        if json {
            println!("{}", serde_json::to_string_pretty(&result)?);
        } else {
            print_simulation_result(&result, samples);
        }
    } else {
        let test_data = generate_test_samples(samples, height, width);
        let config = PipelineConfig {
            dvfs,
            ..PipelineConfig::default()
        };

        let result = if threads == 0 || threads == 1 {
            println!("Running sequential simulation ({} samples, {}x{}, {} MHz)...",
                samples, height, width, clock_mhz);
            run_simulation(&test_data, &config)
        } else {
            println!("Running parallel simulation ({} samples, {} threads, {} MHz)...",
                samples, threads, clock_mhz);
            run_simulation_parallel(&test_data, &config, Some(threads))
        };

        if json {
            println!("{}", serde_json::to_string_pretty(&result)?);
        } else {
            print_simulation_result(&result, samples);
        }
    }

    Ok(())
}

fn cmd_bench(samples: usize, compare: bool, json: bool) -> Result<()> {
    let test_data = generate_test_samples(samples, 28, 28);
    let config = PipelineConfig::default();

    println!("Running benchmark ({} samples)...\n", samples);

    // Sequential
    let seq_result = run_simulation(&test_data, &config);

    if compare {
        // Parallel
        let par_result = run_simulation_parallel(&test_data, &config, None);

        if json {
            println!("{}", serde_json::to_string_pretty(&serde_json::json!({
                "sequential": seq_result,
                "parallel": par_result,
                "speedup": seq_result.execution_time_secs / par_result.execution_time_secs
            }))?);
        } else {
            let mut table = Table::new();
            table.load_preset(UTF8_FULL);
            table.set_header(vec!["Metric", "Sequential", "Parallel"]);
            table.add_row(vec![
                "Execution Time".into(),
                format!("{:.4} s", seq_result.execution_time_secs),
                format!("{:.4} s", par_result.execution_time_secs),
            ]);
            table.add_row(vec![
                "Throughput".into(),
                format!("{:.1} samples/s", seq_result.throughput_samples_per_sec),
                format!("{:.1} samples/s", par_result.throughput_samples_per_sec),
            ]);
            table.add_row(vec![
                "MAC Operations".into(),
                format!("{}", seq_result.mac_operations),
                format!("{}", par_result.mac_operations),
            ]);
            table.add_row(vec![
                "Speedup".into(),
                "1.00x".into(),
                format!("{:.2}x", seq_result.execution_time_secs / par_result.execution_time_secs),
            ]);
            println!("{table}");
        }
    } else {
        if json {
            println!("{}", serde_json::to_string_pretty(&seq_result)?);
        } else {
            print_simulation_result(&seq_result, samples);
        }
    }

    Ok(())
}

async fn cmd_dashboard(
    host: String,
    port: u16,
    frontend_dir: PathBuf,
    metrics_interval: u64,
) -> Result<()> {
    let addr: SocketAddr = format!("{}:{}", host, port).parse()?;

    // Start the scheduler
    let (mut scheduler, handle) = agent_runtime::Scheduler::new(64);
    tokio::spawn(async move {
        scheduler.run().await;
    });

    println!("Starting Edge VLSI Monitor dashboard at http://{}:{}", host, port);
    println!("Press Ctrl+C to stop\n");

    dashboard::serve(addr, &frontend_dir, handle, metrics_interval).await?;

    Ok(())
}

fn cmd_sysinfo(json: bool) -> Result<()> {
    use sysinfo::System;

    let mut sys = System::new_all();
    sys.refresh_all();

    if json {
        println!("{}", serde_json::to_string_pretty(&serde_json::json!({
            "cpu_count": sys.cpus().len(),
            "cpu_brand": sys.cpus().first().map(|c| c.brand().to_string()),
            "total_memory_gb": sys.total_memory() as f64 / 1_073_741_824.0,
            "used_memory_gb": sys.used_memory() as f64 / 1_073_741_824.0,
            "os": System::long_os_version(),
            "hostname": System::host_name(),
        }))?);
    } else {
        let mut table = Table::new();
        table.load_preset(UTF8_FULL);
        table.set_header(vec!["Property", "Value"]);

        if let Some(cpu) = sys.cpus().first() {
            table.add_row(vec!["CPU".into(), cpu.brand().to_string()]);
        }
        table.add_row(vec![
            "CPU Cores".into(),
            format!("{}", sys.cpus().len()),
        ]);
        table.add_row(vec![
            "Total Memory".into(),
            format!("{:.1} GB", sys.total_memory() as f64 / 1_073_741_824.0),
        ]);
        table.add_row(vec![
            "Used Memory".into(),
            format!("{:.1} GB", sys.used_memory() as f64 / 1_073_741_824.0),
        ]);
        if let Some(os) = System::long_os_version() {
            table.add_row(vec!["OS".into(), os]);
        }
        if let Some(host) = System::host_name() {
            table.add_row(vec!["Hostname".into(), host]);
        }

        println!("{table}");
    }

    Ok(())
}

fn cmd_dvfs(samples: usize, json: bool) -> Result<()> {
    let test_data = generate_test_samples(samples, 28, 28);

    let dvfs_levels = vec![
        ("Ultra-Low", 0.6, 50.0),
        ("Low Power", 0.7, 100.0),
        ("Balanced", 0.8, 150.0),
        ("Performance", 1.0, 200.0),
        ("Turbo", 1.1, 250.0),
    ];

    println!("DVFS Power Scaling Analysis ({} samples)\n", samples);

    let mut results = Vec::new();

    for (name, voltage, freq) in &dvfs_levels {
        let config = PipelineConfig {
            dvfs: DvfsConfig {
                voltage: *voltage,
                frequency_mhz: *freq,
                ..DvfsConfig::default()
            },
            ..PipelineConfig::default()
        };

        let result = run_simulation(&test_data, &config);
        results.push((name, voltage, freq, result));
    }

    if json {
        let json_results: Vec<_> = results
            .iter()
            .map(|(name, v, f, r)| {
                serde_json::json!({
                    "name": name,
                    "voltage": v,
                    "frequency_mhz": f,
                    "power_w": r.estimated_power_w,
                    "theoretical_time_s": r.theoretical_hw_time_secs,
                })
            })
            .collect();
        println!("{}", serde_json::to_string_pretty(&json_results)?);
    } else {
        let mut table = Table::new();
        table.load_preset(UTF8_FULL);
        table.set_header(vec!["Mode", "Voltage", "Freq (MHz)", "Power (mW)", "HW Time (us)", "Temp (C)"]);

        for (name, voltage, freq, result) in &results {
            table.add_row(vec![
                name.to_string(),
                format!("{:.1}V", voltage),
                format!("{:.0}", freq),
                format!("{:.4}", result.estimated_power_w * 1000.0),
                format!("{:.2}", result.theoretical_hw_time_secs * 1_000_000.0),
                format!("{:.1}", result.thermal_state.junction_temp_c),
            ]);
        }

        println!("{table}");
    }

    Ok(())
}

fn cmd_attention_sweep(
    model_name: &str,
    seq_lengths: &[usize],
    num_inferences: u64,
    dvfs_level: Option<usize>,
    csv: bool,
    json: bool,
) -> Result<()> {
    let config = match model_name {
        "tinybert" => vlsi_sim::TransformerConfig::tinybert(),
        "distilgpt2" => vlsi_sim::TransformerConfig::distilgpt2(),
        "bert-base" => vlsi_sim::TransformerConfig::bert_base(),
        other => anyhow::bail!("Unknown transformer model: '{}'. Use tinybert, distilgpt2, or bert-base.", other),
    };

    let mut dvfs = DvfsConfig::default();
    if let Some(level) = dvfs_level {
        if let Some(&(v, f)) = dvfs.levels.get(level) {
            dvfs.voltage = v;
            dvfs.frequency_mhz = f;
        } else {
            anyhow::bail!("Invalid DVFS level {}. Use 0-4.", level);
        }
    }
    let thermal_config = ThermalConfig::default();

    let ops_list = vlsi_sim::sweep_sequence_lengths(&config, seq_lengths);

    if csv {
        println!("seq_length,total_macs,attention_macs,linear_macs,total_activations,power_mw,theoretical_hw_time_ms,steady_state_temp_c");
        for ops in &ops_list {
            let profile = vlsi_sim::transformer_ops_to_profile(ops, num_inferences);
            let sim = run_simulation_from_profile(&profile, &dvfs, &thermal_config);
            let thermal = vlsi_sim::ThermalModel::new(thermal_config.clone());
            let steady_temp = thermal.steady_state_temp(sim.estimated_power_w);

            println!("{},{},{},{},{},{:.4},{:.6},{:.1}",
                ops.seq_length,
                ops.total_macs,
                ops.attention_quadratic_macs,
                ops.linear_macs,
                ops.total_activations,
                sim.estimated_power_w * 1000.0,
                sim.theoretical_hw_time_secs * 1000.0,
                steady_temp,
            );
        }
        return Ok(());
    }

    let mut rows = Vec::new();
    for ops in &ops_list {
        let profile = vlsi_sim::transformer_ops_to_profile(ops, num_inferences);
        let sim = run_simulation_from_profile(&profile, &dvfs, &thermal_config);
        let thermal = vlsi_sim::ThermalModel::new(thermal_config.clone());
        let steady_temp = thermal.steady_state_temp(sim.estimated_power_w);
        rows.push((ops, sim, steady_temp));
    }

    if json {
        let json_rows: Vec<_> = rows
            .iter()
            .map(|(ops, sim, temp)| {
                serde_json::json!({
                    "seq_length": ops.seq_length,
                    "total_macs": ops.total_macs,
                    "attention_quadratic_macs": ops.attention_quadratic_macs,
                    "linear_macs": ops.linear_macs,
                    "total_activations": ops.total_activations,
                    "power_mw": sim.estimated_power_w * 1000.0,
                    "theoretical_hw_time_ms": sim.theoretical_hw_time_secs * 1000.0,
                    "steady_state_temp_c": temp,
                })
            })
            .collect();
        println!("{}", serde_json::to_string_pretty(&json_rows)?);
    } else {
        println!("\nAttention Sweep: {} (V={:.1}V, f={:.0}MHz, {} inference(s))\n",
            config.name, dvfs.voltage, dvfs.frequency_mhz, num_inferences);

        let mut table = Table::new();
        table.load_preset(UTF8_FULL);
        table.set_header(vec![
            "Seq Len", "Total MACs", "Attn MACs (n^2)", "Linear MACs",
            "Power (mW)", "HW Time (ms)", "Steady T (C)",
        ]);
        for (ops, sim, temp) in &rows {
            table.add_row(vec![
                format!("{}", ops.seq_length),
                format_mac_count(ops.total_macs),
                format_mac_count(ops.attention_quadratic_macs),
                format_mac_count(ops.linear_macs),
                format!("{:.3}", sim.estimated_power_w * 1000.0),
                format!("{:.3}", sim.theoretical_hw_time_secs * 1000.0),
                format!("{:.1}", temp),
            ]);
        }
        println!("{table}");
    }

    Ok(())
}

fn cmd_models(json: bool) -> Result<()> {
    let registry = ModelRegistry::new();
    let names = registry.list_names();

    if json {
        let entries: Vec<_> = names
            .iter()
            .filter_map(|name| {
                registry.get(name).map(|e| {
                    serde_json::json!({
                        "key": name,
                        "name": e.name,
                        "arch_type": format!("{:?}", e.arch_type),
                        "input_shape": e.input_shape,
                    })
                })
            })
            .collect();
        println!("{}", serde_json::to_string_pretty(&entries)?);
    } else {
        let mut table = Table::new();
        table.load_preset(UTF8_FULL);
        table.set_header(vec!["Key", "Name", "Type", "Input Shape"]);
        for name in &names {
            if let Some(entry) = registry.get(name) {
                let arch = match &entry.arch_type {
                    agent_runtime::ModelArchType::Cnn => "CNN".to_string(),
                    agent_runtime::ModelArchType::Transformer(c) => {
                        format!("Transformer ({}L, d={})", c.num_layers, c.d_model)
                    }
                    agent_runtime::ModelArchType::Custom => "Custom".to_string(),
                };
                table.add_row(vec![
                    name.to_string(),
                    entry.name.clone(),
                    arch,
                    format!("{:?}", entry.input_shape),
                ]);
            }
        }
        println!("{table}");
    }

    Ok(())
}

fn cmd_infer(model_path: PathBuf, duration_secs: u64, batch_size: usize, latency_log: Option<PathBuf>) -> Result<()> {
    use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
    use std::io::Write;
    use agent_runtime::OnnxModel;

    eprintln!("Loading ONNX model from {}...", model_path.display());
    let model = OnnxModel::load(&model_path)?;
    let input_shape = model.input_shape().to_vec();
    let input_elements: usize = input_shape.iter().product::<usize>() / input_shape[0].max(1) * batch_size;

    eprintln!(
        "Model loaded. Input shape: {:?}, batch_size: {}, duration: {}s",
        input_shape, batch_size, duration_secs
    );

    // Set up latency log file if requested
    let mut log_writer = if let Some(ref log_path) = latency_log {
        let mut f = std::fs::File::create(log_path)?;
        writeln!(f, "timestamp_epoch_ms,inference_num,latency_ms")?;
        eprintln!("Logging per-inference latency to {}", log_path.display());
        Some(f)
    } else {
        None
    };

    // Generate random input data (reused across iterations to avoid allocation overhead)
    let input_data: Vec<f32> = (0..input_elements)
        .map(|i| ((i as f32 * 0.001).sin() + 1.0) / 2.0)
        .collect();

    let deadline = Instant::now() + Duration::from_secs(duration_secs);
    let start = Instant::now();
    let mut iteration = 0u64;
    let mut total_latency_us = 0u64;
    let mut last_report = Instant::now();

    while Instant::now() < deadline {
        let result = model.run(&input_data, batch_size)?;
        iteration += 1;
        total_latency_us += result.latency_us;

        // Write per-inference latency to CSV
        if let Some(ref mut writer) = log_writer {
            let epoch_ms = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_millis();
            let latency_ms = result.latency_us as f64 / 1000.0;
            writeln!(writer, "{},{},{:.3}", epoch_ms, iteration, latency_ms)?;
        }

        if last_report.elapsed() >= Duration::from_secs(10) {
            let elapsed = start.elapsed().as_secs();
            let avg_latency_ms = total_latency_us as f64 / iteration as f64 / 1000.0;
            eprintln!(
                "  [{}/{}s] {} inferences, avg latency: {:.1}ms, throughput: {:.1} inf/s",
                elapsed,
                duration_secs,
                iteration,
                avg_latency_ms,
                iteration as f64 / start.elapsed().as_secs_f64()
            );
            last_report = Instant::now();
        }
    }

    // Flush the log
    if let Some(ref mut writer) = log_writer {
        writer.flush()?;
    }

    let elapsed = start.elapsed().as_secs_f64();
    let avg_latency_ms = total_latency_us as f64 / iteration as f64 / 1000.0;
    eprintln!(
        "Done: {} inferences in {:.1}s ({:.1} inf/s, avg latency: {:.1}ms)",
        iteration,
        elapsed,
        iteration as f64 / elapsed,
        avg_latency_ms
    );

    Ok(())
}

fn cmd_stress(
    duration_secs: u64,
    samples: usize,
    threads: usize,
    height: usize,
    width: usize,
) -> Result<()> {
    use std::time::{Duration, Instant};

    let config = PipelineConfig::default();
    let deadline = Instant::now() + Duration::from_secs(duration_secs);
    let effective_threads = if threads == 0 {
        num_cpus::get()
    } else {
        threads
    };

    eprintln!(
        "Stress test: {} threads, {} samples/batch, {}x{}, {} seconds",
        effective_threads, samples, height, width, duration_secs
    );

    let mut iteration = 0u64;
    let mut total_samples = 0u64;
    let start = Instant::now();
    let mut last_report = Instant::now();

    while Instant::now() < deadline {
        let test_data = generate_test_samples(samples, height, width);
        if effective_threads <= 1 {
            run_simulation(&test_data, &config);
        } else {
            run_simulation_parallel(&test_data, &config, Some(effective_threads));
        }
        iteration += 1;
        total_samples += samples as u64;

        if last_report.elapsed() >= Duration::from_secs(10) {
            let elapsed = start.elapsed().as_secs();
            eprintln!(
                "  [{}/{}s] {} iterations, {} total samples",
                elapsed, duration_secs, iteration, total_samples
            );
            last_report = Instant::now();
        }
    }

    let elapsed = start.elapsed().as_secs_f64();
    eprintln!(
        "Done: {} iterations, {} samples in {:.1}s ({:.0} samples/s)",
        iteration,
        total_samples,
        elapsed,
        total_samples as f64 / elapsed
    );

    Ok(())
}

fn format_mac_count(macs: u64) -> String {
    if macs >= 1_000_000_000 {
        format!("{:.2}B", macs as f64 / 1e9)
    } else if macs >= 1_000_000 {
        format!("{:.1}M", macs as f64 / 1e6)
    } else if macs >= 1_000 {
        format!("{:.1}K", macs as f64 / 1e3)
    } else {
        format!("{}", macs)
    }
}

fn print_simulation_result(result: &vlsi_sim::SimulationResult, samples: usize) {
    let mut table = Table::new();
    table.load_preset(UTF8_FULL);
    table.set_header(vec!["Metric", "Value"]);

    table.add_row(vec!["Samples".into(), format!("{}", samples)]);
    table.add_row(vec![
        "Execution Time".into(),
        format!("{:.4} s", result.execution_time_secs),
    ]);
    table.add_row(vec![
        "Theoretical HW Time".into(),
        format!("{:.2} us", result.theoretical_hw_time_secs * 1_000_000.0),
    ]);
    table.add_row(vec![
        "Clock Frequency".into(),
        format!("{:.0} MHz", result.clock_frequency_mhz),
    ]);
    table.add_row(vec![
        "Simulated Cycles".into(),
        format!("{}", result.simulated_cycles),
    ]);
    table.add_row(vec![
        "MAC Operations".into(),
        format!("{}", result.mac_operations),
    ]);
    table.add_row(vec![
        "ReLU Operations".into(),
        format!("{}", result.relu_operations),
    ]);
    table.add_row(vec![
        "Total Operations".into(),
        format!("{}", result.operations_count),
    ]);
    table.add_row(vec![
        "Total Power".into(),
        format!("{:.4} mW", result.estimated_power_w * 1000.0),
    ]);
    table.add_row(vec![
        "  Base Power".into(),
        format!("{:.4} mW", result.power_breakdown.base_power_w * 1000.0),
    ]);
    table.add_row(vec![
        "  Dynamic Power".into(),
        format!("{:.6} mW", result.power_breakdown.dynamic_power_w * 1000.0),
    ]);
    table.add_row(vec![
        "  Leakage Power".into(),
        format!("{:.4} mW", result.power_breakdown.leakage_power_w * 1000.0),
    ]);
    table.add_row(vec![
        "Throughput".into(),
        format!("{:.1} samples/s", result.throughput_samples_per_sec),
    ]);
    table.add_row(vec![
        "Junction Temp".into(),
        format!("{:.1} C", result.thermal_state.junction_temp_c),
    ]);
    table.add_row(vec![
        "Thermal Headroom".into(),
        format!("{:.1} C", result.thermal_state.headroom_c),
    ]);

    println!("{table}");
}
