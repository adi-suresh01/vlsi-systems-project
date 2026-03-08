use std::net::SocketAddr;
use std::path::PathBuf;

use anyhow::Result;
use clap::{Parser, Subcommand};
use comfy_table::{Table, presets::UTF8_FULL};

use vlsi_sim::{
    generate_test_samples, run_simulation, run_simulation_parallel, DvfsConfig, PipelineConfig,
};

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
        } => cmd_sim(samples, clock_mhz, threads, height, width, cli.json),

        Commands::Bench { samples, compare } => cmd_bench(samples, compare, cli.json),

        Commands::Dashboard {
            host,
            port,
            frontend_dir,
            metrics_interval,
        } => cmd_dashboard(host, port, frontend_dir, metrics_interval).await,

        Commands::Sysinfo => cmd_sysinfo(cli.json),

        Commands::Dvfs { samples } => cmd_dvfs(samples, cli.json),
    }
}

fn cmd_sim(
    samples: usize,
    clock_mhz: f64,
    threads: usize,
    height: usize,
    width: usize,
    json: bool,
) -> Result<()> {
    let test_data = generate_test_samples(samples, height, width);
    let mut config = PipelineConfig::default();
    config.dvfs.frequency_mhz = clock_mhz;

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
