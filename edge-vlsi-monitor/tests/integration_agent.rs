/// Integration tests for the agent runtime.

use agent_runtime::{Agent, AgentState};
use ndarray::Array2;

#[test]
fn test_agent_simulation_mode_lifecycle() {
    let mut agent = Agent::new("test-agent".into());
    assert_eq!(agent.state(), AgentState::Idle);

    // Start in simulation mode (no ONNX model needed)
    agent.start_simulation_mode().unwrap();
    assert_eq!(agent.state(), AgentState::Running);

    // Run hardware simulation
    let samples: Vec<Array2<f64>> = (0..3)
        .map(|_| Array2::from_elem((28, 28), 0.5))
        .collect();

    let result = agent.run_hw_simulation(&samples);
    assert_eq!(result.results.len(), 3);
    assert!(result.mac_operations > 0);

    let info = agent.info();
    assert_eq!(info.inference_count, 1);
    assert!(info.total_mac_ops > 0);

    // Pause and resume
    agent.pause().unwrap();
    assert_eq!(agent.state(), AgentState::Paused);

    agent.resume().unwrap();
    assert_eq!(agent.state(), AgentState::Running);

    // Run another simulation
    let result2 = agent.run_hw_simulation(&samples);
    assert_eq!(result2.results.len(), 3);

    let info2 = agent.info();
    assert_eq!(info2.inference_count, 2);

    // Terminate
    agent.terminate().unwrap();
    assert_eq!(agent.state(), AgentState::Terminated);
}

#[test]
fn test_agent_invalid_transitions() {
    let mut agent = Agent::new("bad-agent".into());

    // Can't go directly to Running from Idle
    assert!(agent.pause().is_err());
    assert!(agent.resume().is_err());

    // Start properly
    agent.start_simulation_mode().unwrap();

    // Can't go back to Loading
    // (resume from Running is invalid — already running)
    assert!(agent.resume().is_err());
}

#[test]
fn test_agent_metrics_accumulation() {
    let mut agent = Agent::new("metrics-agent".into());
    agent.start_simulation_mode().unwrap();

    let samples: Vec<Array2<f64>> = vec![Array2::ones((28, 28))];

    // Run 5 simulations
    for _ in 0..5 {
        agent.run_hw_simulation(&samples);
    }

    let info = agent.info();
    assert_eq!(info.inference_count, 5);
    // Each simulation: 1 sample * 8 kernels * 49 patches * 9 MACs = 3528
    assert_eq!(info.total_mac_ops, 5 * 3528);
}
