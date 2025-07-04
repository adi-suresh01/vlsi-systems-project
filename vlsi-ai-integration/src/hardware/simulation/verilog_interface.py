"""
Verilog Hardware Interface
Provides compilation and simulation interface for Verilog hardware modules
"""

from typing import List
import subprocess
import os

class VerilogInterface:
    def __init__(self, verilog_file: str):
        self.verilog_file = verilog_file
        self.compiled = False

    def compile(self) -> bool:
        """Compile Verilog file using available simulator"""
        try:
            # Try different simulators
            simulators = ['iverilog', 'vlog', 'xvlog']
            
            for sim in simulators:
                try:
                    command = f"{sim} {self.verilog_file}"
                    result = subprocess.run(command, shell=True, check=True, 
                                          capture_output=True, text=True)
                    self.compiled = True
                    print(f"Compiled with {sim}")
                    return True
                except subprocess.CalledProcessError:
                    continue
                except FileNotFoundError:
                    continue
            
            print("No Verilog simulator found, using software simulation")
            return False
            
        except Exception as e:
            print(f"Compilation failed: {e}")
            return False

    def simulate(self) -> bool:
        """Execute simulation of the compiled Verilog design"""
        if not self.compiled:
            print("File not compiled, attempting to compile first...")
            if not self.compile():
                return False
        
        try:
            # Try different simulation commands
            base_name = os.path.splitext(os.path.basename(self.verilog_file))[0]
            commands = [
                f"vvp {base_name}",
                f"vsim -c {base_name} -do 'run -all; quit'",
                f"./{base_name}"
            ]
            
            for cmd in commands:
                try:
                    result = subprocess.run(cmd, shell=True, check=True,
                                          capture_output=True, text=True)
                    print("Simulation completed successfully")
                    return True
                except subprocess.CalledProcessError:
                    continue
                except FileNotFoundError:
                    continue
            
            print("Simulation failed, using software fallback")
            return False
            
        except Exception as e:
            print(f"Simulation error: {e}")
            return False

    def get_results(self) -> List[str]:
        """Retrieve simulation output from result files"""
        try:
            # Look for common result files
            result_files = ['output.txt', 'results.txt', 'simulation.log']
            
            for file in result_files:
                if os.path.exists(file):
                    with open(file, 'r') as f:
                        return f.readlines()
            
            return ["No results found"]
            
        except Exception as e:
            print(f"Error reading results: {e}")
            return ["Error reading results"]

    def clean(self) -> None:
        """Clean up generated simulation files"""
        try:
            cleanup_files = ['*.vcd', '*.wlf', 'transcript', 'work', '*.out']
            for pattern in cleanup_files:
                try:
                    subprocess.run(f"rm -rf {pattern}", shell=True)
                except:
                    pass
            print("Cleanup completed")
        except Exception as e:
            print(f"Cleanup error: {e}")