module mac_tb;

  // Parameters
  parameter DATA_WIDTH = 8;
  parameter ACC_WIDTH = 16;

  // Inputs
  reg [DATA_WIDTH-1:0] a;
  reg [DATA_WIDTH-1:0] b;
  reg clk;
  reg start;
  
  // Outputs
  wire [ACC_WIDTH-1:0] result;
  wire done;

  // Instantiate the MAC unit
  mac_unit #(
    .DATA_WIDTH(DATA_WIDTH),
    .ACC_WIDTH(ACC_WIDTH)
  ) uut (
    .a(a),
    .b(b),
    .clk(clk),
    .start(start),
    .result(result),
    .done(done)
  );

  // Clock generation
  initial begin
    clk = 0;
    forever #5 clk = ~clk; // 10 time units clock period
  end

  // Test sequence
  initial begin
    // Initialize inputs
    a = 0;
    b = 0;
    start = 0;

    // Wait for a few clock cycles
    #10;

    // Test case 1: Multiply 3 and 4
    a = 8'd3;
    b = 8'd4;
    start = 1;
    #10; // Wait for one clock cycle
    start = 0;

    // Wait for the operation to complete
    wait(done);
    #10; // Wait for a few clock cycles

    // Check result
    if (result !== 12) begin
      $display("Test case 1 failed: expected 12, got %d", result);
    end else begin
      $display("Test case 1 passed: %d * %d = %d", a, b, result);
    end

    // Test case 2: Multiply 5 and 6
    a = 8'd5;
    b = 8'd6;
    start = 1;
    #10; // Wait for one clock cycle
    start = 0;

    // Wait for the operation to complete
    wait(done);
    #10; // Wait for a few clock cycles

    // Check result
    if (result !== 30) begin
      $display("Test case 2 failed: expected 30, got %d", result);
    end else begin
      $display("Test case 2 passed: %d * %d = %d", a, b, result);
    end

    // Finish simulation
    $finish;
  end

endmodule