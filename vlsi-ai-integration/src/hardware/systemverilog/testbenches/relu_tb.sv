module relu_unit (
    input logic [31:0] in_data,
    output logic [31:0] out_data
);
    always_comb begin
        out_data = (in_data > 0) ? in_data : 0;
    end
endmodule

module relu_tb;
    logic [31:0] test_in;
    logic [31:0] test_out;

    // Instantiate the ReLU unit
    relu_unit uut (
        .in_data(test_in),
        .out_data(test_out)
    );

    initial begin
        // Test case 1: Positive input
        test_in = 32'h00000005; // 5
        #10; // Wait for 10 time units
        assert(test_out == 32'h00000005) else $fatal("Test case 1 failed!");

        // Test case 2: Zero input
        test_in = 32'h00000000; // 0
        #10; // Wait for 10 time units
        assert(test_out == 32'h00000000) else $fatal("Test case 2 failed!");

        // Test case 3: Negative input
        test_in = 32'hFFFFFFFF; // -1 in 32-bit two's complement
        #10; // Wait for 10 time units
        assert(test_out == 32'h00000000) else $fatal("Test case 3 failed!");

        // Test case 4: Large positive input
        test_in = 32'h7FFFFFFF; // Max positive value
        #10; // Wait for 10 time units
        assert(test_out == 32'h7FFFFFFF) else $fatal("Test case 4 failed!");

        // End simulation
        $finish;
    end
endmodule