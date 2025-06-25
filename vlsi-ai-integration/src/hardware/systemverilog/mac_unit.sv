module mac_unit (
    input logic clk,
    input logic rst_n,
    input logic [15:0] a, // First operand
    input logic [15:0] b, // Second operand
    output logic [31:0] result // Result of multiplication
);

    logic [31:0] mult_result;

    // Multiply operation
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            mult_result <= 32'b0;
        end else begin
            mult_result <= a * b;
        end
    end

    // Accumulate result
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            result <= 32'b0;
        end else begin
            result <= result + mult_result;
        end
    end

endmodule