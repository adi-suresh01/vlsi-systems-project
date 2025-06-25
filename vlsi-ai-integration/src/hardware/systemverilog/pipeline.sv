module pipeline (
    input logic clk,
    input logic rst_n,
    input logic [7:0] data_in,
    output logic [7:0] data_out
);

    logic [7:0] mac_out;
    logic [7:0] relu_out;

    // Instantiate the MAC unit
    mac_unit mac_inst (
        .clk(clk),
        .rst_n(rst_n),
        .data_in(data_in),
        .data_out(mac_out)
    );

    // Instantiate the ReLU unit
    relu_unit relu_inst (
        .data_in(mac_out),
        .data_out(relu_out)
    );

    // Output the final result
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            data_out <= 8'b0;
        end else begin
            data_out <= relu_out;
        end
    end

endmodule