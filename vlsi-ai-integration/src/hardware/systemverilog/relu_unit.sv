module relu_unit (
    input logic [31:0] data_in,
    output logic [31:0] data_out
);
    always_comb begin
        if (data_in < 0) 
            data_out = 0;
        else 
            data_out = data_in;
    end
endmodule