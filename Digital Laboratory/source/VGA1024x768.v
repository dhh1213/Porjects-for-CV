`timescale 1ns / 1ps
module VGA1024x768(
    input i_clk,           // base clock
    input i_pix_stb,       // pixel clock strobe
    input i_rst,           // reset: restarts frame
    output wire o_hs,           // horizontal sync
    output wire o_vs,           // vertical sync
    output wire [10:0] o_x,      // current pixel x position
    output wire [10:0] o_y,       // current pixel y position
    output wire o_animate         // high for one tick at end of active drawing
    );
    // VGA timings 
    localparam HS_STA = 24;              // horizontal sync start
    localparam HS_END = 24 + 136;         // horizontal sync end
    localparam HA_STA = 24 + 136 + 160;    // horizontal active pixel start
    localparam VS_STA = 768 + 3;        // vertical sync start
    localparam VS_END = 768 + 3 + 6;    // vertical sync end
    localparam VA_END = 768;             // vertical active pixel end
    localparam LINE   = 1344;             // complete line (pixels)
    localparam SCREEN = 806;             // complete screen (lines)

    reg [10:0] h_count;  // line position
    reg [10:0] v_count;  // screen position

    // generate sync signals (active low for 640x480)
    assign o_hs = ~((h_count >= HS_STA) & (h_count < HS_END));
    assign o_vs = ~((v_count >= VS_STA) & (v_count < VS_END));

    // keep x and y bound within the active pixels
    assign o_x = (h_count < HA_STA) ? 0 : (h_count - HA_STA);
    assign o_y = (v_count >= VA_END) ? (VA_END - 1) : (v_count);

    // animate: high for one tick at the end of the final active pixel line
    assign o_animate = ((v_count == VA_END - 1) & (h_count == LINE));

    always @ (posedge i_clk)
    begin
        if (i_rst)  // reset to start of frame
        begin
            h_count <= 0;
            v_count <= 0;
        end
        if(i_pix_stb)  // once per pixel
        begin
            if (h_count == LINE)  // end of line
            begin
                h_count <= 0;
                v_count <= v_count + 1;
            end
            else 
                h_count <= h_count + 1;

            if (v_count == SCREEN)  // end of screen
                v_count <= 0;
        end
    end
endmodule