//############################################################################
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//   (C) Copyright Si2 LAB @NYCU ED430
//   All Right Reserved
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//
//   ICLAB 2022 SPRING
//   Final Proejct              : TOF  
//   Author                     : Wen-Yue, Lin
//
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//
//   File Name   : TOF.v
//   Module Name : TOF
//   Release version : V1.0 (Release Date: 2022-5)
//
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//############################################################################

module TOF(
    // CHIP IO
    clk,
    rst_n,
    in_valid,
    start,
    stop,
    inputtype,
    frame_id,
    busy,

    // AXI4 IO
    arid_m_inf,
    araddr_m_inf,
    arlen_m_inf,
    arsize_m_inf,
    arburst_m_inf,
    arvalid_m_inf,
    arready_m_inf,
    
    rid_m_inf,
    rdata_m_inf,
    rresp_m_inf,
    rlast_m_inf,
    rvalid_m_inf,
    rready_m_inf,

    awid_m_inf,
    awaddr_m_inf,
    awsize_m_inf,
    awburst_m_inf,
    awlen_m_inf,
    awvalid_m_inf,
    awready_m_inf,

    wdata_m_inf,
    wlast_m_inf,
    wvalid_m_inf,
    wready_m_inf,
    
    bid_m_inf,
    bresp_m_inf,
    bvalid_m_inf,
    bready_m_inf 
);
// ===============================================================
//                      Parameter Declaration 
// ===============================================================
parameter ID_WIDTH=4, DATA_WIDTH=128, ADDR_WIDTH=32;    // DO NOT modify AXI4 Parameter


// ===============================================================
//                      Input / Output 
// ===============================================================

// << CHIP io port with system >>
input           clk, rst_n;
input           in_valid;
input           start;
input [15:0]    stop;     
input [1:0]     inputtype; 
input [4:0]     frame_id;
output reg      busy;       

// AXI Interface wire connecttion for pseudo DRAM read/write
/* Hint:
    Your AXI-4 interface could be designed as a bridge in submodule,
    therefore I declared output of AXI as wire.  
    Ex: AXI4_interface AXI4_INF(...);
*/

// ------------------------
// <<<<< AXI READ >>>>>
// ------------------------
// (1)    axi read address channel 
output wire [ID_WIDTH-1:0]      arid_m_inf;
output wire [1:0]            arburst_m_inf;
output wire [2:0]             arsize_m_inf;
output wire [7:0]              arlen_m_inf;
output reg                   arvalid_m_inf;
input  wire                  arready_m_inf;
output reg  [ADDR_WIDTH-1:0]  araddr_m_inf;
// ------------------------
// (2)    axi read data channel 
input  wire [ID_WIDTH-1:0]       rid_m_inf;
input  wire                   rvalid_m_inf;
output reg                    rready_m_inf;
input  wire [DATA_WIDTH-1:0]   rdata_m_inf;
input  wire                    rlast_m_inf;
input  wire [1:0]              rresp_m_inf;
// ------------------------
// <<<<< AXI WRITE >>>>>
// ------------------------
// (1)     axi write address channel 
output wire [ID_WIDTH-1:0]      awid_m_inf;
output wire [1:0]            awburst_m_inf;
output wire [2:0]             awsize_m_inf;
output reg  [7:0]              awlen_m_inf;
output reg                   awvalid_m_inf;
input  wire                  awready_m_inf;
output reg  [ADDR_WIDTH-1:0]  awaddr_m_inf;
// -------------------------
// (2)    axi write data channel 
output reg                    wvalid_m_inf;
input  wire                   wready_m_inf;
output reg  [DATA_WIDTH-1:0]   wdata_m_inf;
output reg                     wlast_m_inf;
// -------------------------
// (3)    axi write response channel 
input  wire  [ID_WIDTH-1:0]      bid_m_inf;
input  wire                   bvalid_m_inf;
output reg                    bready_m_inf;
input  wire  [1:0]             bresp_m_inf;
// -----------------------------

// AXI4 READ
assign arid_m_inf = 0;
assign arsize_m_inf = 3'b100;
assign arburst_m_inf = 2'b01;
assign arlen_m_inf = 8'd255;

// AXI4 WRITE
assign awid_m_inf = 0;
assign awsize_m_inf = 3'b100;
assign awburst_m_inf = 2'b01;


//--------- Integer and Parameter ------------
integer i, j;
parameter WINDOW_SIZE = 4'd5;

// states
parameter S_IDLE = 4'd 0;
parameter S_READ = 4'd 1;
parameter S_CAL0 = 4'd 2;
parameter S_IN_H = 4'd 3;
parameter S_R_ad = 4'd 4;
parameter S_R_da = 4'd 5;
parameter S_Feth = 4'd 6;
parameter S_Wind = 4'd 7;
parameter S_Dist = 4'd 8;
parameter S_W1_a = 4'd 9;
parameter S_W1_d = 4'd10;
parameter S_W1_r = 4'd11;
parameter S_Busy = 4'd12;
parameter S_IN_S = 4'd13;
parameter S_OUT  = 4'd15;

//-------------- Wire and Reg ----------------

// FSM
reg [3:0]cur_state, next_state;

// READ
reg [1:0]mode_reg;
reg [4:0]frame_reg;

// DRAM
reg [ADDR_WIDTH-1:0]read_addr;
reg [ADDR_WIDTH-1:0]write_addr;
reg [3:0]write_histo_cnt, write_histo_cnt_reg, write_histo_idx, write_histo_idx_reg;
reg [DATA_WIDTH-1:0]write_data;
reg [8:0]write_cnt, write_cnt_reg;

// SRAM16
wire oen, cen;
reg  SRAM_wen[0:15];
wire [DATA_WIDTH-1:0]SRAM_Data_o[0:15];
reg  [DATA_WIDTH-1:0]SRAM_Data_i[0:15];
reg  [3:0]SRAM16_add[0:15];

// HISTOGRAM for Window
reg [7:0]histo[0:254], histo_reg[0:254];
reg [4:0]histo_cnt, histo_cnt_reg, histo_cnt_reg_d1, fetch_histo_cnt, fetch_histo_cnt_reg; // 0~15

// WINDOW
reg [8:0]wind_cnt, wind_cnt_reg;
reg [7:0]win_first, win_first_reg, win_first_idx, win_first_idx_reg, win_max_idx, win_max_idx_reg;
reg [10:0]win_max, win_max_reg, win_cur, win_cur_reg;
reg [3:0]win_size;

// Distance
reg [4:0]dist_cnt, dist_cnt_reg;
reg [7:0]dist_cal, dist_cal_d1;

// Build Hist
reg first, first_reg;
reg [4:0]histo_add_cnt, histo_add_cnt_reg, histo_data_cnt, histo_data_cnt_reg;
reg [8:0]build_cnt, build_cnt_reg;
reg [7:0]data_read[0:15][0:15], data_read_reg[0:15][0:15], data_write[0:15][0:15], data_write_reg[0:15][0:15];
reg [15:0]stop_reg;

assign cen = 0;
assign oen = 0;

SRAM_16_128 SRAM16_0(.Q(SRAM_Data_o[0]), .CLK(clk), .CEN(cen), .WEN(SRAM_wen[0]), .A(SRAM16_add[0]), .D(SRAM_Data_i[0]), .OEN(oen));
SRAM_16_128 SRAM16_1(.Q(SRAM_Data_o[1]), .CLK(clk), .CEN(cen), .WEN(SRAM_wen[1]), .A(SRAM16_add[1]), .D(SRAM_Data_i[1]), .OEN(oen));
SRAM_16_128 SRAM16_2(.Q(SRAM_Data_o[2]), .CLK(clk), .CEN(cen), .WEN(SRAM_wen[2]), .A(SRAM16_add[2]), .D(SRAM_Data_i[2]), .OEN(oen));
SRAM_16_128 SRAM16_3(.Q(SRAM_Data_o[3]), .CLK(clk), .CEN(cen), .WEN(SRAM_wen[3]), .A(SRAM16_add[3]), .D(SRAM_Data_i[3]), .OEN(oen));
SRAM_16_128 SRAM16_4(.Q(SRAM_Data_o[4]), .CLK(clk), .CEN(cen), .WEN(SRAM_wen[4]), .A(SRAM16_add[4]), .D(SRAM_Data_i[4]), .OEN(oen));
SRAM_16_128 SRAM16_5(.Q(SRAM_Data_o[5]), .CLK(clk), .CEN(cen), .WEN(SRAM_wen[5]), .A(SRAM16_add[5]), .D(SRAM_Data_i[5]), .OEN(oen));
SRAM_16_128 SRAM16_6(.Q(SRAM_Data_o[6]), .CLK(clk), .CEN(cen), .WEN(SRAM_wen[6]), .A(SRAM16_add[6]), .D(SRAM_Data_i[6]), .OEN(oen));
SRAM_16_128 SRAM16_7(.Q(SRAM_Data_o[7]), .CLK(clk), .CEN(cen), .WEN(SRAM_wen[7]), .A(SRAM16_add[7]), .D(SRAM_Data_i[7]), .OEN(oen));
SRAM_16_128 SRAM16_8(.Q(SRAM_Data_o[8]), .CLK(clk), .CEN(cen), .WEN(SRAM_wen[8]), .A(SRAM16_add[8]), .D(SRAM_Data_i[8]), .OEN(oen));
SRAM_16_128 SRAM16_9(.Q(SRAM_Data_o[9]), .CLK(clk), .CEN(cen), .WEN(SRAM_wen[9]), .A(SRAM16_add[9]), .D(SRAM_Data_i[9]), .OEN(oen));
SRAM_16_128 SRAM16_10(.Q(SRAM_Data_o[10]), .CLK(clk), .CEN(cen), .WEN(SRAM_wen[10]), .A(SRAM16_add[10]), .D(SRAM_Data_i[10]), .OEN(oen));
SRAM_16_128 SRAM16_11(.Q(SRAM_Data_o[11]), .CLK(clk), .CEN(cen), .WEN(SRAM_wen[11]), .A(SRAM16_add[11]), .D(SRAM_Data_i[11]), .OEN(oen));
SRAM_16_128 SRAM16_12(.Q(SRAM_Data_o[12]), .CLK(clk), .CEN(cen), .WEN(SRAM_wen[12]), .A(SRAM16_add[12]), .D(SRAM_Data_i[12]), .OEN(oen));
SRAM_16_128 SRAM16_13(.Q(SRAM_Data_o[13]), .CLK(clk), .CEN(cen), .WEN(SRAM_wen[13]), .A(SRAM16_add[13]), .D(SRAM_Data_i[13]), .OEN(oen));
SRAM_16_128 SRAM16_14(.Q(SRAM_Data_o[14]), .CLK(clk), .CEN(cen), .WEN(SRAM_wen[14]), .A(SRAM16_add[14]), .D(SRAM_Data_i[14]), .OEN(oen));
SRAM_16_128 SRAM16_15(.Q(SRAM_Data_o[15]), .CLK(clk), .CEN(cen), .WEN(SRAM_wen[15]), .A(SRAM16_add[15]), .D(SRAM_Data_i[15]), .OEN(oen));

//----------------- Design -------------------
// FSM
always@(posedge clk or negedge rst_n)begin
	if(!rst_n)
		cur_state <= S_IDLE;
	else
		cur_state <= next_state;
end
always @(*) begin
	case (cur_state)
	S_IDLE: begin
				if(in_valid)
					next_state = S_READ;
				else
					next_state = S_IDLE;
			end
	S_READ: begin
				if(mode_reg == 0)
					next_state = S_R_ad;
				else
					next_state = S_CAL0;
			end
	S_CAL0:	begin
				if(start)
					next_state = S_IN_H;
				else
					next_state = S_CAL0;
			end
	S_R_ad:	begin
				if(arready_m_inf == 1 && arvalid_m_inf == 1)
					next_state = S_R_da;
				else
					next_state = S_R_ad;
			end
	S_R_da:	begin
				if(rlast_m_inf)
					next_state = S_Feth;
				else
					next_state = S_R_da;
			end
	S_Feth: begin
				if(histo_cnt_reg_d1 == 15)
					next_state = S_Wind;
				else
					next_state = S_Feth;
			end
	S_IN_H: begin
				if(build_cnt_reg == 254)
					next_state = S_IN_S;
				else
					next_state = S_IN_H;
			end
	S_IN_S: begin
				if(!in_valid)
					next_state = S_Feth;
				else if(start == 1)
					next_state = S_IN_H;
				else
					next_state = S_IN_S;
			end
	S_Wind: begin
				if(wind_cnt_reg == 256)
					next_state = S_Dist;
				else
					next_state = S_Wind;
			end
	S_Dist: begin
				if(dist_cnt_reg == 4 && mode_reg == 1)
					next_state = S_W1_a;
				else if(dist_cnt_reg == 16 && mode_reg != 1)
					next_state = S_W1_a;
				else
					next_state = S_Feth;
			end
	S_W1_a: begin
				if(awready_m_inf == 1 && awvalid_m_inf == 1)
					next_state = S_W1_d;
				else
					next_state = S_W1_a;
			end
	S_W1_d: begin
				if(wready_m_inf && mode_reg == 0)
					next_state = S_W1_r;
				else if(wready_m_inf && write_cnt_reg == 257 && mode_reg != 0)
					next_state = S_W1_r;
				else
					next_state = S_W1_d;
			end
	S_W1_r: begin
				if(bready_m_inf == 1 && bvalid_m_inf == 1)begin
					if(write_histo_cnt_reg == 0 && mode_reg == 0)
						next_state = S_Busy;
					else if(mode_reg != 0)
						next_state = S_Busy;
					else
						next_state = S_W1_a;
				end
				else
					next_state = S_W1_r;
			end
	S_Busy: begin
				if(in_valid)
					next_state = S_READ;
				else
					next_state = S_Busy;
			end
	S_OUT:	begin
				next_state = S_OUT;
			end
	default:
			next_state = cur_state;
	endcase
end


// READ
always@(posedge clk or negedge rst_n)begin
	if(!rst_n)begin
		mode_reg <= 0;
		frame_reg <= 0;
	end
	else if(next_state == S_IDLE || next_state == S_Busy)begin
		mode_reg <= 0;
		frame_reg <= 0;
	end
	else if(next_state == S_READ)begin
		mode_reg <= inputtype;
		frame_reg <= frame_id;
	end
end

// BUSY
always@(*)begin
	busy = 0;
	case(cur_state)
		S_IDLE, S_READ, S_CAL0, S_IN_H, S_IN_S, S_R_ad, S_Busy: busy = 0;
		S_R_ad, S_R_da, S_Wind, S_Dist, S_Feth, S_W1_a, S_W1_d, S_W1_r: busy = 1;
	endcase
end

// ---------------------------   DRAM   ---------------------------
// read addr
always@(posedge clk or negedge rst_n)begin
	if(!rst_n)begin
		araddr_m_inf <= 0;
		arvalid_m_inf <= 0;
	end
	else if(next_state == S_IDLE || next_state == S_Busy)begin
		araddr_m_inf <= 0;
		arvalid_m_inf <= 0;
	end
	else if(next_state == S_R_ad)begin
		araddr_m_inf <= read_addr;
		arvalid_m_inf <= 1;
	end
	else if(next_state == S_R_da)begin
		araddr_m_inf <= araddr_m_inf;
		arvalid_m_inf <= 0;
	end
end
always@(*)begin
	read_addr = araddr_m_inf;
	if(next_state == S_R_ad)
		read_addr = 32'h00010000 + {frame_reg, 12'b0};
end

// read data
always@(posedge clk or negedge rst_n)begin
	if(!rst_n)begin
		rready_m_inf <= 0;
	end
	else if(next_state == S_IDLE || next_state == S_Busy)begin
		rready_m_inf <= 0;
	end
	else if(next_state == S_R_da)begin
		rready_m_inf <= 1;
	end
	else begin
		rready_m_inf <= 0;
	end
end

// write addr
always@(posedge clk or negedge rst_n)begin
	if(!rst_n)begin
		awaddr_m_inf <= 0;
		awvalid_m_inf <= 0;
		write_cnt_reg <= 0;
		write_histo_cnt_reg <= 0;
		write_histo_idx_reg <= 0;
	end
	else if(next_state == S_IDLE || next_state == S_Busy) begin
		awaddr_m_inf <= 0;
		awvalid_m_inf <= 0;
		write_cnt_reg <= 0;
		write_histo_cnt_reg <= 0;
		write_histo_idx_reg <= 0;
	end
	else if(next_state == S_W1_a) begin
		awaddr_m_inf <= write_addr;
		awvalid_m_inf <= 1;
		write_cnt_reg <= 0;
		write_histo_cnt_reg <= write_histo_cnt;
		write_histo_idx_reg <= 0;
	end
	else if(next_state == S_W1_d)begin
		awaddr_m_inf <= awaddr_m_inf;
		awvalid_m_inf <= 0;
		write_cnt_reg <= write_cnt;
		write_histo_cnt_reg <= write_histo_cnt;
		write_histo_idx_reg <= write_histo_idx;
	end
	else if(next_state == S_W1_r)begin
		awaddr_m_inf <= awaddr_m_inf;
		awvalid_m_inf <= 0;
		write_cnt_reg <= write_cnt;
		write_histo_cnt_reg <= write_histo_cnt;
		write_histo_idx_reg <= write_histo_idx;
	end
end
always@(*)begin
	awlen_m_inf = 0;
	write_addr = awaddr_m_inf;
	write_cnt = write_cnt_reg;
	write_histo_cnt = write_histo_cnt_reg;
	write_histo_idx = write_histo_idx_reg;
	if(mode_reg != 0)begin
		awlen_m_inf = 255;
		write_addr = 32'h00010000 + {frame_reg, 12'b0};
		if(cur_state == S_W1_d)begin
			write_cnt = write_cnt_reg + 1;
			if(write_histo_cnt_reg < 15)
				write_histo_cnt = write_histo_cnt_reg + 1;
			else begin
				write_histo_cnt = 0;
			end
			if(write_histo_cnt_reg == 1 && write_cnt_reg > 1)
				write_histo_idx = write_histo_idx_reg + 1;
		end
	end
	else begin
		awlen_m_inf = 0;
		write_addr = 32'h000100F0 + {frame_reg, write_histo_cnt_reg, 8'b0};
		if(next_state == S_W1_r && cur_state == S_W1_d)
			write_histo_cnt = write_histo_cnt_reg + 1;
	end
end

// write data
always@(posedge clk or negedge rst_n)begin
	if(!rst_n)begin
		wvalid_m_inf <= 0;
		wdata_m_inf <= 0;
	end
	else if(next_state == S_IDLE || next_state == S_Busy)begin
		wvalid_m_inf <= 0;
		wdata_m_inf <= 0;
	end
	else if(next_state == S_W1_d)begin
		wvalid_m_inf <= 1;
		wdata_m_inf <= write_data;
	end
	else begin
		wvalid_m_inf <= 0;
		wdata_m_inf <= 0;
	end
end
always@(*)begin
	if(mode_reg != 0)begin
		if(write_cnt_reg == 257)
			wlast_m_inf = 1;
		else
			wlast_m_inf = 0;
		if(write_histo_cnt_reg == 1 && write_cnt_reg > 16)
			write_data	=  SRAM_Data_o[write_histo_idx_reg+1];
		else
			write_data	=  SRAM_Data_o[write_histo_idx_reg];
	end
	else begin
		if(cur_state == S_W1_d)
			wlast_m_inf = 1;
		else
			wlast_m_inf = 0;
		write_data = SRAM_Data_o[write_histo_cnt_reg];
	end
end

// write response
always@(posedge clk or negedge rst_n)begin
	if(!rst_n)begin
		bready_m_inf <= 0;
	end
	else if(next_state == S_IDLE || next_state == S_Busy)begin
		bready_m_inf <= 0;
	end
	else if(next_state == S_W1_r || next_state == S_W1_d)begin
		bready_m_inf <= 1;
	end
	else begin
		bready_m_inf <= 0;
	end
end

// ---------------------------   SRAM16   ---------------------------
// wen 16
always@(*)begin
	for(i = 0; i < 16; i = i + 1)
		SRAM_wen[i] = 1;
	case(cur_state)
		S_IN_H:begin
			if(histo_data_cnt_reg == 0 && histo_add_cnt_reg > 0)begin
				for(i = 0; i < 16; i = i + 1)
					SRAM_wen[i] = 0;
			end
		end
		S_IN_S:begin
			if(build_cnt_reg == 255)begin
				for(i = 0; i < 16; i = i + 1)
					SRAM_wen[i] = 0;
			end
		end
		S_Dist:begin
			if(mode_reg != 1)
				SRAM_wen[dist_cnt_reg-1] = 0;
			else begin
				if(dist_cnt_reg == 1)begin
					SRAM_wen[0] = 0;
					SRAM_wen[1] = 0;
					SRAM_wen[4] = 0;
					SRAM_wen[5] = 0;
				end
				else if(dist_cnt_reg == 2)begin
					SRAM_wen[2] = 0;
					SRAM_wen[3] = 0;
					SRAM_wen[6] = 0;
					SRAM_wen[7] = 0;
				end
				else if(dist_cnt_reg == 3)begin
					SRAM_wen[8] = 0;
					SRAM_wen[9] = 0;
					SRAM_wen[12] = 0;
					SRAM_wen[13] = 0;
				end
				else if(dist_cnt_reg == 4)begin
					SRAM_wen[10] = 0;
					SRAM_wen[11] = 0;
					SRAM_wen[14] = 0;
					SRAM_wen[15] = 0;
				end
			end
		end
		S_R_da:begin
			if(rvalid_m_inf)
				SRAM_wen[histo_add_cnt_reg] = 0;
		end
	endcase
end

// add
always@(posedge clk or negedge rst_n)begin
	if(!rst_n)begin
		histo_add_cnt_reg <= 0;
		histo_data_cnt_reg <= 0;
		build_cnt_reg <= 0;
	end
	else begin
		histo_add_cnt_reg <= histo_add_cnt;
		histo_data_cnt_reg <= histo_data_cnt;
		build_cnt_reg <= build_cnt;
	end
end

always@(*)begin
	build_cnt = build_cnt_reg;
	case(cur_state)
	S_IN_H: build_cnt = build_cnt_reg + 1;
	S_IN_S: build_cnt = 0;
	endcase
end

always@(*)begin
	histo_add_cnt = histo_add_cnt_reg;
	histo_data_cnt = histo_data_cnt_reg;
	
	for(i = 0; i < 16; i = i + 1)
		SRAM16_add[i] = histo_add_cnt_reg;
	
	case(cur_state)
		S_Busy:begin
			histo_add_cnt = 0;
			histo_data_cnt = 0;
		end
		S_IN_H:begin
			if((histo_data_cnt_reg < 15 && histo_add_cnt_reg < 15) || histo_data_cnt_reg < 14)begin
				histo_data_cnt = histo_data_cnt_reg + 1;
			end
			else begin
				histo_data_cnt = 0;
				histo_add_cnt = histo_add_cnt_reg + 1;
			end
			
			if(histo_data_cnt_reg == 0 && histo_add_cnt_reg > 0)begin
				for(i = 0; i < 16; i = i + 1)
					SRAM16_add[i] = histo_add_cnt_reg - 1;
			end
			else if(histo_data_cnt_reg == 15)begin
				for(i = 0; i < 16; i = i + 1)
					SRAM16_add[i] = histo_add_cnt_reg + 1;
			end
		end
		S_IN_S:begin
			histo_add_cnt = 0;
			if(build_cnt_reg == 255)begin
				for(i = 0; i < 16; i = i + 1)
					SRAM16_add[i] = 15;
			end
		end
		S_Feth:begin
			histo_add_cnt = histo_add_cnt_reg + 1;
			for(i = 0; i < 16; i = i + 1)
				SRAM16_add[i] = histo_add_cnt_reg;
		end
		S_Wind:begin
			for(i = 0; i < 16; i = i + 1)
				SRAM16_add[i] = 15;
		end
		S_Dist:begin
			histo_add_cnt = 0;
			for(i = 0; i < 16; i = i + 1)
				SRAM16_add[i] = 15;
		end
		S_W1_d:begin
			if(mode_reg != 0)begin
				if(write_histo_cnt_reg < 2 && write_cnt_reg >15)
					SRAM16_add[write_histo_idx_reg+1] = write_histo_cnt_reg;
				else
					SRAM16_add[write_histo_idx_reg] = write_histo_cnt_reg;
			end
			else begin
				for(i = 0; i < 16; i = i + 1)
					SRAM16_add[i] = 15;
			end
		end
		S_R_da:begin
			if(rvalid_m_inf)begin
				if(histo_data_cnt_reg < 15)begin
					histo_data_cnt = histo_data_cnt_reg + 1;
				end
				else begin
					histo_data_cnt = 0;
					histo_add_cnt = histo_add_cnt_reg + 1;
				end
			end
			SRAM16_add[histo_add_cnt_reg] = histo_data_cnt_reg;
		end
	endcase
end

// data
// input
always@(*)begin
	for(i = 0; i < 16; i = i + 1)
		SRAM_Data_i[i] = 0;
	case(cur_state)
		S_IN_H, S_IN_S:begin
			for(i = 0; i < 16; i = i + 1)begin
				if(build_cnt_reg != 256)
					SRAM_Data_i[i] = {data_write_reg[i][15], data_write_reg[i][14], data_write_reg[i][13], data_write_reg[i][12], 
									  data_write_reg[i][11], data_write_reg[i][10], data_write_reg[i][ 9], data_write_reg[i][ 8], 
									  data_write_reg[i][ 7], data_write_reg[i][ 6], data_write_reg[i][ 5], data_write_reg[i][ 4], 
									  data_write_reg[i][ 3], data_write_reg[i][ 2], data_write_reg[i][ 1], data_write_reg[i][ 0]};
				else
					SRAM_Data_i[i] = {				   8'b0, data_write_reg[i][14], data_write_reg[i][13], data_write_reg[i][12], 
									  data_write_reg[i][11], data_write_reg[i][10], data_write_reg[i][ 9], data_write_reg[i][ 8], 
									  data_write_reg[i][ 7], data_write_reg[i][ 6], data_write_reg[i][ 5], data_write_reg[i][ 4], 
									  data_write_reg[i][ 3], data_write_reg[i][ 2], data_write_reg[i][ 1], data_write_reg[i][ 0]};
			end
		end
		S_Dist:begin
			if(mode_reg != 1)begin
				SRAM_Data_i[dist_cnt_reg-1] = {dist_cal_d1, histo_reg[254], histo_reg[253], histo_reg[252], histo_reg[251], histo_reg[250], histo_reg[249], histo_reg[248], histo_reg[247], histo_reg[246], histo_reg[245], histo_reg[244], histo_reg[243], histo_reg[242], histo_reg[241], histo_reg[240]};
			end
			else begin
				if(dist_cnt_reg == 1)begin
					SRAM_Data_i[0] = {dist_cal_d1, SRAM_Data_o[0][119:0]};
					SRAM_Data_i[1] = {dist_cal_d1, SRAM_Data_o[1][119:0]};
					SRAM_Data_i[4] = {dist_cal_d1, SRAM_Data_o[4][119:0]};
					SRAM_Data_i[5] = {dist_cal_d1, SRAM_Data_o[5][119:0]};
				end
				else if(dist_cnt_reg == 2)begin
					SRAM_Data_i[2] = {dist_cal_d1, SRAM_Data_o[2][119:0]};
					SRAM_Data_i[3] = {dist_cal_d1, SRAM_Data_o[3][119:0]};
					SRAM_Data_i[6] = {dist_cal_d1, SRAM_Data_o[6][119:0]};
					SRAM_Data_i[7] = {dist_cal_d1, SRAM_Data_o[7][119:0]};
				end
				else if(dist_cnt_reg == 3)begin
					SRAM_Data_i[ 8] = {dist_cal_d1, SRAM_Data_o[ 8][119:0]};
					SRAM_Data_i[ 9] = {dist_cal_d1, SRAM_Data_o[ 9][119:0]};
					SRAM_Data_i[12] = {dist_cal_d1, SRAM_Data_o[12][119:0]};
					SRAM_Data_i[13] = {dist_cal_d1, SRAM_Data_o[13][119:0]};
				end
				else if(dist_cnt_reg == 4)begin
					SRAM_Data_i[10] = {dist_cal_d1, SRAM_Data_o[10][119:0]};
					SRAM_Data_i[11] = {dist_cal_d1, SRAM_Data_o[11][119:0]};
					SRAM_Data_i[14] = {dist_cal_d1, SRAM_Data_o[14][119:0]};
					SRAM_Data_i[15] = {dist_cal_d1, SRAM_Data_o[15][119:0]};
				end
			end
		end
		S_R_da:begin
			SRAM_Data_i[histo_add_cnt_reg] = rdata_m_inf;
		end
	endcase
end

// stop
always@(posedge clk or negedge rst_n)begin
	if(!rst_n)begin
		stop_reg <= 0;
	end
	else begin
		stop_reg <= stop;
	end
end

// build
always@(posedge clk or negedge rst_n)begin
	if(!rst_n)begin
		for(i = 0; i < 16; i = i + 1)begin
			for(j = 0; j < 16; j = j + 1)begin
				data_read_reg[i][j] <= 0;
				data_write_reg[i][j] <= 0;
			end
		end
	end
	else begin
		for(i = 0; i < 16; i = i + 1)begin
			for(j = 0; j < 16; j = j + 1)begin
				data_read_reg[i][j] <= data_read[i][j];
				data_write_reg[i][j] <= data_write[i][j];
			end
		end
	end
end

always@(*)begin
	for(i = 0; i < 16; i = i + 1)begin
		for(j = 0; j < 16; j = j + 1)begin
			data_read[i][j] = data_read_reg[i][j];
			data_write[i][j] = data_write_reg[i][j];
		end
	end
	case(cur_state)
		S_CAL0, S_IN_S:begin
				if(histo_data_cnt_reg == 0)begin
					for(i = 0; i < 16; i = i + 1)begin
						if(first_reg)
							for(j = 0; j < 16; j = j + 1)
								data_read[i][j] = 0;
						else begin
							data_read[i][ 0] = SRAM_Data_o[i][  7:  0];
							data_read[i][ 1] = SRAM_Data_o[i][ 15:  8];
							data_read[i][ 2] = SRAM_Data_o[i][ 23: 16];
							data_read[i][ 3] = SRAM_Data_o[i][ 31: 24];
							data_read[i][ 4] = SRAM_Data_o[i][ 39: 32];
							data_read[i][ 5] = SRAM_Data_o[i][ 47: 40];
							data_read[i][ 6] = SRAM_Data_o[i][ 55: 48];
							data_read[i][ 7] = SRAM_Data_o[i][ 63: 56];
							data_read[i][ 8] = SRAM_Data_o[i][ 71: 64];
							data_read[i][ 9] = SRAM_Data_o[i][ 79: 72];
							data_read[i][10] = SRAM_Data_o[i][ 87: 80];
							data_read[i][11] = SRAM_Data_o[i][ 95: 88];
							data_read[i][12] = SRAM_Data_o[i][103: 96];
							data_read[i][13] = SRAM_Data_o[i][111:104];
							data_read[i][14] = SRAM_Data_o[i][119:112];
							if(histo_add_cnt_reg != 15)
							data_read[i][15] = SRAM_Data_o[i][127:120];
						end
					end
				end
			end
		S_IN_H: begin
				if(histo_data_cnt_reg == 0)begin
					for(i = 0; i < 16; i = i + 1)begin
						if(first_reg)
							for(j = 0; j < 16; j = j + 1)
								data_read[i][j] = 0;
						else begin
							data_read[i][0]  = SRAM_Data_o[i][  7:  0];
							data_read[i][1]  = SRAM_Data_o[i][ 15:  8];
							data_read[i][2]  = SRAM_Data_o[i][ 23: 16];
							data_read[i][3]  = SRAM_Data_o[i][ 31: 24];
							data_read[i][4]  = SRAM_Data_o[i][ 39: 32];
							data_read[i][5]  = SRAM_Data_o[i][ 47: 40];
							data_read[i][6]  = SRAM_Data_o[i][ 55: 48];
							data_read[i][7]  = SRAM_Data_o[i][ 63: 56];
							data_read[i][8]  = SRAM_Data_o[i][ 71: 64];
							data_read[i][9]  = SRAM_Data_o[i][ 79: 72];
							data_read[i][10] = SRAM_Data_o[i][ 87: 80];
							data_read[i][11] = SRAM_Data_o[i][ 95: 88];
							data_read[i][12] = SRAM_Data_o[i][103: 96];
							data_read[i][13] = SRAM_Data_o[i][111:104];
							data_read[i][14] = SRAM_Data_o[i][119:112];
							if(histo_add_cnt_reg != 15)
							data_read[i][15] = SRAM_Data_o[i][127:120];
						end
					end
				end
				for(i = 0; i < 16; i = i + 1)begin
					data_write[i][histo_data_cnt_reg] = data_read[i][histo_data_cnt_reg] + stop_reg[i];
					if(histo_add_cnt_reg == 15)
						data_write[i][15] = 0;
				end
			end
		S_Busy:begin
				for(i = 0; i < 16; i = i + 1)begin
					for(j = 0; j < 16; j = j + 1)begin
						data_read[i][j] = 0;
						data_write[i][j] = 0;
					end
				end
			end
	endcase
end
// ---------------------------   Build Histogram   ---------------------------
// first
always@(posedge clk or negedge rst_n)begin
	if(!rst_n)
		first_reg <= 0;
	else 
		first_reg <= first;
end
always@(*)begin
	first = first_reg;
	case(cur_state)
		S_CAL0: first = 1;
		S_IN_S: first = 0;
	endcase
end

// ---------------------------   FETCH   ---------------------------
// fetch histogram
always@(posedge clk or negedge rst_n)begin
	if(!rst_n)begin
		for(i = 0; i < 255; i = i+1)
			histo_reg[i] <= 0;
		histo_cnt_reg <= 0;
		histo_cnt_reg_d1 <= 0;
	end
	else if(next_state == S_IDLE || next_state == S_Busy) begin
		for(i = 0; i < 255; i = i+1)
			histo_reg[i] <= 0;
		histo_cnt_reg <= 0;
		histo_cnt_reg_d1 <= 0;
	end
	else if(next_state == S_Wind)begin
		histo_cnt_reg <= 0;
		histo_cnt_reg_d1 <= histo_cnt_reg;
		if(mode_reg == 0)begin
			for(i = 0; i < 255; i = i+1)
				histo_reg[i] <= histo[i];
		end
		else begin
			for(i = 0; i < 255; i = i+1)
				histo_reg[i] <= histo[i];
		end
	end
	else if(cur_state == S_Feth)begin
		histo_cnt_reg <= histo_cnt;
		histo_cnt_reg_d1 <= histo_cnt_reg;
		for(i = 0; i < 255; i = i+1)
			histo_reg[i] <= histo[i];
	end
end

always@(posedge clk or negedge rst_n)begin
	if(!rst_n)begin
		fetch_histo_cnt_reg <= 0;
	end
	else begin
		fetch_histo_cnt_reg <= fetch_histo_cnt;
	end
end

always@(*)begin
	histo_cnt = histo_cnt_reg;
	fetch_histo_cnt = fetch_histo_cnt_reg;
	if(cur_state == S_Busy)
		fetch_histo_cnt = 0;
	for(i = 0; i < 255; i = i+1)
		histo[i] = histo_reg[i];
	if(cur_state == S_Feth)begin
		if(mode_reg != 1)begin
			histo[histo_cnt_reg_d1*16   ] = SRAM_Data_o[fetch_histo_cnt_reg][  7:  0];
			histo[histo_cnt_reg_d1*16+ 1] = SRAM_Data_o[fetch_histo_cnt_reg][ 15:  8];
			histo[histo_cnt_reg_d1*16+ 2] = SRAM_Data_o[fetch_histo_cnt_reg][ 23: 16];
			histo[histo_cnt_reg_d1*16+ 3] = SRAM_Data_o[fetch_histo_cnt_reg][ 31: 24];
			histo[histo_cnt_reg_d1*16+ 4] = SRAM_Data_o[fetch_histo_cnt_reg][ 39: 32];
			histo[histo_cnt_reg_d1*16+ 5] = SRAM_Data_o[fetch_histo_cnt_reg][ 47: 40];
			histo[histo_cnt_reg_d1*16+ 6] = SRAM_Data_o[fetch_histo_cnt_reg][ 55: 48];
			histo[histo_cnt_reg_d1*16+ 7] = SRAM_Data_o[fetch_histo_cnt_reg][ 63: 56];
			histo[histo_cnt_reg_d1*16+ 8] = SRAM_Data_o[fetch_histo_cnt_reg][ 71: 64];
			histo[histo_cnt_reg_d1*16+ 9] = SRAM_Data_o[fetch_histo_cnt_reg][ 79: 72];
			histo[histo_cnt_reg_d1*16+10] = SRAM_Data_o[fetch_histo_cnt_reg][ 87: 80];
			histo[histo_cnt_reg_d1*16+11] = SRAM_Data_o[fetch_histo_cnt_reg][ 95: 88];
			histo[histo_cnt_reg_d1*16+12] = SRAM_Data_o[fetch_histo_cnt_reg][103: 96];
			histo[histo_cnt_reg_d1*16+13] = SRAM_Data_o[fetch_histo_cnt_reg][111:104];
			histo[histo_cnt_reg_d1*16+14] = SRAM_Data_o[fetch_histo_cnt_reg][119:112];
			if(histo_cnt_reg_d1 != 15)
			histo[histo_cnt_reg_d1*16+15] = SRAM_Data_o[fetch_histo_cnt_reg][127:120];
			
			histo_cnt = histo_cnt_reg + 1;
			if(histo_cnt_reg_d1 == 15)
				fetch_histo_cnt = fetch_histo_cnt_reg + 1;
		end
		else begin
			histo[histo_cnt_reg_d1*16   ] = SRAM_Data_o[fetch_histo_cnt_reg][  7:  0] + SRAM_Data_o[fetch_histo_cnt_reg+1][  7:  0] + SRAM_Data_o[fetch_histo_cnt_reg+4][  7:  0] + SRAM_Data_o[fetch_histo_cnt_reg+5][  7:  0];
			histo[histo_cnt_reg_d1*16+ 1] = SRAM_Data_o[fetch_histo_cnt_reg][ 15:  8] + SRAM_Data_o[fetch_histo_cnt_reg+1][ 15:  8] + SRAM_Data_o[fetch_histo_cnt_reg+4][ 15:  8] + SRAM_Data_o[fetch_histo_cnt_reg+5][ 15:  8];
			histo[histo_cnt_reg_d1*16+ 2] = SRAM_Data_o[fetch_histo_cnt_reg][ 23: 16] + SRAM_Data_o[fetch_histo_cnt_reg+1][ 23: 16] + SRAM_Data_o[fetch_histo_cnt_reg+4][ 23: 16] + SRAM_Data_o[fetch_histo_cnt_reg+5][ 23: 16];
			histo[histo_cnt_reg_d1*16+ 3] = SRAM_Data_o[fetch_histo_cnt_reg][ 31: 24] + SRAM_Data_o[fetch_histo_cnt_reg+1][ 31: 24] + SRAM_Data_o[fetch_histo_cnt_reg+4][ 31: 24] + SRAM_Data_o[fetch_histo_cnt_reg+5][ 31: 24];
			histo[histo_cnt_reg_d1*16+ 4] = SRAM_Data_o[fetch_histo_cnt_reg][ 39: 32] + SRAM_Data_o[fetch_histo_cnt_reg+1][ 39: 32] + SRAM_Data_o[fetch_histo_cnt_reg+4][ 39: 32] + SRAM_Data_o[fetch_histo_cnt_reg+5][ 39: 32];
			histo[histo_cnt_reg_d1*16+ 5] = SRAM_Data_o[fetch_histo_cnt_reg][ 47: 40] + SRAM_Data_o[fetch_histo_cnt_reg+1][ 47: 40] + SRAM_Data_o[fetch_histo_cnt_reg+4][ 47: 40] + SRAM_Data_o[fetch_histo_cnt_reg+5][ 47: 40];
			histo[histo_cnt_reg_d1*16+ 6] = SRAM_Data_o[fetch_histo_cnt_reg][ 55: 48] + SRAM_Data_o[fetch_histo_cnt_reg+1][ 55: 48] + SRAM_Data_o[fetch_histo_cnt_reg+4][ 55: 48] + SRAM_Data_o[fetch_histo_cnt_reg+5][ 55: 48];
			histo[histo_cnt_reg_d1*16+ 7] = SRAM_Data_o[fetch_histo_cnt_reg][ 63: 56] + SRAM_Data_o[fetch_histo_cnt_reg+1][ 63: 56] + SRAM_Data_o[fetch_histo_cnt_reg+4][ 63: 56] + SRAM_Data_o[fetch_histo_cnt_reg+5][ 63: 56];
			histo[histo_cnt_reg_d1*16+ 8] = SRAM_Data_o[fetch_histo_cnt_reg][ 71: 64] + SRAM_Data_o[fetch_histo_cnt_reg+1][ 71: 64] + SRAM_Data_o[fetch_histo_cnt_reg+4][ 71: 64] + SRAM_Data_o[fetch_histo_cnt_reg+5][ 71: 64];
			histo[histo_cnt_reg_d1*16+ 9] = SRAM_Data_o[fetch_histo_cnt_reg][ 79: 72] + SRAM_Data_o[fetch_histo_cnt_reg+1][ 79: 72] + SRAM_Data_o[fetch_histo_cnt_reg+4][ 79: 72] + SRAM_Data_o[fetch_histo_cnt_reg+5][ 79: 72];
			histo[histo_cnt_reg_d1*16+10] = SRAM_Data_o[fetch_histo_cnt_reg][ 87: 80] + SRAM_Data_o[fetch_histo_cnt_reg+1][ 87: 80] + SRAM_Data_o[fetch_histo_cnt_reg+4][ 87: 80] + SRAM_Data_o[fetch_histo_cnt_reg+5][ 87: 80];
			histo[histo_cnt_reg_d1*16+11] = SRAM_Data_o[fetch_histo_cnt_reg][ 95: 88] + SRAM_Data_o[fetch_histo_cnt_reg+1][ 95: 88] + SRAM_Data_o[fetch_histo_cnt_reg+4][ 95: 88] + SRAM_Data_o[fetch_histo_cnt_reg+5][ 95: 88];
			histo[histo_cnt_reg_d1*16+12] = SRAM_Data_o[fetch_histo_cnt_reg][103: 96] + SRAM_Data_o[fetch_histo_cnt_reg+1][103: 96] + SRAM_Data_o[fetch_histo_cnt_reg+4][103: 96] + SRAM_Data_o[fetch_histo_cnt_reg+5][103: 96];
			histo[histo_cnt_reg_d1*16+13] = SRAM_Data_o[fetch_histo_cnt_reg][111:104] + SRAM_Data_o[fetch_histo_cnt_reg+1][111:104] + SRAM_Data_o[fetch_histo_cnt_reg+4][111:104] + SRAM_Data_o[fetch_histo_cnt_reg+5][111:104];
			histo[histo_cnt_reg_d1*16+14] = SRAM_Data_o[fetch_histo_cnt_reg][119:112] + SRAM_Data_o[fetch_histo_cnt_reg+1][119:112] + SRAM_Data_o[fetch_histo_cnt_reg+4][119:112] + SRAM_Data_o[fetch_histo_cnt_reg+5][119:112];
			if(histo_cnt_reg_d1 != 15)                                                                                                                                                                             
			histo[histo_cnt_reg_d1*16+15] = SRAM_Data_o[fetch_histo_cnt_reg][127:120] + SRAM_Data_o[fetch_histo_cnt_reg+1][127:120] + SRAM_Data_o[fetch_histo_cnt_reg+4][127:120] + SRAM_Data_o[fetch_histo_cnt_reg+5][127:120];
			
			histo_cnt = histo_cnt_reg + 1;
			if(histo_cnt_reg_d1 == 15)begin
				if(fetch_histo_cnt_reg == 0)
					fetch_histo_cnt = 2;
				else if(fetch_histo_cnt_reg == 2)
					fetch_histo_cnt = 8;
				else if(fetch_histo_cnt_reg == 8)
					fetch_histo_cnt = 10;
			end
		end
	end
end

// ---------------------------   WINDOW   ---------------------------
always@(posedge clk or negedge rst_n)begin
	if(!rst_n)begin
		wind_cnt_reg <= 0;
		win_cur_reg <= 0;
		win_max_reg <= 0;
		win_max_idx_reg <= 0;
		win_first_reg <= 0;
		win_first_idx_reg <= 0;
	end
	else if(next_state == S_IDLE || next_state == S_Busy) begin
		wind_cnt_reg <= 0;
		win_cur_reg <= 0;
		win_max_reg <= 0;
		win_max_idx_reg <= 0;
		win_first_reg <= 0;
		win_first_idx_reg <= 0;
	end
	else if(next_state == S_Wind)begin
		wind_cnt_reg <= wind_cnt;
		win_cur_reg <= win_cur;
		win_max_reg <= win_max;
		win_max_idx_reg <= win_max_idx;
		win_first_reg <= win_first;
		win_first_idx_reg <= win_first_idx;
	end
	else if(next_state == S_Dist) begin
		wind_cnt_reg <= 0;
		win_cur_reg <= 0;
		win_max_reg <= 0;
		win_max_idx_reg <= 0;
		win_first_reg <= 0;
		win_first_idx_reg <= 0;
	end
end
always@(*)begin
	wind_cnt = wind_cnt_reg;
	win_max = win_max_reg;
	win_max_idx = win_max_idx_reg;
	win_cur = win_cur_reg;
	win_first_idx = win_first_idx_reg;
	
	win_size = WINDOW_SIZE;
	
	if(wind_cnt_reg < win_size)begin
		win_cur = win_cur_reg + histo_reg[wind_cnt_reg];
		win_first = histo_reg[0];
		win_max = win_cur;
		win_max_idx = win_first_idx_reg;
	end
	else begin
		win_first = histo_reg[wind_cnt_reg-win_size];
		win_cur = win_cur_reg + histo_reg[wind_cnt_reg] - win_first;
		
		win_first_idx = win_first_idx_reg + 1;
	end
	
	if(win_max_reg < win_cur && wind_cnt_reg >= win_size && wind_cnt_reg <= 255)begin
		win_max = win_cur;
		win_max_idx = win_first_idx_reg + 1;
	end
	wind_cnt = wind_cnt_reg + 1;
end

// Store Distance
always@(posedge clk or negedge rst_n)begin
	if(!rst_n)begin
		dist_cnt_reg <= 0;
	end
	else if(next_state == S_IDLE || next_state == S_Busy) begin
		dist_cnt_reg <= 0;
	end
	else if(next_state == S_Dist) begin
		dist_cnt_reg <= dist_cnt + 1;
	end
end
always@(*)begin
	dist_cnt = dist_cnt_reg;
	dist_cal = win_max_idx_reg + 1;
end
always@(posedge clk or negedge rst_n)begin
	if(!rst_n)
		dist_cal_d1 <= 0;
	else
		dist_cal_d1 <= dist_cal;
end

endmodule
