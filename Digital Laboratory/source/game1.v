`timescale 1ns / 1ps
module game1(pCLK, pCLK2, vgaPixel_Y, vgaPixel_X, pReset, vga_out, BTN, animate, start, over, resault);
input pCLK;             //100Mhz
input pCLK2;            //65.0 Mhz
input [10:0] vgaPixel_Y; //from VGA
input [10:0] vgaPixel_X; //From VGA
input pReset;           //reset btn
input [3:0]BTN;         //  BTN  Up_Down_Left_Right
input animate;          // 60Hz
input start;            // Game Start

output reg [11:0]  vga_out = 12'd0; // VGA輸出
output reg over = 0;    // 1 = over;    0 = not over yet;
output reg resault = 0; //  1 = win; 0 = lose

// 生成各種降頻 CLK
reg [18:0]div;
always @(posedge pCLK)
    div <= div + 1;
wire CLK_Scan = div[16];
wire Mn_CLK = div[17];
wire Mn_Bu_CLK = div[17];

//Background
wire [10:0] picPixel_Y_B,picPixel_X_B;
//Zooming for background from 256x192 to 1024x768
assign picPixel_Y_B =  vgaPixel_Y [10:2];
assign picPixel_X_B =  vgaPixel_X [10:2];
//Address to be passed to the Block RAM
reg [15:0] Addr_back = 16'd0;
//Out from the Block RAM
wire [11:0] back_out;
// Block ROM to store the image information of background
blk_mem_gen_0 back (
  .clka(pCLK2),    // input wire clka
  .addra(Addr_back),  // input wire [17 : 0] addra
  .douta(back_out)  // output wire [11 : 0] douta
);

//Jet Display
//address out
reg [13:0] Addr_jet = 14'd0;
wire [11:0] Jet_out;
//jet image
blk_mem_gen_1 jet (
    .clka(pCLK2),
    .addra(Addr_jet),  // input  [13:0] address
    .douta(Jet_out)    // output  [11:0] jet
);
reg [10:0]Jet_top = 610;                    // Jet initial location Y
reg [10:0]Jet_left = 448;                   // Jet initial location X
wire [6:0]Jet_addr_y = vgaPixel_Y - Jet_top;  // Jet Address first 7 digit
wire [6:0]Jet_addr_x = vgaPixel_X - Jet_left; // Jet Address last 7 digit

//////////////////  minion display
reg [13:0]Addr_Mn = 14'b0;
wire [11:0]Mn_out;
blk_mem_gen_8 Minion(
    .clka(pCLK2),
    .addra(Addr_Mn),  // input  [13:0] address
    .douta(Mn_out)    // output  [11:0] Mn_out
);


///////////////  Jet Location 控制飛機移動方向，共上, 下, 左, 右, 左上, 左下, 右上, 右下 8個方向
always @(posedge CLK_Scan)
begin
    if(pReset || start) // Jet location 初始化
        begin
            Jet_top <= 610;
            Jet_left <= 448;
        end
    else
    begin
        case(BTN)
        4'b0000 :   //None
            begin
            Jet_left <= Jet_left;
            Jet_top <= Jet_top;
            end
        4'b1000 :   //Up
            begin
            Jet_left <= Jet_left;
            if(Jet_top > 2) Jet_top <= Jet_top - 1;
            end
        4'b0100 :   // Down
            begin
            Jet_left <= Jet_left;
            if(Jet_top < 633 )Jet_top <= Jet_top + 1;
            end
        4'b0010 :   // Left
            begin
            if(Jet_left > 2)Jet_left <= Jet_left - 1;
            Jet_top <= Jet_top;
            end
        4'b0001 :   // Right
            begin
            if(Jet_left < 896)Jet_left <= Jet_left + 1;
            Jet_top <= Jet_top;
            end
        4'b1010 :   // Up Left
            begin
            if(Jet_left > 2)Jet_left <= Jet_left - 1;
            if(Jet_top > 2) Jet_top <= Jet_top - 1;
            end
        4'b1001 :   // Up Right
            begin
            if(Jet_left < 896)Jet_left <= Jet_left + 1;
            if(Jet_top > 2) Jet_top <= Jet_top - 1;
            end
        4'b0110 :   // Down Left
            begin
            if(Jet_left > 2)Jet_left <= Jet_left - 1;
            if(Jet_top < 633 )Jet_top <= Jet_top + 1;
            end
        4'b0101 :   // Down Right
            begin
            if(Jet_left < 896)Jet_left <= Jet_left + 1;
            if(Jet_top < 633 )Jet_top <= Jet_top + 1;
            end
        endcase
    end
end

// Minion location
reg [10:0]Mn_top[1:0];
reg [10:0]Mn_left[1:0];
wire [7:0]Mn_addr_y[1:0];
wire [8:0]Mn_addr_x[1:0];

// Minions 紀錄參數，

integer Mn_cnt=0;
integer Mn_move;

initial begin
Mn_top[0]=20;   // 小怪1 初始y座標
Mn_left[0]=40;  //小怪1  初始x座標
Mn_top[1]=160;  // 小怪2 初始y座標
Mn_left[1]=540; // 小怪2 初始y座標
end
//顯示小怪
assign Mn_addr_y[0] = vgaPixel_Y - Mn_top[0];   
assign Mn_addr_x[0] = vgaPixel_X - Mn_left[0];  
assign Mn_addr_y[1] = vgaPixel_Y - Mn_top[1];   
assign Mn_addr_x[1] = vgaPixel_X - Mn_left[1];  

///////////////////////life initial
reg [6:0]Jet_life=50;
reg [7:0]Mn_life[1:0];
initial begin
Mn_life[0]=50;
Mn_life[1]=50;
end

//////////////// Bullet  初始值
// Display
reg [7:0]Addr_Bu;
wire [11:0]Bu_out;
blk_mem_gen_2 Bu(
    .clka(pCLK2),
    .addra(Addr_Bu),  // input  [7:0] address
    .douta(Bu_out)    // output  [11:0] jet
);
reg [10:0]Bu_top[9:0];
reg [10:0]Bu_left[9:0];
reg [4:0]Bu_addr_y[9:0];
reg [4:0]Bu_addr_x[9:0];

//Bu 控制參數
reg Bu_en[9:0];
integer bcounter = 0;

///////////////  子彈控制區
integer i = 0;
integer j = 0;
integer bmove = 0;
reg shoot_cnt = 0;
integer bnum;
reg damage = 0;
always @(posedge CLK_Scan)
begin
    if(pReset || start)
        begin
        Mn_life[0]=50;
        Mn_life[1]=50;
        for(i = 0; i <= 9; i = i + 1)
            begin
                Bu_en[i] <= 0;
                Bu_top[i] <= 0;
                Bu_left[i] <= 0;
            end
        end
    else
    
    begin
        /////////////////////////////  子彈定時產生
        
        if(bcounter >= 1000)
        bcounter = 0;
        else  bcounter = bcounter + 1;
        
        for(bnum=0;bnum<=9;bnum=bnum+1)begin
            if(bcounter==(bnum+1)*100)
                begin
                Bu_en[bnum] <= 1;
                Bu_top[bnum] <= Jet_top - 23;
                Bu_left[bnum] <= Jet_left + 61;    
            end
        end
        
        /////////////////////////////  子彈移動與判定
        for(bmove = 0; bmove <= 9;bmove = bmove + 1)begin
            if(Bu_top[bmove] == 0)
                Bu_en[bmove] = 0;
            
            if((Bu_top[bmove] <= Mn_top[0] + 100) && (Bu_top[bmove]+23 >= Mn_top[0]) && (Bu_left[bmove]+9 >= Mn_left[0]) && (Bu_left[bmove] <= Mn_left[0] + 100) && Bu_en[bmove] == 1 && Mn_life[0]>0)
                begin
                    Bu_en[bmove] = 0;
                    if((Mn_life[0]!=0) && (Jet_life!=0) )
                        Mn_life[0] = Mn_life[0] - 1;
                end
            else if((Bu_top[bmove] <= Mn_top[1] + 100) && (Bu_top[bmove]+23 >= Mn_top[1]) && (Bu_left[bmove]+9 >= Mn_left[1]) && (Bu_left[bmove] <= Mn_left[1] + 100) && Bu_en[bmove] == 1 && Mn_life[1]>0 )
                begin
                    Bu_en[bmove] = 0;
                    if((Mn_life[1]!=0) && (Jet_life!=0) )
                        Mn_life[1] = Mn_life[1] - 1;
                end
             
               
            if(Bu_en[bmove] == 1)
                Bu_top[bmove] = Bu_top[bmove] - 1;
            else Bu_top[bmove] = Bu_top[bmove];
         end
    end
end

////////////////////////////////////////////////////結束子彈區


reg Mnstate0=0;
reg Mnstate1=1;

//////////////////////////////////小怪控制區

always @(posedge Mn_CLK)
begin
    if((pReset || start))begin ////reset
        Mn_top[0]=20;
        Mn_left[0]=40;
        Mn_top[1]=160;
        Mn_left[1]=540;
     end
    else begin
    
        //////////////第一隻小怪左右移動
        
        case(Mnstate0) 
        0:begin
            Mn_left[0] = Mn_left[0] + 1 ;
            if( (Mn_left[0]>=924) && (Mn_life[0]>0) )
                 Mnstate0=1;
            else  Mnstate0=0;
        end
        1:begin
            Mn_left[0] = Mn_left[0] - 1 ;
            if( (Mn_left[0]<=0) && (Mn_life[0]>0) )
                Mnstate0=0;
            else Mnstate0=1;
        end
        endcase
        
        //////////////第二隻小怪左右移動
        case(Mnstate1) 
        0:begin
            Mn_left[1] = Mn_left[1] + 1 ;
            if( (Mn_left[1]>=924) && (Mn_life[1]>0) )
                 Mnstate1=1;
            else  Mnstate1=0;
        end
        1:begin
            Mn_left[1] = Mn_left[1] - 1 ;
            if( (Mn_left[1]<=0) && (Mn_life[1]>0) )
                Mnstate1=0;
            else Mnstate1=1;
        end
        endcase
            
            
        end
end

//////////////// 小怪子彈控制區 
reg [7:0]Addr_Mn_Bu;
wire [11:0]Mn_Bu_out;
blk_mem_gen_9 Mn_Bu(
    .clka(pCLK2),
    .addra(Addr_Mn_Bu),  // input  [7:0] address
    .douta(Mn_Bu_out)    // output  [11:0] jet
);
reg [10:0]Mn_Bu_top[19:0];
reg [10:0]Mn_Bu_left[19:0];
reg [4:0]Mn_Bu_addr_y[19:0] ;
reg [4:0]Mn_Bu_addr_x[19:0] ;
reg Mn_Bu_en[19:0];
integer Mn_bcounter=0;
integer en_bnum;
integer en_bhit;
integer en_bmove;

always @(posedge Mn_Bu_CLK) 
begin
    if(pReset || start)
        begin
        Jet_life=50;
        for(i = 0; i <= 19; i = i + 1)
            begin
                Mn_Bu_en[i] <= 0;
                Mn_Bu_top[i] <= 0;
                Mn_Bu_left[i] <= 0;
            end
        end
    else
    begin     
        if(Mn_bcounter >= 2000)begin
            for(en_bnum = 0; en_bnum <= 19; en_bnum = en_bnum + 1)
                Mn_Bu_en[en_bnum] = 0;
                Mn_bcounter = 0;
          end
        else  Mn_bcounter = Mn_bcounter + 1;
        
        ////////小怪子彈產生與移動

        for(en_bnum = 0; en_bnum <= 19; en_bnum = en_bnum + 1)begin
            if(Mn_bcounter == (en_bnum + 1) * 100)begin
                Mn_Bu_en[en_bnum] <= 1;
                if(( ((en_bnum%2)==0) && (Mn_life[0]>0))|| (Mn_life[1]<=0) )begin
                    Mn_Bu_top[en_bnum] <= Mn_top[0] + 100;
                    Mn_Bu_left[en_bnum] <= Mn_left[0] + 45;
                end
                else begin
                    Mn_Bu_top[en_bnum] <= Mn_top[1] + 100;
                    Mn_Bu_left[en_bnum] <= Mn_left[1] + 45;
                end
            end
        end
        
        
        ///////////////////////判定子彈碰到戰機or跑出畫面
        for(en_bmove = 0; en_bmove <= 19; en_bmove = en_bmove + 1)begin
            if(Mn_Bu_top[en_bmove] >= 707 && (Mn_Bu_en[en_bmove] == 1) )
                Mn_Bu_en[en_bmove] = 0;
            if((Mn_Bu_top[en_bmove]+22 >= Jet_top) && (Mn_Bu_top[en_bmove] <= Jet_top+105) && (Mn_Bu_left[en_bmove]+9 >= Jet_left) && (Mn_Bu_left[en_bmove] <= Jet_left + 128) && Mn_Bu_en[en_bmove] == 1 )
                begin
                    Mn_Bu_en[en_bmove] = 0;
                    if( ((Mn_life[0]!=0) || (Mn_life[1]!=0) ) && (Jet_life!=0) )
                        Jet_life = Jet_life - 1;
                end
            else begin
                if(Mn_Bu_en[en_bmove] == 1)
                    Mn_Bu_top[en_bmove] = Mn_Bu_top[en_bmove] + 2;
                else Mn_Bu_top[en_bmove] = Mn_Bu_top[en_bmove];
            end
        end
    end
end


/////////////////////////////////////   關卡結束判定
always @(posedge pCLK2)
begin
    if(pReset || start)
    begin
        over <= 0;
        resault <= 0;
    end
    else
    begin
        if ( (Mn_life[0] <= 0) && (Mn_life[1] <= 0) ) //條件一:兩隻小怪血量歸零
        begin
            over <= 1;
            resault <= 1;
        end
        
        if (Jet_life <= 0)                          //條件二:我方戰機血量歸零
        begin
            over <= 1;
            resault <= 0;
        end
    end
end

/////////////////////////////////////   螢幕顯示區

integer k=0;
reg [11:0]Mn_life_out;
reg [11:0]Jet_life_out;
wire [11:0] RED = 12'b1111_0000_0000;
wire [11:0] WHITE = 12'b1111_1111_1111;
wire [11:0] BLACK = 12'b0001_0000_0000;


always @(posedge pCLK2)
begin

//////////////////   背景顯示
    if (pReset || start)
        begin
            vga_out <= 12'd0;
            Addr_back = 16'd0;
        end
    else
        begin 
            // 血條1 去背
            if (~((vgaPixel_X >= Mn_left[0] + 22 ) && ( vgaPixel_X <= Mn_left[0] + 82 ) && ( vgaPixel_Y >= Mn_top[0] - 18 ) && ( vgaPixel_Y <= Mn_top[0] - 3 ) && ( Mn_life[0] > 0 ))) begin
            //Where ever is the pixel pointer, take the rgb value from the BRAM and print it.
                Addr_back <= {picPixel_Y_B[7:0], picPixel_X_B[7:0]};
                vga_out <= back_out;
            end
            
            // 血條2 去背
            if (~((vgaPixel_X >= Mn_left[1] + 22 ) && ( vgaPixel_X <= Mn_left[1] + 82 ) && ( vgaPixel_Y >= Mn_top[1] - 18 ) && ( vgaPixel_Y <= Mn_top[1] - 3 ) && ( Mn_life[1] > 0 ))) begin
            //Where ever is the pixel pointer, take the rgb value from the BRAM and print it.
                Addr_back <= {picPixel_Y_B[7:0], picPixel_X_B[7:0]};
                vga_out <= back_out;
            end
            
            // Jet 去背
            if (~((vgaPixel_X >= Jet_left + 10 ) && ( vgaPixel_X <= Jet_left + 120 ) && ( vgaPixel_Y >= Jet_top + 107 ) && ( vgaPixel_Y <= Jet_top + 122 ))) begin
                Addr_back <= {picPixel_Y_B[7:0], picPixel_X_B[7:0]};
                vga_out <= back_out;
            end
        end
        
//////////////////   我方戰機顯示
    if(pReset || start)
        begin
            Addr_jet <= 14'd0;
        end
    else
        begin
            Addr_jet <= 0;
            if ((vgaPixel_X >= Jet_left ) && ( vgaPixel_X <= (Jet_left + 127)) && (vgaPixel_Y >= Jet_top ) && ( vgaPixel_Y <= (Jet_top + 104)))
            begin // 設置顯示順序、去背
                if(Mn_Bu_out != 4095)
                    vga_out <= Mn_Bu_out;
                else if(Mn_life_out != 0)
                    vga_out <= Mn_life_out;
                else begin
                    if(Mn_out != 4095)
                    begin
                        vga_out <= Mn_out;
                    end
                    if(Mn_out == 4095)
                    begin
                        if(Jet_out != 4095)
                            vga_out <= Jet_out;   // output jet
                        if(Jet_out == 4095)
                            vga_out <= back_out;
                    end
                end
                Addr_jet <= Jet_addr_y*128 + Jet_addr_x;        //  set address
            end
        end
        
//////////////////   子彈顯示
    if(pReset || start)
        begin
            Addr_Bu <= 10'd0;
        end
    else
        begin
        Addr_Bu <= 0;
        for(k=0;k<=9;k=k+1) begin
            if ((vgaPixel_X >= Bu_left[k] ) && ( vgaPixel_X <= (Bu_left[k] + 8)) && (vgaPixel_Y >= Bu_top[k] ) && ( vgaPixel_Y <= (Bu_top[k] + 22)) && Bu_en[k]== 1)
                begin
                    Bu_addr_y[k] = vgaPixel_Y - Bu_top[k];
                    Bu_addr_x[k] = vgaPixel_X - Bu_left[k];
                    if(Bu_out != 4095)// 設置顯示順序、去背
                            vga_out <= Bu_out;   // output Bullet
                    if(Bu_out == 4095)
                            vga_out <= back_out;

                    Addr_Bu <= Bu_addr_y[k]*9 + Bu_addr_x[k];    // set address
                end
            end
        end

/////////////////   我方戰機血條顯示
 
    if(pReset || start)
        begin
            Jet_life_out <= 12'd0;
        end
    else
    begin
        Jet_life_out <= 12'd0;
        if ((vgaPixel_X >= Jet_left + 10 ) && ( vgaPixel_X <= Jet_left + 120 ) && ( vgaPixel_Y >= Jet_top + 107 ) && ( vgaPixel_Y <= Jet_top + 122 ))
        begin
            if ((vgaPixel_X >=Jet_left + 15 ) && ( vgaPixel_X <= ( Jet_left + 15 + Jet_life*2 )) && ( vgaPixel_Y >= Jet_top + 111 ) && ( vgaPixel_Y <= Jet_top + 117)) // RED
                Jet_life_out <= RED;
            else if ((vgaPixel_X >=Jet_left + 15 ) && ( vgaPixel_X <= ( Jet_left + 15 + 100 - Jet_life*2 )) && ( vgaPixel_Y >= Jet_top + 111 ) && ( vgaPixel_Y <= Jet_top + 117) ) // BLACK
                Jet_life_out <= BLACK;
            else if ((vgaPixel_X >= Jet_left + 10 ) && ( vgaPixel_X <= Jet_left + 120 ) && ( vgaPixel_Y >= Jet_top + 107 ) && ( vgaPixel_Y <= Jet_top + 110)) // UP
                Jet_life_out <= WHITE;
            else if ((vgaPixel_X >= Jet_left + 10 ) && ( vgaPixel_X <= Jet_left + 15) && ( vgaPixel_Y >= Jet_top + 111 ) && ( vgaPixel_Y <= Jet_top + 117 )) // LEFT
                Jet_life_out <= WHITE;
            else if ((vgaPixel_X >= Jet_left + 115 ) && ( vgaPixel_X <= Jet_left + 120) && (vgaPixel_Y >= Jet_top + 111 ) && ( vgaPixel_Y <= Jet_top + 117 )) // RIGHT
                Jet_life_out <= WHITE;
            else if ((vgaPixel_X >=Jet_left + 10 ) && ( vgaPixel_X <= Jet_left + 120) && (vgaPixel_Y >= Jet_top + 118 ) && ( vgaPixel_Y <= Jet_top + 122 )) // DOWN
                Jet_life_out <= WHITE;
            vga_out <= Jet_life_out;
        end        
    end
 
//////////////////   小怪顯示

    if(pReset || start)
        begin
            Addr_Mn <= 16'd0;
        end
    else
        begin
         Addr_Mn <= 0;
        if ((vgaPixel_X >= Mn_left[0] ) && ( vgaPixel_X <= (Mn_left[0] + 100)) && (vgaPixel_Y >= Mn_top[0] ) && ( vgaPixel_Y <= (Mn_top[0] + 99)) && (Mn_life[0]>0))
            begin // 設置顯示順序、去背
                if(Bu_out != 4095)
                        vga_out <= Bu_out;
                else 
                begin
                    if(Mn_out != 4095)
                        vga_out <= Mn_out;   // output enemy
                    if(Mn_out == 4095)
                    begin
                            if(Jet_out != 4095)
                                vga_out <= Jet_out;
                            else if(Jet_life_out != 0)
                                vga_out <= Jet_life_out;
                            else
                                vga_out <= back_out;
                    end
                    Addr_Mn <= Mn_addr_y[0]*100 + Mn_addr_x[0];        //  set address
                end
           end
                
        if ((vgaPixel_X >= Mn_left[1] ) && ( vgaPixel_X <= (Mn_left[1] + 100)) && (vgaPixel_Y >= Mn_top[1] ) && ( vgaPixel_Y <= (Mn_top[1] + 99))&& (Mn_life[1]>0))
            begin// 設置顯示順序、去背
                if(Bu_out != 4095)
                        vga_out <= Bu_out;
                else 
                begin
                    if(Mn_out != 4095)
                        vga_out <= Mn_out;   // output enemy
                    if(Mn_out == 4095)
                    begin
                            if(Jet_out != 4095)
                                vga_out <= Jet_out;
                            else if(Jet_life_out != 0)
                                vga_out <= Jet_life_out;
                            else
                                vga_out <= back_out;
                    end
                    Addr_Mn <= Mn_addr_y[1]*100 + Mn_addr_x[1];        // set  address 
                end
            end
        end

///////////////////////  小怪子彈顯示
    if(pReset || start)
        begin
            Addr_Mn_Bu <= 10'd0;
        end
    else
        begin
        Addr_Mn_Bu <= 0;
            for(k=0;k<=19;k=k+1) 
            begin
                if ((vgaPixel_X >= Mn_Bu_left[k] ) && ( vgaPixel_X <= (Mn_Bu_left[k] + 8)) && (vgaPixel_Y >= Mn_Bu_top[k] ) && ( vgaPixel_Y <= (Mn_Bu_top[k] + 22)) && Mn_Bu_en[k]== 1)
                begin
                    Mn_Bu_addr_y[k] = vgaPixel_Y -  Mn_Bu_top[k];
                    Mn_Bu_addr_x[k] = vgaPixel_X -  Mn_Bu_left[k];
                    if(Mn_Bu_out == 4095) // 設置顯示順序、去背
                            vga_out <= back_out;
                    if(Mn_Bu_out != 4095)
                            vga_out <=  Mn_Bu_out;   // output Bullet
                    Addr_Mn_Bu <=  Mn_Bu_addr_y[k]*9 +  Mn_Bu_addr_x[k];        // set address
                end
            end
        end
        
///////////////////////////   小怪1血條顯示
    if(pReset || start)
        begin
            Mn_life_out <= 12'd0;
        end
    else
    begin
        Mn_life_out <= 12'd0;
        if ((vgaPixel_X >= Mn_left[0] + 22 ) && ( vgaPixel_X <= Mn_left[0] + 82 ) && ( vgaPixel_Y >= Mn_top[0] - 18 ) && ( vgaPixel_Y <= Mn_top[0] - 3 ) && (Mn_life[0]>0) ) begin
            if ((vgaPixel_X >= Mn_left[0] + 22 ) && ( vgaPixel_X <= Mn_left[0] + 82 ) && ( vgaPixel_Y >= Mn_top[0] - 18 ) && ( vgaPixel_Y <= Mn_top[0] - 14)) // UP
                Mn_life_out <= WHITE;
            else if ((vgaPixel_X >= Mn_left[0] + 22 ) && ( vgaPixel_X <= Mn_left[0] + 27 ) && ( vgaPixel_Y >= Mn_top[0] - 18 ) && ( vgaPixel_Y <= Mn_top[0] - 3 )) // LEFT
                Mn_life_out <= WHITE;
            else if ((vgaPixel_X >= Mn_left[0] + 77 ) && ( vgaPixel_X <= Mn_left[0] + 82 ) && (vgaPixel_Y >= Mn_top[0] - 18 ) && ( vgaPixel_Y <= Mn_top[0] - 3 )) // RIGHT
                Mn_life_out <= WHITE;
            else if ((vgaPixel_X >=Mn_left[0] + 22 ) && ( vgaPixel_X <= Mn_left[0] + 82 ) && (vgaPixel_Y >= Mn_top[0] - 8 ) && ( vgaPixel_Y <= Mn_top[0] -3 )) // DOWN
                Mn_life_out <= WHITE;
            else if ((vgaPixel_X >=Mn_left[0] + 27 ) && ( vgaPixel_X <= ( Mn_left[0] + 27 + Mn_life[0] )) && ( vgaPixel_Y >= Mn_top[0] - 14 ) && ( vgaPixel_Y <= Mn_top[0] - 8) ) // RED
                Mn_life_out <= RED;
            else if ((vgaPixel_X >=Mn_left[0] + 27 ) && ( vgaPixel_X <= ( Mn_left[0] + 27 + 50 - Mn_life[0] )) && ( vgaPixel_Y >= Mn_top[0] - 14 ) && ( vgaPixel_Y <= Mn_top[0] - 8) ) // BLACK
                Mn_life_out <= BLACK;
            vga_out <= Mn_life_out;
        end
        ///////////////////////////   小怪1血條顯示
        if ((vgaPixel_X >= Mn_left[1] + 22 ) && ( vgaPixel_X <= Mn_left[1] + 82 ) && ( vgaPixel_Y >= Mn_top[1] - 18 ) && ( vgaPixel_Y <= Mn_top[1] - 3 ) && (Mn_life[1]>0) ) begin
            if ((vgaPixel_X >= Mn_left[1] + 22 ) && ( vgaPixel_X <= Mn_left[1] + 82 ) && ( vgaPixel_Y >= Mn_top[1] - 18 ) && ( vgaPixel_Y <= Mn_top[1] - 14)) // UP
                Mn_life_out <= WHITE;
            else if ((vgaPixel_X >= Mn_left[1] + 22 ) && ( vgaPixel_X <= Mn_left[1] + 27 ) && ( vgaPixel_Y >= Mn_top[1] - 18 ) && ( vgaPixel_Y <= Mn_top[1] - 3 )) // LEFT
                Mn_life_out <= WHITE;
            else if ((vgaPixel_X >= Mn_left[1] + 77 ) && ( vgaPixel_X <= Mn_left[1] + 82 ) && (vgaPixel_Y >= Mn_top[1] - 18 ) && ( vgaPixel_Y <= Mn_top[1] - 3 )) // RIGHT
                Mn_life_out <= WHITE;
            else if ((vgaPixel_X >=Mn_left[1] + 22 ) && ( vgaPixel_X <= Mn_left[1] + 82 ) && (vgaPixel_Y >= Mn_top[1] - 8 ) && ( vgaPixel_Y <= Mn_top[1] -3 )) // DOWN
                Mn_life_out <= WHITE;
            else if ((vgaPixel_X >=Mn_left[1] + 27 ) && ( vgaPixel_X <= ( Mn_left[1] + 27 + Mn_life[1] )) && ( vgaPixel_Y >= Mn_top[1] - 14 ) && ( vgaPixel_Y <= Mn_top[1] - 8) ) // RED
                Mn_life_out <= RED;
            else if ((vgaPixel_X >=Mn_left[1] + 27 ) && ( vgaPixel_X <= ( Mn_left[1] + 27 + 50 - Mn_life[1] )) && ( vgaPixel_Y >= Mn_top[1] - 14 ) && ( vgaPixel_Y <= Mn_top[1] - 8) ) // BLACK
                Mn_life_out <= BLACK;
            vga_out <= Mn_life_out;
        end
     end
     
end

endmodule