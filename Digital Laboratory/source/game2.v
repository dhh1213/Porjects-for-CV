`timescale 1ns / 1ps
module game2(pCLK, pCLK2, vgaPixel_Y, vgaPixel_X, pReset, vga_out, BTN, animate, start, over, resault);
input pCLK; //100Mhz
input pCLK2; //65.0 Mhz
input [10:0] vgaPixel_Y; //from VGA
input [10:0] vgaPixel_X; //From VGA
input pReset;
input [3:0]BTN;          //  BTN  UDLR
input animate;
input start;

// 生成各種降頻 CLK

output reg [11:0]  vga_out = 12'd0; 
output reg over = 0;    // 1 = over;    0 = not over yet;
output reg resault = 0; //  1 = win; 0 = lose

reg [18:0]div;
always @(posedge pCLK)
    div <= div + 1;
wire CLK_Scan = div[16];
wire En_CLK = div[17];
wire En_Bu_CLK = div[18];

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

//Jet
//address out
reg [13:0] Addr_jet = 14'd0;
wire [11:0] Jet_out;
//jet image
blk_mem_gen_1 jet (
    .clka(pCLK2),
    .addra(Addr_jet),  // input  [13:0] address
    .douta(Jet_out)    // output  [11:0] jet
);
reg [10:0]Jet_top = 610;
reg [10:0]Jet_left = 448;
wire [6:0]Jet_addr_y = vgaPixel_Y - Jet_top;
wire [6:0]Jet_addr_x = vgaPixel_X - Jet_left;

//////////////////  enemy
reg [15:0]Addr_En = 16'b0;
wire [11:0]En_out;
blk_mem_gen_3 Enemy(
    .clka(pCLK2),
    .addra(Addr_En),  // input  [16:0] address
    .douta(En_out)    // output  [11:0] Enemy
);
reg [10:0]En_top = 38;
reg [10:0]En_left = 340;
wire [7:0]En_addr_y = vgaPixel_Y - En_top;
wire [8:0]En_addr_x = vgaPixel_X - En_left;
integer En_cnt=0;
integer En_move;

///////////////  Jet Location 控制飛機移動方向，共上, 下, 左, 右, 左上, 左下, 右上, 右下 8個方向
always @(posedge CLK_Scan)
begin
    if(pReset || start)
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
            if(Jet_left > 0)Jet_left <= Jet_left - 1;
            Jet_top <= Jet_top;
            end
        4'b0001 :   // Right
            begin
            if(Jet_left < 896)Jet_left <= Jet_left + 1;
            Jet_top <= Jet_top;
            end
        4'b1010 :   // Up Left
            begin
            if(Jet_left > 0)Jet_left <= Jet_left - 1;
            if(Jet_top > 2) Jet_top <= Jet_top - 1;
            end
        4'b1001 :   // Up Right
            begin
            if(Jet_left < 896)Jet_left <= Jet_left + 1;
            if(Jet_top > 2) Jet_top <= Jet_top - 1;
            end
        4'b0110 :   // Down Left
            begin
            if(Jet_left > 0)Jet_left <= Jet_left - 1;
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

///////////////////////life initial
reg [6:0]Jet_life=25;
reg [7:0]En_life=200;

//////////////// Bullet 初始值

reg [7:0]Addr_Bu;
wire [11:0]Bu_out;
blk_mem_gen_2 Bu(
    .clka(pCLK2),
    .addra(Addr_Bu),  // input  [13:0] address
    .douta(Bu_out)    // output  [11:0] jet
);
reg Bu_en[9:0];
reg [10:0]Bu_top[9:0];
reg [10:0]Bu_left[9:0];
reg [4:0]Bu_addr_y[9:0];
reg [4:0]Bu_addr_x[9:0];
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
        En_life=200;
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
            if((Bu_top[bmove] <= En_top + 200) && (Bu_top[bmove]+23 >= En_top) && (Bu_left[bmove]+9 >= En_left) && (Bu_left[bmove] <= En_left + 255) && Bu_en[bmove] == 1 )
                begin
                    Bu_en[bmove] = 0;
                    if((En_life!=0) && (Jet_life!=0) )
                        En_life = En_life - 1;
                end
            if(Bu_en[bmove] == 1)
                Bu_top[bmove] = Bu_top[bmove] - 1;
            else Bu_top[bmove] = Bu_top[bmove];
         end
    end
end
////////////////////////////////////////////////////結束子彈區

///////          假隨機數列 用來判定魔王移動方向     //////////
 reg [7:0] rand_num;
parameter seed = 8'b1111_1111;
integer rcnt;
always@(posedge En_CLK)
begin
   if(pReset || start)
       rand_num  <= seed;
   else 
     begin
       if(rcnt==100) begin
          rcnt=0;
           rand_num[0] <= rand_num[1] ;
           rand_num[1] <= rand_num[2] + rand_num[7];
           rand_num[2] <= rand_num[3] + rand_num[7];
           rand_num[3] <= rand_num[4] ;
           rand_num[4] <= rand_num[5] + rand_num[7];
           rand_num[5] <= rand_num[6] + rand_num[7];
           rand_num[6] <= rand_num[7] ;
           rand_num[7] <= rand_num[0] + rand_num[7];  
           end
       else rcnt<=rcnt+1;
     end
end
wire [2:0]rand;
assign rand = {rand_num[3],rand_num[6],rand_num[2]};

////////////////////////////////////////////////////

//////////////////////////////////魔王控制區
always @(posedge En_CLK)
begin
    if((pReset || start))begin
        En_top <= 38; En_left <= 340;
     end
    else begin
    
    ////////魔王移動
        if(En_cnt>=500)begin
            En_cnt=0;En_move<=rand; end
        else begin En_cnt=En_cnt+1; En_move=En_move; end

        case(En_move) 
        0:begin if( En_top < 200 ) En_top  = En_top+1 ;end     //down
        1:begin if( En_left < 768) En_left = En_left+1;end     //right
        2:begin if( En_top > 34 ) En_top  = En_top-1 ;end       //up
        3:begin if( En_left > 5) En_left = En_left-1; end      //left
        4:begin if( En_top  < 200 && En_left < 768) begin En_top  <= En_top+1; En_left  <= En_left+1; end end //down right
        5:begin if( En_top  > 200 && En_left > 5) begin En_top  <= En_top+1; En_left  <= En_left-1; end end //down left
        6:begin if( En_top  > 34 && En_left < 768) begin En_top  <= En_top-1 ; En_left <= En_left+1; end end //up right
        7:begin if( En_top  > 34 && En_left > 5) begin En_top  <= En_top-1 ; En_left <= En_left-1; end end //up left
        endcase
            
        end
end
//////////////// 魔王子彈控制

reg [9:0]Addr_En_Bu;
wire [11:0]En_Bu_out;
blk_mem_gen_4 En_Bu(
    .clka(pCLK2),
    .addra(Addr_En_Bu),  // input  [13:0] address
    .douta(En_Bu_out)    // output  [11:0] jet
);
reg En_Bu_en[9:0];
reg [10:0]En_Bu_top[9:0];
reg [10:0]En_Bu_left[9:0];
reg [5:0]En_Bu_addr_y[9:0] ;
reg [5:0]En_Bu_addr_x[9:0] ;
integer En_bcounter=0;
integer en_bnum;
integer en_bhit;
integer en_bmove;
reg [2:0]typecounter;

always @(posedge En_Bu_CLK)
begin
    if(pReset || start)
        begin
        Jet_life=25;
        for(i = 0; i <= 9; i = i + 1)
            begin
                En_Bu_en[i] <= 0;
                En_Bu_top[i] <= 0;
                En_Bu_left[i] <= 0;
                typecounter <= 0;
            end
        end
    else
    begin
    case(typecounter)
   /////////////魔王第一種攻擊模式
     0,1:begin
        if(En_bcounter >= 1500)begin
            for(en_bnum = 0; en_bnum <= 9; en_bnum = en_bnum + 1)
                En_Bu_en[en_bnum] = 0;
            En_bcounter = 0;
            typecounter <= typecounter + 1;
        end
        else  En_bcounter = En_bcounter + 2;
        end
        //////////魔王第二種攻擊模式
      2,3,4,5:begin
            if(En_bcounter >= 800)begin
                for(en_bnum = 0; en_bnum <= 9; en_bnum = en_bnum + 1)
                En_Bu_en[en_bnum] = 0;
                En_bcounter = 0;
                if(typecounter>=5) 
                    typecounter=0;
                else typecounter <= typecounter + 1;
            end
                else  En_bcounter = En_bcounter + 2;
         end
      endcase  
        
      //////////子彈移動與判定
        case(typecounter)
            0,1:begin
                for(en_bnum = 0; en_bnum <= 9; en_bnum = en_bnum + 1)begin
                    if(En_bcounter == (en_bnum + 1) * 100)begin
                        En_Bu_en[en_bnum] <= 1;
                        En_Bu_top[en_bnum] <= En_top + 200;
                        En_Bu_left[en_bnum] <= En_left + 116;  
                    end
                end
                for(en_bmove = 0; en_bmove <= 9; en_bmove = en_bmove + 1)begin
                    if(En_Bu_top[en_bmove] >= 707)
                        En_Bu_en[en_bmove] = 0;
                    if((En_Bu_top[en_bmove]+32 >= Jet_top) && (En_Bu_top[en_bmove] <= Jet_top+105) && (En_Bu_left[en_bmove]+32 >= Jet_left) && (En_Bu_left[en_bmove] <= Jet_left + 128) && En_Bu_en[en_bmove] == 1 )
                        begin
                            En_Bu_en[en_bmove] = 0;
                            if( (En_life!=0) && (Jet_life!=0) )
                                Jet_life = Jet_life - 1;
                        end
                    else begin
                        if(En_Bu_en[en_bmove] == 1)
                            En_Bu_top[en_bmove] = En_Bu_top[en_bmove] + 2;
                        else En_Bu_top[en_bmove] = En_Bu_top[en_bmove];
                    end
                end
             end
             
             2,3,4,5:begin
                for(en_bnum = 0; en_bnum <= 9; en_bnum = en_bnum + 1)begin
                    if(En_bcounter == 300)begin
                        En_Bu_en[en_bnum] <= 1;
                        En_Bu_top[en_bnum] <= En_top + 200;
                        En_Bu_left[en_bnum] <= En_left + 116;  
                    end
                    else if(En_bcounter == 700)begin
                    end
                    
                end

                for(en_bmove = 0; en_bmove <= 9; en_bmove = en_bmove + 1)begin
                    if((En_Bu_top[en_bmove] >= 707) && (En_Bu_en[en_bmove] == 1))
                        En_Bu_en[en_bmove] = 0;
                    else if(En_Bu_left[en_bmove] <= 5 && (En_Bu_en[en_bmove] == 1))
                        En_Bu_en[en_bmove] = 0;
                    else if(En_Bu_left[en_bmove] >= 992 && (En_Bu_en[en_bmove] == 1))
                        En_Bu_en[en_bmove] = 0;
                    if((En_Bu_top[en_bmove]+32 >= Jet_top) && (En_Bu_top[en_bmove] <= Jet_top+105) && (En_Bu_left[en_bmove]+32 >= Jet_left) && (En_Bu_left[en_bmove] <= Jet_left + 128) && (En_Bu_en[en_bmove] == 1 ))
                        begin
                            En_Bu_en[en_bmove] = 0;
                            if( (En_life!=0) && (Jet_life!=0) )
                                Jet_life = Jet_life - 1;
                        end
                        
                    if(En_Bu_en[en_bmove] == 1)begin
                        En_Bu_top[en_bmove] = En_Bu_top[en_bmove] + 2;
                        if(en_bmove % 2 == 0)
                            En_Bu_left[en_bmove] = En_Bu_left[en_bmove] + en_bmove / 2;
                        else
                            En_Bu_left[en_bmove] = En_Bu_left[en_bmove] - en_bmove / 2;
                     end
                    else En_Bu_top[en_bmove] = En_Bu_top[en_bmove];
                    end
                 end
         endcase
     
         
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
        if (En_life <= 0)
        begin
            over <= 1;
            resault <= 1;
        end
        if (Jet_life <= 0)
        begin
            over <= 1;
            resault <= 0;
        end
    end
end

/////////////////////////////////////   螢幕顯示區

integer k=0;
reg [11:0]En_life_out;
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
        //血條去背
            if (~((vgaPixel_X >= En_left + 21 ) && ( vgaPixel_X <= En_left + 241 ) && ( vgaPixel_Y >= En_top - 33 ) && ( vgaPixel_Y <= En_top - 3 ))) begin
            //Where ever is the pixel pointer, take the rgb value from the BRAM and print it.
                Addr_back <= {picPixel_Y_B[7:0], picPixel_X_B[7:0]};
                vga_out <= back_out;
            end
        //戰機去背
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
                if(En_Bu_out != 0)
                    vga_out <= En_Bu_out;
                else if(En_life_out != 0)
                    vga_out <= En_life_out;
                else begin
                    if(En_out != 4095)
                    begin
                        vga_out <= En_out;
                    end
                    if(En_out == 4095)
                    begin
                        if(Jet_out != 4095)
                            vga_out <= Jet_out;   // output jet
                        if(Jet_out == 4095)
                            vga_out <= back_out;
                    end
                end
                Addr_jet <= Jet_addr_y*128 + Jet_addr_x;        //  address
            end
        end
///////////////   子彈顯示
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
                    if(Bu_out != 4095)
                            vga_out <= Bu_out;   // output Bullet
                    if(Bu_out == 4095)
                            vga_out <= back_out;

                    Addr_Bu <= Bu_addr_y[k]*9 + Bu_addr_x[k];        // address
                end
            end
        end

//////////////////////   我方戰機血條顯示
    if(pReset || start)
        begin
            Jet_life_out <= 12'd0;
        end
    else
    begin
        Jet_life_out <= 12'd0;
        if ((vgaPixel_X >= Jet_left + 10 ) && ( vgaPixel_X <= Jet_left + 120 ) && ( vgaPixel_Y >= Jet_top + 107 ) && ( vgaPixel_Y <= Jet_top + 122 ))
        begin
            if ((vgaPixel_X >=Jet_left + 15 ) && ( vgaPixel_X <= ( Jet_left + 15 + Jet_life*4 )) && ( vgaPixel_Y >= Jet_top + 111 ) && ( vgaPixel_Y <= Jet_top + 117))//RED
                Jet_life_out <= RED;
            else if ((vgaPixel_X >=Jet_left + 15 ) && ( vgaPixel_X <= ( Jet_left + 15 + 100 - Jet_life*4 )) && ( vgaPixel_Y >= Jet_top + 111 ) && ( vgaPixel_Y <= Jet_top + 117) )//BLACK
                Jet_life_out <= BLACK;
            else if ((vgaPixel_X >= Jet_left + 10 ) && ( vgaPixel_X <= Jet_left + 120 ) && ( vgaPixel_Y >= Jet_top + 107 ) && ( vgaPixel_Y <= Jet_top + 110)) //UP
                Jet_life_out <= WHITE;
            else if ((vgaPixel_X >= Jet_left + 10 ) && ( vgaPixel_X <= Jet_left + 15) && ( vgaPixel_Y >= Jet_top + 111 ) && ( vgaPixel_Y <= Jet_top + 117 ))//LEFT
                Jet_life_out <= WHITE;
            else if ((vgaPixel_X >= Jet_left + 115 ) && ( vgaPixel_X <= Jet_left + 120) && (vgaPixel_Y >= Jet_top + 111 ) && ( vgaPixel_Y <= Jet_top + 117 ))//RIGHT
                Jet_life_out <= WHITE;
            else if ((vgaPixel_X >=Jet_left + 10 ) && ( vgaPixel_X <= Jet_left + 120) && (vgaPixel_Y >= Jet_top + 118 ) && ( vgaPixel_Y <= Jet_top + 122 ))//DOWN
                Jet_life_out <= WHITE;
            vga_out <= Jet_life_out;
        end        
    end
 
/////////////////////   魔王顯示
    if(pReset || start)
        begin
            Addr_En <= 16'd0;
        end
    else
        begin
         Addr_En <= 0;
        if ((vgaPixel_X >= En_left ) && ( vgaPixel_X <= (En_left + 255)) && (vgaPixel_Y >= En_top ) && ( vgaPixel_Y <= (En_top + 199)))
            begin
                if(Bu_out != 4095)
                        vga_out <= Bu_out;
                else 
                begin
                    if(En_out != 4095)
                        vga_out <= En_out;   // output enemy
                    if(En_out == 4095)
                    begin  // 設置顯示順序、去背
                            if(Jet_out != 4095)
                                vga_out <= Jet_out;
                            else if(Jet_life_out != 0)
                                vga_out <= Jet_life_out;
                            else
                                vga_out <= back_out;
                    end
                end
                Addr_En <= En_addr_y*256 + En_addr_x;        //  address
            end
        end

/////////////////////////////  魔王子彈顯示
    if(pReset || start)
        begin
            Addr_En_Bu <= 10'd0;
        end
    else
        begin
        Addr_En_Bu <= 0;
        for(k=0;k<=9;k=k+1) 
            begin
              if ((vgaPixel_X >= En_Bu_left[k] ) && ( vgaPixel_X <= (En_Bu_left[k] + 31)) && (vgaPixel_Y >= En_Bu_top[k] ) && ( vgaPixel_Y <= (En_Bu_top[k] + 31)) && En_Bu_en[k]== 1)
                begin
                    En_Bu_addr_y[k] = vgaPixel_Y -  En_Bu_top[k];
                    En_Bu_addr_x[k] = vgaPixel_X -  En_Bu_left[k] + 4;
                    if(Bu_out == 0) // 設置顯示順序、去背
                            vga_out <= back_out;
                    if(Bu_out != 0)
                            vga_out <=  En_Bu_out;   // output Bullet
                    Addr_En_Bu <=  En_Bu_addr_y[k]*32 +  En_Bu_addr_x[k];        // address
                end
            end
        end
        
///////////////////////////////   魔王血條顯示
    if(pReset || start)
        begin
            En_life_out <= 12'd0;
        end
    else
    begin
        En_life_out <= 12'd0;
        if ((vgaPixel_X >= En_left + 21 ) && ( vgaPixel_X <= En_left + 241 ) && ( vgaPixel_Y >= En_top - 33 ) && ( vgaPixel_Y <= En_top - 3 )) begin
            if ((vgaPixel_X >= En_left + 21 ) && ( vgaPixel_X <= En_left + 241 ) && ( vgaPixel_Y >= En_top - 33 ) && ( vgaPixel_Y <= En_top - 28)) //UP
                En_life_out <= WHITE;
            else if ((vgaPixel_X >= En_left + 21 ) && ( vgaPixel_X <= En_left + 31 ) && ( vgaPixel_Y >= En_top - 28 ) && ( vgaPixel_Y <= En_top - 8 ))//LEFT
                En_life_out <= WHITE;
            else if ((vgaPixel_X >= En_left + 231 ) && ( vgaPixel_X <= En_left + 241 ) && (vgaPixel_Y >= En_top - 28 ) && ( vgaPixel_Y <= En_top - 8 ))//RIGHT
                En_life_out <= WHITE;
            else if ((vgaPixel_X >=En_left + 21 ) && ( vgaPixel_X <= En_left + 241 ) && (vgaPixel_Y >= En_top - 8 ) && ( vgaPixel_Y <= En_top -3 ))//DOWN
                En_life_out <= WHITE;
            else if ((vgaPixel_X >=En_left + 31 ) && ( vgaPixel_X <= ( En_left + 31 + En_life )) && ( vgaPixel_Y >= En_top - 28 ) && ( vgaPixel_Y <= En_top - 8) )//RED
                En_life_out <= RED;
            else if ((vgaPixel_X >=En_left + 31 ) && ( vgaPixel_X <= ( En_left + 31 + 200 - En_life )) && ( vgaPixel_Y >= En_top - 28 ) && ( vgaPixel_Y <= En_top - 8) )//BLACK
                En_life_out <= BLACK;
            vga_out <= En_life_out;
        end
    end
    
end
endmodule