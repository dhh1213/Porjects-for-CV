
.SUBCKT MAC6 CLK A[5] A[4] A[3] A[2] A[1] A[0]
+ B[5] B[4] B[3] B[2] B[1] B[0] 
+ MODE 
+ ACC[11] ACC[10] ACC[9] ACC[8] ACC[7] ACC[6] ACC[5] ACC[4] ACC[3] ACC[2] ACC[1] ACC[0]
+ OUT[12] OUT[11] OUT[10] OUT[9] OUT[8] OUT[7] OUT[6] OUT[5] OUT[4] OUT[3] OUT[2] OUT[1] OUT[0] 

x_clkb	CLK	clkB INV 
x_DFF_A0  A[0] CLK clkB A_D[0] DFF
x_DFF_A1  A[1] CLK clkB A_D[1] DFF
x_DFF_A2  A[2] CLK clkB A_D[2] DFF
x_DFF_A3  A[3] CLK clkB A_D[3] DFF
x_DFF_A4  A[4] CLK clkB A_D[4] DFF
x_DFF_A5  A[5] CLK clkB A_D[5] DFF

x_DFF_B0  B[0] CLK clkB B_D[0] DFF
x_DFF_B1  B[1] CLK clkB B_D[1] DFF
x_DFF_B2  B[2] CLK clkB B_D[2] DFF
x_DFF_B3  B[3] CLK clkB B_D[3] DFF
x_DFF_B4  B[4] CLK clkB B_D[4] DFF
x_DFF_B5  B[5] CLK clkB B_D[5] DFF
x_DFF_mo  MODE CLK clkB MODE_D DFF

x_DFF_ACC0  ACC[0]  CLK clkB ACC_D[0] DFF
x_DFF_ACC1  ACC[1]  CLK clkB ACC_D[1] DFF
x_DFF_ACC2  ACC[2]  CLK clkB ACC_D[2] DFF
x_DFF_ACC3  ACC[3]  CLK clkB ACC_D[3] DFF
x_DFF_ACC4  ACC[4]  CLK clkB ACC_D[4] DFF
x_DFF_ACC5  ACC[5]  CLK clkB ACC_D[5] DFF
x_DFF_ACC6  ACC[6]  CLK clkB ACC_D[6] DFF
x_DFF_ACC7  ACC[7]  CLK clkB ACC_D[7] DFF
x_DFF_ACC8  ACC[8]  CLK clkB ACC_D[8] DFF
x_DFF_ACC9  ACC[9]  CLK clkB ACC_D[9] DFF
x_DFF_ACC10 ACC[10] CLK clkB ACC_D[10] DFF
x_DFF_ACC11 ACC[11] CLK clkB ACC_D[11] DFF




X_B1 A_D[5] A_D[4] A_D[3] A_D[2] A_D[1] A_D[0] GND 
+ B_D[1] B_D[0] GND  
+ B1_pp[6] B1_pp[5] B1_pp[4] B1_pp[3] B1_pp[2] B1_pp[1] B1_pp[0] Booth

X_B2 A_D[5] A_D[4] A_D[3] A_D[2] A_D[1] A_D[0] GND 
+ B_D[3] B_D[2] B_D[1]  
+ B2_pp[6] B2_pp[5] B2_pp[4] B2_pp[3] B2_pp[2] B2_pp[1] B2_pp[0] Booth

X_B3 A_D[5] A_D[4] A_D[3] A_D[2] A_D[1] A_D[0] GND 
+ B_D[5] B_D[4] B_D[3]  
+ B3_pp[6] B3_pp[5] B3_pp[4] B3_pp[3] B3_pp[2] B3_pp[1] B3_pp[0] Booth

X_B1_Sbar B1_pp[6] B1_Sbar INV
X_B2_Sbar B2_pp[6] B2_Sbar INV
X_B3_Sbar B3_pp[6] B3_Sbar INV

**1st CSA
X_L1_0  B1_pp[0] B_D[1]    GND      S[0]  A1[0]  FA 
**V_d2    S[1] B1_pp[1] 0
X_L1_2  B1_pp[2]  B2_pp[0] B_D[3]   S[2]  A1[2]  FA
X_L1_3  B1_pp[3]  B2_pp[1]    GND   S[3]  A1[3]  FA
X_L1_4  B1_pp[4]  B2_pp[2] B3_pp[0] S[4]  A1[4]  FA
X_L1_5  B1_pp[5]  B2_pp[3] B3_pp[1] S[5]  A1[5]  FA
X_L1_6  B1_pp[6]  B2_pp[4] B3_pp[2] S[6]  A1[6]  FA
X_L1_7  B1_pp[6]  B2_pp[5] B3_pp[3] S[7]  A1[7]  FA
X_L1_8  B1_pp[6]  B2_pp[6] B3_pp[4] S[8]  A1[8]  FA
X_L1_9  B1_Sbar   B2_Sbar  B3_pp[5] S[9]  A1[9]  FA
X_L1_10 	VDD	  GND	   B3_pp[6] S[10] A1[10] FA

**2nd CSA
X_L2_3  S[3]    A1[2] 	GND		 S2[3]  A2[3]   FA
X_L2_4  S[4]    A1[3]	B_D[5]   S2[4]  A2[4]   FA
X_L2_5  S[5]    A1[4] 	GND	 	 S2[5]  A2[5]   FA
X_L2_6  S[6]    A1[5] 	GND	 	 S2[6]  A2[6]   FA
X_L2_7  S[7]    A1[6] 	GND		 S2[7]  A2[7]   FA
X_L2_8  S[8]    A1[7] 	GND		 S2[8]  A2[8]   FA
X_L2_9  S[9]    A1[8] 	GND		 S2[9]  A2[9]   FA
X_L2_10 S[10]   A1[9] 	GND		 S2[10] A2[10]  FA
X_L2_11 B3_Sbar A1[10] 	GND		 S2[11] A2[11]  FA

**ACC -> XOR

X_ACC0  MODE_D ACC_D[0]  MB[0]  XOR
X_ACC1  MODE_D ACC_D[1]  MB[1]  XOR
X_ACC2  MODE_D ACC_D[2]  MB[2]  XOR
X_ACC3  MODE_D ACC_D[3]  MB[3]  XOR
X_ACC4  MODE_D ACC_D[4]  MB[4]  XOR
X_ACC5  MODE_D ACC_D[5]  MB[5]  XOR
X_ACC6  MODE_D ACC_D[6]  MB[6]  XOR
X_ACC7  MODE_D ACC_D[7]  MB[7]  XOR
X_ACC8  MODE_D ACC_D[8]  MB[8]  XOR
X_ACC9  MODE_D ACC_D[9]  MB[9]  XOR
X_ACC10 MODE_D ACC_D[10] MB[10] XOR
X_ACC11 MODE_D ACC_D[11] MB[11] XOR

**3rd CSA
X_L3_0   S[0]       MODE_D	MB[0]    S3[0]   A3[0]	FA
X_L3_1   B1_pp[1]   A1[0] 	MB[1]    S3[1]   A3[1]	FA
X_L3_2   S[2]       GND 	MB[2]    S3[2]   A3[2]	FA
X_L3_3   S2[3]      GND 	MB[3]    S3[3]   A3[3]	FA
X_L3_4   S2[4]      A2[3] 	MB[4]    S3[4]   A3[4]	FA
X_L3_5   S2[5]      A2[4] 	MB[5]    S3[5]   A3[5]	FA
X_L3_6   S2[6]      A2[5] 	MB[6]    S3[6]   A3[6]	FA
X_L3_7   S2[7]      A2[6] 	MB[7]    S3[7]   A3[7]	FA
X_L3_8   S2[8]      A2[7] 	MB[8]    S3[8]   A3[8]	FA
X_L3_9   S2[9]      A2[8] 	MB[9]    S3[9]   A3[9]	FA
X_L3_10  S2[10]     A2[9] 	MB[10]   S3[10]  A3[10]	FA
X_L3_11  S2[11]     A2[10] 	MB[11]   S3[11]  A3[11]	FA
X_L3_12  S2[11]     A2[11] 	MB[11]   S3[12]  A3[12]	FA

**DFF

x_DFF_S3_0  S3[0]  CLK clkB S3_D[0] DFF
x_DFF_S3_1  S3[1]  CLK clkB S3_D[1] DFF
x_DFF_S3_2  S3[2]  CLK clkB S3_D[2] DFF
x_DFF_S3_3  S3[3]  CLK clkB S3_D[3] DFF
x_DFF_S3_4  S3[4]  CLK clkB S3_D[4] DFF
x_DFF_S3_5  S3[5]  CLK clkB S3_D[5] DFF
x_DFF_S3_6  S3[6]  CLK clkB S3_D[6] DFF
x_DFF_S3_7  S3[7]  CLK clkB S3_D[7] DFF
x_DFF_S3_8  S3[8]  CLK clkB S3_D[8] DFF
x_DFF_S3_9  S3[9]  CLK clkB S3_D[9] DFF
x_DFF_S3_10 S3[10] CLK clkB S3_D[10] DFF
x_DFF_S3_11 S3[11] CLK clkB S3_D[11] DFF
x_DFF_S3_12 S3[12] CLK clkB S3_D[12] DFF

x_DFF_A3_0  A3[0]  CLK clkB A3_D[0] DFF
x_DFF_A3_1  A3[1]  CLK clkB A3_D[1] DFF
x_DFF_A3_2  A3[2]  CLK clkB A3_D[2] DFF
x_DFF_A3_3  A3[3]  CLK clkB A3_D[3] DFF
x_DFF_A3_4  A3[4]  CLK clkB A3_D[4] DFF
x_DFF_A3_5  A3[5]  CLK clkB A3_D[5] DFF
x_DFF_A3_6  A3[6]  CLK clkB A3_D[6] DFF
x_DFF_A3_7  A3[7]  CLK clkB A3_D[7] DFF
x_DFF_A3_8  A3[8]  CLK clkB A3_D[8] DFF
x_DFF_A3_9  A3[9]  CLK clkB A3_D[9] DFF
x_DFF_A3_10 A3[10] CLK clkB A3_D[10] DFF
x_DFF_A3_11 A3[11] CLK clkB A3_D[11] DFF


**CRA
X_L4_1	S3_D[1]  A3_D[0] 	GND	   ans[1]  Cout[1]  FA
X_L4_2	S3_D[2]  A3_D[1]  Cout[1]  ans[2]  Cout[2]  FA
X_L4_3	S3_D[3]  A3_D[2]  Cout[2]  ans[3]  Cout[3]  FA
X_L4_4	S3_D[4]  A3_D[3]  Cout[3]  ans[4]  Cout[4]  FA
X_L4_5	S3_D[5]  A3_D[4]  Cout[4]  ans[5]  Cout[5]  FA
X_L4_6	S3_D[6]  A3_D[5]  Cout[5]  ans[6]  Cout[6]  FA
X_L4_7	S3_D[7]  A3_D[6]  Cout[6]  ans[7]  Cout[7]  FA
X_L4_8	S3_D[8]  A3_D[7]  Cout[7]  ans[8]  Cout[8]  FA
X_L4_9	S3_D[9]  A3_D[8]  Cout[8]  ans[9]  Cout[9]  FA
X_L4_10	S3_D[10] A3_D[9]  Cout[9]  ans[10] Cout[10] FA
X_L4_11	S3_D[11] A3_D[10] Cout[10] ans[11] Cout[11] FA
X_L4_12	S3_D[12] A3_D[11] Cout[11] ans[12] Cout[12] FA


**DFF
x_DFF0  S3_D[0]  CLK clkB OUT[0] DFF
x_DFF1  ans[1]  CLK clkB OUT[1] DFF
x_DFF2  ans[2]  CLK clkB OUT[2] DFF
x_DFF3  ans[3]  CLK clkB OUT[3] DFF
x_DFF4  ans[4]  CLK clkB OUT[4] DFF
x_DFF5  ans[5]  CLK clkB OUT[5] DFF
x_DFF6  ans[6]  CLK clkB OUT[6] DFF
x_DFF7  ans[7]  CLK clkB OUT[7] DFF
x_DFF8  ans[8]  CLK clkB OUT[8] DFF
x_DFF9  ans[9]  CLK clkB OUT[9] DFF
x_DFF10 ans[10] CLK clkB OUT[10] DFF
x_DFF11 ans[11] CLK clkB OUT[11] DFF
x_DFF12 ans[12] CLK clkB OUT[12] DFF

.ends


**----------------------------------------------------------------------------------------------------------------
**----------------------------------------------------------------------------------------------------------------
**----------------------------------------------------------------------------------------------------------------

.SUBCKT booth y[6] y[5] y[4] y[3] y[2] y[1] y[0]
+ x[2] x[1] x[0] 
+ pp[6] pp[5] pp[4] pp[3] pp[2] pp[1] pp[0]

x_Booth_E_xor0 x[0] x[1] e1 XOR
x_Booth_E_Not0 x[0] not0 INV
x_Booth_E_Not1 x[1] not1 INV
x_Booth_E_Not2 x[2] not2 INV
x_Booth_E_NAND1 x[0] 		x[1] 		not2 NAND1_out  NAND3
x_Booth_E_NAND2 not0 		not1		x[2] NAND2_out  NAND3
x_Booth_E_NAND3 NAND1_out   NAND2_out 	e2 	        NAND2

**selector

x_booth_B0_nand0 y[0] 		   e2 	    B0_nand0_out NAND2
x_booth_B0_nand1 y[1] 		   e1  	    B0_nand1_out NAND2
x_booth_B0_NAND2 B0_nand0_out  B0_nand1_out B0_NAND2_out NAND2
x_booth_B0_XOR3  B0_NAND2_out  X[2]			pp[0]		 XOR

x_booth_B1_nand0 y[1] 		 e2 		    B1_nand0_out NAND2
x_booth_B1_nand1 y[2] 		 e1  		    B1_nand1_out NAND2
x_booth_B1_NAND2  B1_nand0_out  B1_nand1_out   B1_NAND2_out  NAND2
x_booth_B1_XOR3  B1_NAND2_out	 X[2]			pp[1]		 XOR

x_booth_B2_nand0 y[2] 		 e2 		    B2_nand0_out NAND2
x_booth_B2_nand1 y[3] 		 e1  		    B2_nand1_out NAND2
x_booth_B2_NAND2  B2_nand0_out  B2_nand1_out   B2_NAND2_out  NAND2
x_booth_B2_XOR3  B2_NAND2_out	 X[2]			pp[2]		 XOR

x_booth_B3_nand0 y[3] 		 e2 		    B3_nand0_out NAND2
x_booth_B3_nand1 y[4] 		 e1  		    B3_nand1_out NAND2
x_booth_B3_NAND2  B3_nand0_out  B3_nand1_out   B3_NAND2_out  NAND2
x_booth_B3_XOR3  B3_NAND2_out	 X[2]			pp[3]		 XOR

x_booth_B4_nand0 y[4] 		 e2 		    B4_nand0_out NAND2
x_booth_B4_nand1 y[5] 		 e1  		    B4_nand1_out NAND2
x_booth_B4_NAND2  B4_nand0_out  B4_nand1_out   B4_NAND2_out  NAND2
x_booth_B4_XOR3  B4_NAND2_out	 X[2]			pp[4]		 XOR

x_booth_B5_nand0 y[5] 		 e2 		    B5_nand0_out NAND2
x_booth_B5_nand1 y[6] 		 e1  		    B5_nand1_out NAND2
x_booth_B5_NAND2  B5_nand0_out  B5_nand1_out   B5_NAND2_out  NAND2
x_booth_B5_XOR3  B5_NAND2_out	 X[2]			pp[5]		 XOR

x_booth_B6_nand0 y[6] 		 e2 		    B6_nand0_out NAND2
x_booth_B6_nand1 y[6] 		 e1  		    B6_nand1_out NAND2
x_booth_B6_NAND2  B6_nand0_out  B6_nand1_out   B6_NAND2_out  NAND2
x_booth_B6_XOR3  B6_NAND2_out	 X[2]			pp[6]		 XOR

.ends



.SUBCKT FA INA INB CIN SOUT COUT

mp1	CoutB GND	VDD	 VDD P_18_G2  l=0.18u  w=wp_FA
mn1 n1 	  INA	GND  GND N_18_G2  l=0.18u  w=wn_FA_COUT
mn2 n1 	  INB	GND  GND N_18_G2  l=0.18u  w=wn_FA_COUT
mn3 CoutB CIN	n1   GND N_18_G2  l=0.18u  w=wn_FA_COUT
mn4 n2 	  INB	GND  GND N_18_G2  l=0.18u  w=wn_FA_COUT
mn5 CoutB INA	n2   GND N_18_G2  l=0.18u  w=wn_FA_COUT

x_Cout CoutB COUT INV

mp2	SoutB GND	VDD	 VDD P_18_G2  l=0.18u  w=wp_FA
mn6 n3	  INA	GND  GND N_18_G2  l=0.18u  w=wn_FA_SOUT
mn7 n3	  INB	GND  GND N_18_G2  l=0.18u  w=wn_FA_SOUT
mn8 n3	  CIN	GND  GND N_18_G2  l=0.18u  w=wn_FA_SOUT
mn9 SoutB CoutB	n3   GND N_18_G2  l=0.18u  w=wn_FA_SOUT
mn10 n4   INB	GND  GND N_18_G2  l=0.18u  w=wn_FA_SOUT
mn11 n5   INA	n4   GND N_18_G2  l=0.18u  w=wn_FA_SOUT
mn12 SoutB CIN	n5   GND N_18_G2  l=0.18u  w=wn_FA_SOUT

x_Sout SoutB SOUT INV

.ends

.SUBCKT INV Vin Vout
mp   Vout   GND   VDD   VDD   P_18_G2  l=0.18u  w=wp_INV
mn   Vout   VIn   GND   GND   N_18_G2  l=0.18u  w=wn_INV
.ends 

.SUBCKT DFF_INV Vin Vout
mp   Vout   Vin   VDD   VDD   P_18_G2  l=0.18u  w=wp_DFFINV
mn   Vout   VIn   GND   GND   N_18_G2  l=0.18u  w=wn_DFFINV
.ends 

.SUBCKT NAND2 a b Vout
**mp1	Vout a	VDD	VDD	P_18_G2	 l=0.18u  w=wp_NAND2
**mp2	Vout b	VDD	VDD	P_18_G2	 l=0.18u  w=wp_NAND2
mp2	Vout GND	VDD	VDD	P_18_G2	 l=0.18u  w=wp_NAND2

mn1 Vout a		n1  GND N_18_G2  l=0.18u  w=wn_NAND2
mn2 n1	 b		GND GND N_18_G2  l=0.18u  w=wn_NAND2
.ENDS 


.SUBCKT NAND3 a b c Vout
**mp1	Vout a	VDD	VDD	P_18_G2	 l=0.18u  w=wp_NAND3
**mp2	Vout b	VDD	VDD	P_18_G2	 l=0.18u  w=wp_NAND3
**mp3	Vout c	VDD	VDD	P_18_G2	 l=0.18u  w=wp_NAND3
mp1	Vout GND	VDD	VDD	P_18_G2	 l=0.18u  w=wp_NAND3

mn1 Vout a		n1  GND N_18_G2  l=0.18u  w=wn_NAND3
mn2 n1	 b		n2  GND N_18_G2  l=0.18u  w=wn_NAND3
mn3 n2	 c		GND GND N_18_G2  l=0.18u  w=wn_NAND3
.ENDS 



.SUBCKT AND2 a b Vout
x_ANDGATE_nand a b NAND_out NAND2
x_ANDGATE_inv	 NAND_out Vout INV
.ENDS 

 
.SUBCKT NOR2 a b Vout
**mp1	Vout a	p1	VDD	P_18_G2	 l=0.18u  w=wp_NOR
**mp2	p1   b	VDD	VDD	P_18_G2	 l=0.18u  w=wp_NOR
mp1 Vout GND	VDD	VDD	P_18_G2	 l=0.18u  w=wp_NOR

mn1 Vout a		GND GND N_18_G2  l=0.18u  w=wn_NOR
mn2 Vout b		GND GND N_18_G2  l=0.18u  w=wn_NOR
.ENDS 


.SUBCKT XOR a b Vout

x_xor_Abar a	A_bar INV
x_xor_Bbar b	B_bar INV

mp1	Vout GND   VDD VDD	P_18_G2	l=0.18u	w=wp_xor
                                                                                           
mn1	Vout a	   n1  GND	N_18_G2	l=0.18u	w=wn_xor
mn2	n1	 b	   GND GND	N_18_G2	l=0.18u	w=wn_xor
mn3 Vout A_bar n2  GND	N_18_G2 l=0.18u w=wn_xor
mn4 n2	 B_bar GND GND 	N_18_G2 l=0.18u w=wn_xor
.ENDS 


.SUBCKT buffer a out

x_buffer_1 a 1 INV
x_buffer_2 1 out INV
.ENDS


**DFF
.SUBCKT DFF input clk clkB Q
mp1 3 	clk  input   VDD     P_18_G2  l=0.18u  w=wp_DFF
mn1 input	clkB 3	 GND	 N_18_G2  l=0.18u  w=wn_DFF
X_2 3	2	 DFF_INV
mp2 4 	clkB 2   VDD     P_18_G2  l=0.18u  w=wp_DFF
mn2 2	clk  4	 GND	 N_18_G2  l=0.18u  w=wn_DFF
X_4 4	Q	 DFF_INV
.ends


.param wp_FA		= 	1.3u
.param wn_FA_COUT	= 	1.5u
.param wn_FA_SOUT	= 	1.4u

.param wp_INV		=	1.2u
.param wn_INV		=	1.7u

***
.param wp_xor		=	1.2u
.param wn_xor		=	1.7u
***
.param wp_NAND2		=	1.2u
.param wn_NAND2		=	1.7u
***
.param wp_NAND3		=	1.2u
.param wn_NAND3		=	1.7u
.param wp_NAND4		=	1.2u
.param wn_NAND4		=	1.7u
***
.param wp_NOR		=	1.2u
.param wn_NOR		=	1.7u
***
.param wp_DFFINV	=	1.2u
.param wn_DFFINV	=	1.2u
.param wp_DFF		=	0.44u
.param wn_DFF		=	0.44u

.end


