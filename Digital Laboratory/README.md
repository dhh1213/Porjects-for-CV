# Final Project of Digital Laboratory
## Project Requirement
* Required to use FPGA to develop any topics.
* Form a group with two people.
## Project Abstract
*	Designed an aircraft shooting game with Verilog on the FPGA (Digilent Nexys4 DDR).
*	Utilized IP catalog of FPGA to input images and display objects, background and UI on screen by VGA output.
## Result
[<img src="https://github.com/dhh1213/Porjects-Record/blob/042aaa68a5f56f0940770f5d97bd3d6033d897b3/Digital%20Laboratory/images/start.jpg" width="600">](https://www.youtube.com/watch?v=L7aoIpFIz1E)
(<-video link)

TAs have given this project a high score, since most techniques of this project are self-researched.

```Since the video and report both are written in Mandarin, I briefly translate report into English as below.```

### Outline of The project
* We could use buttons on FPGA to control aircraft movement and attack.
* There are two stages in the game. 
  * First stage is equipped with two minions with horizon movement, vertical attack and 50 health points.
  * The second stage is equipped with one boss with randomized movement, two attacking modes and 200 health points.
* The Gaming Flow Chart:
<img src="https://github.com/dhh1213/Porjects-Record/blob/767c8690eff176f1b91f60a2d4fb907125473df0/Digital%20Laboratory/images/flow%20chart.jpg" width="600">

### Project Principles
* VGA screen output
  * Implemented a 1024*768@60Hz VGA signal output.
* MATLAB and block memory
  * Convert images through MATLAB into coe files and store them in memory by IP Catalog.

### Key design blocks
#### Structure of circuit:
<img src="https://github.com/dhh1213/Porjects-Record/blob/faa22af5e1c9e84338ad8af56e89c6688dedc98e/Digital%20Laboratory/images/code%20structure.png" width="800">

* Display
  * Display scaled background, objects, and health points bar.
* Control
  * Control the movement of aircraft and restrict it within the game screen.
* Enemy
  * Including enemy movement, simple pattern or random, and attack pattern.
* Bullet
  * Define bullet movement, and identify if the bullet was hit or not.
* pseudo-random
  * Provide a parameter for the boss' random movement.
* Background removal
  * Define the order of displaying images and objects, especially when objects overlapped.
