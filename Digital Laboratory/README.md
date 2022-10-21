# Digital Laboratory Final Project
## Project Requirement
* Required to use FPGA to develop any topics.
* Form a group with two people.
## Project Abstract
*	Designed an aircraft shooting game with Verilog on the FPGA (Digilent Nexys4 DDR).
*	Utilized IP catalog of FPGA to input images and display objects, background and UI on screen by VGA output.
## Result
[<img src="../Digital Laboratory/images/start.jpg" width="600">](https://www.youtube.com/watch?v=L7aoIpFIz1E)
(<-video link)

TAs have given this project a high score since most techniques of this project are self-researched.

```Since the video and report both are written in Mandarin, I briefly translate report into English as below.```

### Outline of The project
* We could use buttons on FPGA to control aircraft movement and attack.
* There are two stages in the game. 
  * First stage is equipped with two minions with horizon movement, vertical attack and 50 health points.
  * The second stage is equipped with one boss with randomized movement, two attacking modes and 200 health points.
* The Gaming Flow Chart:
<img src="../Digital Laboratory/images/flow chart.jpg" width="600">

### Project Principles
* VGA screen output
  * Implemented a 1024*768@60Hz VGA signal output.
* MATLAB and block memory
  * Convert images through MATLAB into coe files and store them in memory by IP Catalog.

### Key design blocks
#### Structure of circuit:
<img src="../Digital Laboratory/images/code structure.png" width="800">

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
