# Logi education
#### 2020 Mei-Chu Hackathon - 317 
Won the Third Prize in Team Logitech
## Introduction
### logitype
* Choose your text file and begin the test, the corresponding key on the keyboard will respectively light up with different color.
* While the typing is wrong ,the key will turn red to warn the user. Only if the answer benn correct then the test will keep going.
* After the test completed,it will show your accuracy and typing speed .
### Windows splitter
* Since windows' built-in split windows funcion can only divid screen into half and half, we develop a plugin to contral scale ratio.
## How To Start
### logitype
#### 1. Enviroment
* Logitech G Hub
* Logitech perKey RGB keyboard
#### 2. Running
* unzipping `\Logi_education-master\logitype.zip`

* run `\Logi_education-master\logitype\bin\x86\Debug\netcoreapp3.1\LGT.exe`

### Windows splitter
#### 1. Enviroment
* Install PyAutoGUI from command line:

  `pip install PyAutoGUI`

* Following the 1-2.2 step to install Logi Option
https://github.com/Logitech/logi_craft_sdk/tree/master/samples/CraftPython
#### 2. Setup
* Replace `\logi_craft_sdk-master\samples\CraftPython\Craft.py` 

  with `\Logi_education-master\Logi_education-master\src\Craft\Craft.py`
#### 3. Running
* Following the 2.3-3 step to Enable Windows splitter
https://github.com/Logitech/logi_craft_sdk/tree/master/samples/CraftPython

## Demo videos
#### 1. [logitype](/Hsinchu%20Meichu%20Hackathon%20Team%20Logitech/demo/LED)
#### 2. [Windows splitter](/Hsinchu%20Meichu%20Hackathon%20Team%20Logitech/demo/Craft)
