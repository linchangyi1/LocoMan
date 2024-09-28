# LocoMan

## Table of contents
1. [Overview](#overview)
2. [Installation](#installation)
3. [Play LocoMan](#play)




## Overview <a name="overview"></a>
**This repository provides open-source files of the paper:**

<!-- ![](source/pipelie.png) -->

<b>LocoMan: Advancing Versatile Quadrupedal Dexterity with Lightweight Loco-Manipulators</b> <br>
[Changyi Lin](https://linchangyi1.github.io/),
[Xingyu Liu](https://xingyul.github.io/),
[Yuxiang Yang](https://yxyang.github.io/),
[Yaru Niu](https://yaruniu.com/),
[Wenhao Yu](https://wenhaoyu.weebly.com/),
[Tingnan Zhang](https://research.google/people/tingnan-zhang/),
[Jie Tan](https://www.jie-tan.net/),
[Byron Boots](https://homes.cs.washington.edu/~bboots/), and
[Ding Zhao](https://safeai-lab.github.io/people.html) <br>
*IROS 2024* <br>
[Website](https://linchangyi1.github.io/LocoMan/) /
[Paper](https://arxiv.org/abs/2403.18197)


## Installation <a name="installation"></a>
#### Create a conda environment:
   ```bash
   conda create -n locoman python=3.8
   ```
#### Install the dependencies:
   ```bash
   conda activate locoman
   pip install -e .
   conda install pinocchio -c conda-forge
   ```
Note that the `numpy` version should be no later than `1.19.5` to avoid conflict with the Isaac Gym utility files. But we can modify 'np.float' into 'np.float32' in the function 'get_axis_params' of the python file in 'isaacgym/python/isaacgym/torch_utils.py' to resolve the issue.

#### Install IsaacGym Preview 4 (only required for simulator):
1. Download [IsaacGym](https://developer.nvidia.com/isaac-gym).
2. Install IsaacGym for the locoman environment:
   ```bash
   cd isaacgym/python && pip install -e .
   ```
3. Try running an example `cd examples && python 1080_balls_of_solitude.py`. The code is set to run on CPU so don't worry if you see an error about GPU not being utilized.

#### Install Go1 SDK (only required for real robots):
1. Download the SDK:
   ```bash
   cd locoman
   git clone https://github.com/unitreerobotics/unitree_legged_sdk.git
   ```
2. Make sure the required packages are installed, following Unitree's [guide](https://github.com/unitreerobotics/unitree_legged_sdk). Most nostably, please make sure to install `Boost` and `LCM`:
   ```bash
   sudo apt install libboost-all-dev liblcm-dev
   pip install empy catkin_pkg
   ```
3. Then, go to the `unitree_legged_sdk` directory and build the libraries:
   ```bash
   cd unitree_legged_sdk
   mkdir build && cd build
   cmake -DPYTHON_BUILD=TRUE ..
   make
   ```

#### Configurate the manipulators (only required for real robots with manipulators):
1. Use [Dynamixel Wizard](https://emanual.robotis.com/docs/en/software/dynamixel/dynamixel_wizard2/) to modify the ID, baud rate, and latency, with reference to the [guide](https://github.com/ROBOTIS-GIT/DynamixelSDK/issues/316):
   - Relabel the ID of the motors;
   - Modify the Baud Rate to be 1000000
   - Set the return delay time to be 0

2. I also modify the installed sdk with LATENCY_TIMER = 1 in the file /dynamixel_sdk/port_handler.py.

3. Check the USBid and enable the device(modify the USBid based on the result of the first command):
   ```bash
   lsusb
   sudo chmod 777 /dev/ttyUSB0
   ```


## Play LocoMan <a name="play"></a>
1. Run ROS:
   ```bash
   roscore
   ```

2. Run Joystick:
   ```bash
   python user_commander/joystick.py
   ```

3. Play FSM in Simulation:
   ```bash
   python script/play_fsm.py  # for simulation with manipulators
   python script/play_fsm.py --use_gripper=False  # for simulation without manipulators
   ```

4. Play FSM with a Real Robot:
   If your robot doesn't have manipulators, directly run:
   ```bash
   python script/play_fsm.py --use_real_robot=True --use_gripper=False
   ```

   If your robot has manipulators, firstly run:
   ```bash
   python manipulator/run_manipulators.py
   ```
   and then:
   ```bash
   python script/play_fsm.py --use_real_robot=True
   ```


<!-- REMEMBER to use the usage template from the multi-agent repo: https://github.com/ziyanx02/multiagent-quadruped-environment -->











