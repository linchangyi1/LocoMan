<!-- <h1 align="center">
  LocoMan
</h1> -->
<h2 align="center">
  LocoMan: Advancing Versatile Quadrupedal Dexterity with <br/>Lightweight Loco-Manipulators
</h2>

<div align="center">
  <a href="https://linchangyi1.github.io/"><strong>Changyi Lin</strong></a>,
  <a href="https://xingyul.github.io/"><strong>Xingyu Liu</strong></a>,
  <a href="https://yxyang.github.io/"><strong>Yuxiang Yang</strong></a>,
  <a href="https://yaruniu.com/"><strong>Yaru Niu</strong></a>,
  <br/>
  <a href="https://wenhaoyu.weebly.com/"><strong>Wenhao Yu</strong></a>,
  <a href="https://research.google/people/tingnan-zhang/"><strong>Tingnan Zhang</strong></a>,
  <a href="https://www.jie-tan.net/"><strong>Jie Tan</strong></a>,
  <a href="https://homes.cs.washington.edu/~bboots/"><strong>Byron Boots</strong></a>,
  <a href="https://safeai-lab.github.io/people.html"><strong>Ding Zhao</strong></a>
  <br/>
</div>

<p align="center">
    <a href="https://linchangyi1.github.io/LocoMan/"><em>Website</em></a> |
    <a href="https://arxiv.org/abs/2403.18197"><em>Paper</em></a>
</p>



<p align="center">
<img src="source/design.png" alt="drawing" width=50%/>
<br/>
<br/>
<img src="source/pipeline.png" alt="drawing" width=88%/>
</p>

<!-- --- -->
<!-- <br/> -->

## Table of contents
1. [Overview](#overview)
2. [Installation](#installation)
3. [Usage](#usage)




## Overview <a name="overview"></a>
In this repository, we provide the open-source files for [LocoMan](https://linchangyi1.github.io/LocoMan/), including hardware **design** and **fabrication** files, as well as the code for both **simulation** and **real robot**.
If you don't have a real robot, you can still play it in simulation.
If you have a Unitree GO1 robot without loco-manipulators, you can use it for locomotion and foot-based manipulaiton.
To achieve full functions on the real robot, please build a pair of loco-manipulators and



## Installation <a name="installation"></a>
#### Basic Requirements:
1. Create a conda environment with python3.8:
   ```bash
   conda create -n locoman python=3.8
   ```
2. Install the dependencies:
   ```bash
   conda activate locoman
   pip install -e .
   conda install pinocchio -c conda-forge
   ```
   Note that the `numpy` version should be no later than `1.19.5` to avoid conflict with the Isaac Gym utility files. But we can modify 'np.float' into 'np.float32' in the function 'get_axis_params' of the python file in 'isaacgym/python/isaacgym/torch_utils.py' to resolve the issue.
3. Install [ROS Neotic](https://wiki.ros.org/noetic/Installation/Ubuntu) on Ubuntu 20.04.



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

#### Configure the manipulators (only required for real robots with manipulators):
1. Use [Dynamixel Wizard](https://emanual.robotis.com/docs/en/software/dynamixel/dynamixel_wizard2/) to modify the ID, baud rate, and latency, with reference to the [guide](https://github.com/ROBOTIS-GIT/DynamixelSDK/issues/316):
   - Relabel the ID of the motors([1, 2, 3, 4] for the right manipulator and [5, 6, 7, 8] for the left one  );
   - Modify the Baud Rate to be 1000000
   - Set the return delay time to be 0

2. I also modify the installed sdk with LATENCY_TIMER = 1 in the file /dynamixel_sdk/port_handler.py.

3. Check the USBid and enable the device(modify the USBid based on the result of the first command):
   ```bash
   lsusb
   sudo chmod 777 /dev/ttyUSB0
   ```


## Usage <a name="usage"></a>
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


## Acknowledgements
This repository is developed with inspiration from these repositories: [CAJun](https://github.com/yxyang/cajun), [LEAP Hand](https://github.com/leap-hand/LEAP_Hand_API), and [Cheetah-Software](https://github.com/mit-biomimetics/Cheetah-Software/tree/master). We thank the authors for making the repos open source.


<!-- REMEMBER to use the usage template from the multi-agent repo: https://github.com/ziyanx02/multiagent-quadruped-environment -->











