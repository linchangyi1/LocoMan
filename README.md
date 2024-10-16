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

## Table of Contents
1. [Overview](#overview)
2. [Basic Installation](#basic_installation)
3. [Running in Simulation](#simulation)
   1. [Isaac Gym Installation](#install_simulator)
   2. [Running LocoMan in Simulation](#fsm_sim)
4. [Real Robot Deployment](#deployment)
   1. [Hardware Setup](#hardware)
   2. [Go1 SDK Installation](#install_sdk)
   2. [Running LocoMan in Real World](#fsm_real)
5. [Notes for Future Development](#notes)


## Overview <a name="overview"></a>
This repository provides the open-source files for [LocoMan](https://linchangyi1.github.io/LocoMan/), including hardware design and fabrication files, as well as the code for both simulation and real robot deployment.


## Basic Installation <a name="basic_installation"></a>
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
   Note that the `numpy` version should be no later than `1.19.5` to avoid conflict with the Isaac Gym utility files. But we can modify 'np.float' into 'np.float32' in the function 'get_axis_params' of the python file in 'isaacgym/python/isaacgym/torch_utils.py' to resolve the issue. So don't worry about the version limitation.
3. Install [ROS Neotic](https://wiki.ros.org/noetic/Installation/Ubuntu) (we only test the code on Ubuntu 20.04).


## Running in Simulation <a name="simulation"></a>
### Isaac Gym Installation <a name="install_simulator"></a>
1. Download [IsaacGym Preview 4](https://developer.nvidia.com/isaac-gym).
2. Install IsaacGym for the locoman environment:
   ```bash
   conda activate locoman
   cd isaacgym/python && pip install -e .
   ```
3. Try running an example `cd examples && python 1080_balls_of_solitude.py`. The code is set to run on CPU so don't worry if you see an error about GPU not being utilized.

### Running LocoMan in Simulation <a name="fsm_sim"></a>
1. Run ROS:
   ```bash
   roscore
   ```
2. Run Joystick (it's recommended to read the comments in [joystick.py](/teleoperation/joystick.py) for better understanding of the teleoperation process):
   ```bash
   python teleoperation/joystick.py
   ```
3. Running LocoMan in Simulation:
   - By default, the robot is equiped with two manipualtors. Run:
      ```bash
      python script/play_fsm.py
      ```
   - For playing a pure Go1 robot without manipulators, run:
      ```bash
      python script/play_fsm.py --use_gripper=False
      ```


## Real Robot Deployment <a name="deployment"></a>
If you have a Unitree GO1 robot without loco-manipulators, you can use it for locomotion and foot-based manipulaiton.
To achieve full functions on the real robot, please build a pair of loco-manipulators.

### Material Preparation

| ID | Part                                      | Description                              | Price (per unit) | Quantity |
|-----|-------------------------------------------|-----------------------------------|--------------|----------|
| 1-8 | [DYNAMIXEL XC330-T288-T](https://www.robotis.us/dynamixel-xc330-t288-t/) | Servos for the manipulators | $89.90 | 8 |
| 9 | [DYNAMIXEL U2D2](https://www.robotis.us/u2d2/) | Convert control signals from PC to servos | $32.10 | 1 |
| 10| [3P JST Expansion Board](https://www.robotis.us/3p-jst-expansion-board/) | Combine control signal and power to the manipulators | $5.90 | 1 |
| 11| [Bearings 5x8x2.5mm (10pcs)](https://a.co/d/0Kc5usm) | Bearings for the rotational gripper | $8.19 | 1 |
| 12| [24V to 12V 10A Converter](https://a.co/d/bhacse1) | Convert 24V power from Go1 to 12V for the manipulators | $9.75 | 1 |
| 13| [30cm XT30 Connector (2pcs)](https://a.co/d/2ftzIKc) | Connect Go1 power supply to the converter | $8.59 | 1 |
| 14 | [100cm 3 Pin JST Cable (5pcs)](https://a.co/d/86x52YO) | Connect manipulators to the expansion board | $10.19 | 1 |
| 15 | [20ft USB Extension Cable](https://a.co/d/3ieBPJI) | Connect PC to U2D2 | $18.99 | 1 |
| 16 | [20ft Ethernet Cable](https://a.co/d/bZTsqN4) | Connect PC to Go1 | $18.99 | 1 |



### Hardware Setup <a name="hardware"></a>


Build two loco-manipulators according to our [Hardware Guide](https://docs.google.com/document/d/1ezbiHfkx8hN_0iZrRt7Yo31gfz1KlZ5sO7tAXYdfni4/edit?usp=sharing).



##### Configure the manipulators:
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


### Go1 SDK Installation <a name="install_sdk"></a>
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

### Running LocoMan in Real World <a name="fsm_real"></a>
1. Similar to running in simulaiton, run ROS and Joystick in separate terminals:
   ```bash
   roscore
   ```
   ```bash
   python teleoperation/joystick.py
   ```

2. Deploy on the Real Robot:
   - **Without manipulators**: Run the following command directly:
      ```bash
      python script/play_fsm.py --use_real_robot=True --use_gripper=False
      ```
   - **With two manipulators**: Before running LocoMan, you need to initialize the manipulators.
      - First, start the manipulators:
         ```bash
         python manipulator/run_manipulators.py
         ```
      - Then, run the FSM:
         ```bash
         python script/play_fsm.py --use_real_robot=True
         ```



## Acknowledgements
This repository is developed with inspiration from these repositories: [CAJun](https://github.com/yxyang/cajun), [LEAP Hand](https://github.com/leap-hand/LEAP_Hand_API), and [Cheetah-Software](https://github.com/mit-biomimetics/Cheetah-Software/tree/master). We thank the authors for making the repos open source.











