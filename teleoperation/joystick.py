"""
This script will read the joystick inputs and convert them to the corresponding commands for LocoMan.
It supports two types of joystick: DualSense (e.g. Sony PS5) and Xbox 360 (e.g. GameSir T4 Pro and EasySMX X10).

The commands for controling LocoMan include three parts: the FSM command, the pose command, and the gripper command.
The joystick inputs are mapped to the following commands:
(1) FSM Command:
    - Action Buttons(the four buttons on the right side of the controller):
        - Down(A/Cross Simbol): stance
        - Up(Y/Triangle Simbol): locomotion
        - Left(X/Square Simbol): left gripper manipulation
        - Right(B/Circle Simbol): right gripper manipulation
    - Directional Pad(the four buttons on the left side of the controller):
        - Down: bi-manipulation
        - Up: loco-manipulation
        - Left: left foot manipulation
        - Right: right foot manipulation
(2) Pose Command(xyzrpy):
    - Left Stick Up/Down: x+/x-
    - Left Stick Left/Right: y+/y-
    - L2/R2: z-/z+
    - L1/R1: roll-/roll+
    - Right Stick Up/Down: pitch-/pitch+
    - Right Stick Left/Right: yaw+/yaw-
(3) Gripper Command:
    - L3: open the gripper
    - R3: close the gripper
(4) Command Target Switching(PS button or the center button):
    For the single eef manipulation modes, the target is either the end-effector(gripper or foot) or the torso.
    For the bi-manipulation mode, the target either the left eef or the right eef.
    For the loco-manipulation mode, the target is either the torso or the left gripper or the right gripper.

The script will publish the FSM command and the action command to the corresponding topics.
The FSM command is published as an Int32.
The action command is published as a Float32MultiArray with the following order:
[x_1, y_1, z_1, roll_1, pitch_1, yaw_1, x_2, y_2, z_2, roll_2, pitch_2, yaw_2, gripper_signals]


We define the following input orders for the joysticks:
(1) Buttons: [action_down, action_up, action_left, action_right, l1, r1, l3, r3, ps_button]
(2) Hats: [dpad_down, dpad_up, dpad_left, dpad_right]
(3) Axes: [left_up_down, left_left_right, right_up_down, right_left_right, l2, r2]
The ids of the buttons, hats, and axes are different for different joysticks, which are specified in the config file.
If you want to use a new joystick, you need to identify the ids of the buttons, hats, and axes by running the teleoperation/joystick_calibrate.py script.

The mapping from fsm_command to operation mode:
0: stance, 1: locomotion, 2: left gripper manipulation, 3: right gripper manipulation
4: bi-manipulation, 5: loco-manipulation, 6: left foot manipulation, 7: right foot manipulation

"""


import rospy
from std_msgs.msg import Float32MultiArray, Int32
import pygame
import numpy as np
from config.config import Cfg
import time
import os


class JoystickTeleoperator():
    def __init__(self):
        rospy.init_node('joystick_teleoperation')
        self.rate = rospy.Rate(30)

        os.environ["SDL_VIDEODRIVER"] = "dummy"
        pygame.display.init()
        pygame.joystick.init()
        detected_joysticks = [pygame.joystick.Joystick(x) for x in range(pygame.joystick.get_count())]
        if len(detected_joysticks) == 0:
            print('No joystick is detected!')
            exit()

        candidate_joysticks = Cfg.teleoperation.joystick.feature_names
        joystick_initializd = False
        for joystick in detected_joysticks:
            for i, candidate_joystick in enumerate(candidate_joysticks):
                if candidate_joystick in joystick.get_name():
                    print('The {} joystick is detected.'.format(joystick.get_name()))
                    self.joystick = joystick
                    self.joystick.init()
                    self.button_ids = Cfg.teleoperation.joystick.button_ids[i]
                    self.axis_ids = Cfg.teleoperation.joystick.axis_ids[i]
                    self.hat_values = Cfg.teleoperation.joystick.hat_values[i]
                    joystick_initializd = True
                    break
            if joystick_initializd:
                break
        if not joystick_initializd:
            print('The detected joystick is not in the joystick candidate list! Please update the new joystick candidate in the config file!')
            exit()

        self.fsm_command_publisher = rospy.Publisher(Cfg.fsm.fsm_command_topic, Int32, queue_size=1)
        self.fsm_command_msg = Int32()
        self.fsm_command_msg.data = 0
        self.fsm_command_mapping = Cfg.fsm.fsm_command_mapping

        self.command_publisher = rospy.Publisher(Cfg.teleoperation.joystick.command_topic, Float32MultiArray, queue_size=1)
        self.command = np.zeros(14)
        self.command_msg = Float32MultiArray()
        self.command_msg.data = self.command.tolist()

        self.last_command_switch = 0
        self.command_inpute_for_torso = True
        self.last_fsm_updated_time = time.time()
        self.min_switch_interval = Cfg.switcher.min_switch_interval

    def run(self):
        while not rospy.is_shutdown():
            self.construct_commands()
            self.rate.sleep()

    def construct_commands(self):
        pygame.event.pump()

        # Detect FSM command
        fsm_buttons = [self.joystick.get_button(self.button_ids[i]) for i in range(4)] + [self.joystick.get_hat(0) == self.hat_values[i] for i in range(len(self.hat_values))]
        fsm_buttons = list(map(bool, fsm_buttons))
        if True in fsm_buttons:
            new_fsm = fsm_buttons.index(True)
            if self.fsm_command_msg.data != new_fsm and time.time() - self.last_fsm_updated_time > self.min_switch_interval:
                if self.fsm_command_msg.data != self.fsm_command_mapping['stance']:
                    self.fsm_command_msg.data = self.fsm_command_mapping['stance']
                else:
                    self.fsm_command_msg.data = new_fsm
                self.command_inpute_for_torso = True if self.fsm_command_msg.data in (self.fsm_command_mapping['stance'], self.fsm_command_mapping['locomotion'], self.fsm_command_mapping['loco_mani']) else False
                self.last_fsm_updated_time = time.time()
                self.fsm_command_publisher.publish(self.fsm_command_msg)
                print('fsm_command: ', self.fsm_command_msg.data)

        # Detect the command switch signal
        command_switch_signal = self.joystick.get_button(self.button_ids[8])
        if command_switch_signal == 1 and self.last_command_switch != 1:
            self.command_inpute_for_torso = not self.command_inpute_for_torso
            self.last_command_switch = 1
        else:
            self.last_command_switch = command_switch_signal

        # Construct the pose and gripper commands
        x = -self.joystick.get_axis(self.axis_ids[0])
        y = -self.joystick.get_axis(self.axis_ids[1])
        z = (-self.joystick.get_axis(self.axis_ids[4])+self.joystick.get_axis(self.axis_ids[5])) / 2.0
        roll = - self.joystick.get_button(self.button_ids[4]) + self.joystick.get_button(self.button_ids[5])
        pitch = self.joystick.get_axis(self.axis_ids[2])
        yaw = -self.joystick.get_axis(self.axis_ids[3])
        gripper_signal = self.joystick.get_button(self.button_ids[6]) - self.joystick.get_button(self.button_ids[7])
        self.command[:] = 0
        cmd_start_idx = 0 if self.command_inpute_for_torso else 6
        self.command[cmd_start_idx:cmd_start_idx+6] = np.array([x, y, z, roll, pitch, yaw])
        if self.fsm_command_msg.data == self.fsm_command_mapping['bi_mani']:
            grp_idx = 0 if self.command_inpute_for_torso else 1
            self.command[12+grp_idx] = gripper_signal
        else:
            self.command[12:14] = gripper_signal
        self.command[abs(self.command) < 0.1] = 0
        self.command_msg.data = self.command.tolist()
        self.command_publisher.publish(self.command_msg)

if __name__ == '__main__':
    joystick_controller = JoystickTeleoperator()
    try:
        joystick_controller.run()
    except rospy.ROSInterruptException:
        pass
    finally:
        pygame.quit()
