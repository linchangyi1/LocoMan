"""
This script will read the keyboard inputs and convert them to the corresponding commands for LocoMan.
It's recommended to read the description in the joystick.py script for understanding the command mapping.
"""

#!/usr/bin/env python
from __future__ import print_function
import threading
import sys
from select import select
import termios
import tty
import rospy
from std_msgs.msg import Float32MultiArray, Int32
import numpy as np
from config.config import Cfg
import time


# FSM Command
FSMBindings = {
    '0': 0,
    '1': 1,
    '2': 2,
    '3': 3,
    '4': 4,
    '5': 5,
    '6': 6,
    '7': 7,
}

# Pose Command
PoseBindings = {
    'e': np.array([1, 0, 0, 0, 0, 0]),
    'd': np.array([-1, 0, 0, 0, 0, 0]),
    's': np.array([0, 1, 0, 0, 0, 0]),
    'f': np.array([0, -1, 0, 0, 0, 0]),
    't': np.array([0, 0, 1, 0, 0, 0]),
    'g': np.array([0, 0, -1, 0, 0, 0]),
    'l': np.array([0, 0, 0, 1, 0, 0]),
    'j': np.array([0, 0, 0, -1, 0, 0]),
    'i': np.array([0, 0, 0, 0, 1, 0]),
    'k': np.array([0, 0, 0, 0, -1, 0]),
    'y': np.array([0, 0, 0, 0, 0, 1]),
    'h': np.array([0, 0, 0, 0, 0, -1]),
    }

# Gripper Command
GripperBindings = {
    'v': 1,
    'n': -1,
}


class PublishThread(threading.Thread):
    def __init__(self):
        super(PublishThread, self).__init__()

        self.fsm_command_publisher = rospy.Publisher(Cfg.fsm.fsm_command_topic, Int32, queue_size = 1)
        self.fsm_command_msg = Int32()
        self.fsm_command_msg.data = 0
        self.fsm_command_mapping = Cfg.fsm.fsm_command_mapping

        self.command_publisher = rospy.Publisher(Cfg.teleoperation.joystick.command_topic, Float32MultiArray, queue_size = 1)
        self.command = np.zeros(14)
        self.command_msg = Float32MultiArray()
        self.command_msg.data = self.command.tolist()

        self.last_command_switch = 0
        self.command_inpute_for_torso = True
        self.last_fsm_updated_time = time.time()
        self.min_switch_interval = Cfg.switcher.min_switch_interval

        self.condition = threading.Condition()
        self.done = False
        self.rate = rospy.Rate(30)
        self.start()

    def stop(self):
        self.done = True
        self.join()

    def run(self):
        while not self.done:
            self.condition.acquire()
            self.command[:] = 0
            key = getKey(settings, timeout=0.5)

            # detect FSM command
            if key in FSMBindings.keys():
                new_fsm = FSMBindings[key]
                if self.fsm_command_msg.data != new_fsm and time.time() - self.last_fsm_updated_time > self.min_switch_interval:
                    if self.fsm_command_msg.data != self.fsm_command_mapping['stance']:
                        self.fsm_command_msg.data = self.fsm_command_mapping['stance']
                    else:
                        self.fsm_command_msg.data = new_fsm
                    self.command_inpute_for_torso = True if self.fsm_command_msg.data in (self.fsm_command_mapping['stance'], self.fsm_command_mapping['locomotion']) else False
                    self.last_fsm_updated_time = time.time()
                    self.fsm_command_publisher.publish(self.fsm_command_msg)
                    print('fsm_command: ', self.fsm_command_msg.data)
            # "enter" key for switching between body and eef commands
            elif key == '\r':
                self.command_inpute_for_torso = not self.command_inpute_for_torso
            # construct the pose and gripper commands
            elif key in PoseBindings.keys() or key in GripperBindings.keys():
                self.command[:] = 0
                if key in PoseBindings.keys():
                    cmd_start_idx = 0 if self.command_inpute_for_torso else 6
                    self.command[cmd_start_idx:cmd_start_idx+6] = PoseBindings[key]
                elif key in GripperBindings.keys():
                    self.command[-1] = GripperBindings[key]
                self.command_msg.data = self.command.tolist()
                self.command_publisher.publish(self.command_msg)            
            # "q" key for quitting
            elif key == '\x03':
                self.done = True
            self.condition.notify()
            self.condition.release()
            self.rate.sleep()

def getKey(settings, timeout):
    tty.setraw(sys.stdin.fileno())
    rlist, _, _ = select([sys.stdin], [], [], timeout)
    if rlist:
        key = sys.stdin.read(1)
    else:
        key = ''
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
    return key


if __name__=="__main__":
    settings = termios.tcgetattr(sys.stdin)
    rospy.init_node('keyboard_teleoperation')

    pub_thread = PublishThread()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        pass
    finally:
        pub_thread.stop()
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)






