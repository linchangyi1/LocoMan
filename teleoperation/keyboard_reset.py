#!/usr/bin/env python
from __future__ import print_function
from config.config import Cfg

import threading
import rospy
from std_msgs.msg import Int32
import sys

if sys.platform == 'win32':
    import msvcrt
else:
    import termios
    import tty

class KeyboardInputThread(threading.Thread):
    def __init__(self, topic_name):
        super(KeyboardInputThread, self).__init__()
        self.publisher = rospy.Publisher(topic_name, Int32, queue_size=1)
        self.done = False
        self.state = 0  # Toggle state between 0 and 1

    def run(self):
        while not self.done and not rospy.is_shutdown():
            if sys.platform == 'win32':
                if msvcrt.kbhit():
                    key = msvcrt.getch()
                    if key == b'\r':  # Enter key
                        self.toggle_state()
            else:
                key = self.get_key()
                if key == '\r':  # Enter key
                    self.toggle_state()
                elif key == '\x03':
                    self.done = True

    def toggle_state(self):
        self.state = 1 if self.state == 0 else 0
        self.publisher.publish(self.state)
        print("Published state: {}".format(self.state))

    def get_key(self):
        if sys.platform != 'win32':
            tty.setraw(sys.stdin.fileno())
            key = sys.stdin.read(1)  # Read one character
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
            return key
        return ''

    def stop(self):
        self.done = True

if __name__ == "__main__":
    if sys.platform != 'win32':
        settings = termios.tcgetattr(sys.stdin)

    rospy.init_node('keyboard_reset_node')
    keyboard_thread = KeyboardInputThread(Cfg.teleoperation.human_teleoperator.receive_action_topic)
    
    try:
        keyboard_thread.start()
        rospy.spin()
    except KeyboardInterrupt:
        pass
    finally:
        keyboard_thread.stop()
        if sys.platform != 'win32':
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
