from config.config import Cfg
from config.go1_config import config_go1
import numpy as np
import rospy
from sensor_msgs.msg import JointState
from manipulator.loco_manipulator import LocoManipulator


class ManipulatorRunner:
    def __init__(self, cfg: Cfg):
        rospy.init_node('manipulator_ros')
        self.loco_manipualtor = LocoManipulator(cfg)

        self.cur_state_sim_pub = rospy.Publisher(cfg.manipulator.cur_state_sim_topic, JointState, queue_size=100)
        self.cur_state_sim_msg = JointState()
        self.cur_state_sim_msg.name = cfg.manipulator.motor_names

        self.des_pos_sim_sub = rospy.Subscriber(cfg.manipulator.des_pos_sim_topic, JointState, self.des_pos_sim_callback)
        self.getting_state = False
        print('------- Ready to control the manipulators -------')

    def des_pos_sim_callback(self, joint_msg: JointState):
        if self.getting_state:
            return

        self.loco_manipualtor.move_to_target_pos(np.array(joint_msg.position))
        if joint_msg.name[0] == 'update_state':
            self.getting_state = True
            self.publish_cur_state()
            self.getting_state = False

    def publish_cur_state(self):
        self.loco_manipualtor.update_manipulator_sim_state()
        self.cur_state_sim_msg.header.stamp = rospy.Time.now()
        self.cur_state_sim_msg.position = list(self.loco_manipualtor.cur_pos_vel_sim[0])
        self.cur_state_sim_msg.velocity = list(self.loco_manipualtor.cur_pos_vel_sim[1])
        self.cur_state_sim_pub.publish(self.cur_state_sim_msg)

    def run(self):
        rospy.spin()


def main():
    Cfg.sim.use_real_robot = True
    Cfg.update_parms()
    config_go1(Cfg)
    manipulator_runner = ManipulatorRunner(Cfg)
    manipulator_runner.run()


if __name__ == "__main__":
    main()


