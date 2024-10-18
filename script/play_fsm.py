import argparse
from utilities.argument_utils import str2bool
from config.config import Cfg
from config.go1_config import config_go1
import rospy


def main():
    rospy.init_node('locoman')
    parser = argparse.ArgumentParser(prog="LocoMan")
    parser.add_argument("--use_real_robot", type=bool, default=False, help="whether to use real robot.")
    parser.add_argument("--num_envs", type=int, default=1, help="environment number.")
    parser.add_argument("--use_gpu", type=str2bool, default=True, help="whether to use GPU")
    parser.add_argument("--show_gui", type=str2bool, default=True, help="set as True to show GUI")
    parser.add_argument("--sim_device", default='cuda:0', help="the gpu to use")
    parser.add_argument("--use_gripper", type=str2bool, default=True, help="set as True to use gripper")
    args = parser.parse_args()

    for key, value in vars(args).items():
        setattr(Cfg.sim, key, value)
    Cfg.update_parms()
    config_go1(Cfg)

    from fsm.fsm_runner import FSMRunner
    runner = FSMRunner(Cfg)
    
    while not rospy.is_shutdown():
        runner.step()
    quit()


if __name__ == '__main__':
    main()

