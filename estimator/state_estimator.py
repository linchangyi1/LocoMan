from estimator.foot_state_estimator import FootStateEstimator
from estimator.bimanual_state_estimator import BiManualStateEstimator
from estimator.kf_state_estimator import KFStateEstimator
from estimator.sim_man_state_estimator import SimManStateEstimator
from estimator.sim_bimanual_state_estimator import SimBiManualStateEstimator
from estimator.sim_loco_state_estimator import SimLocoStateEstimator
from fsm.finite_state_machine import FSM_State


class StateEstimator:
    def __init__(self, robot, use_real_robot=False):
        if use_real_robot:
            self.kinematics_state_estimator = FootStateEstimator(robot)
            self.zero_yaw_state_estimator = KFStateEstimator(robot)
            self.bimanual_state_estimator = BiManualStateEstimator(robot)
        else:
            self.kinematics_state_estimator = SimManStateEstimator(robot)
            self.zero_yaw_state_estimator = SimLocoStateEstimator(robot)
            self.bimanual_state_estimator = SimBiManualStateEstimator(robot)

    def __getitem__(self, key):
        if key in (FSM_State.STANCE, FSM_State.SF_MANIPULATION, FSM_State.SG_MANIPULATION):
            return self.kinematics_state_estimator
        elif key in (FSM_State.LOCOMOTION, FSM_State.LOCOMANIPULATION):
            return self.zero_yaw_state_estimator
        elif key == FSM_State.BIMANIPULATION:
            return self.bimanual_state_estimator
        else:
            raise print("Unknown state: {}".format(key))


