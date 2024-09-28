from fsm.finite_state_machine import FSM_State
import numpy as np
import copy


class WBCCommand:
    def __init__(self):
        self.operation_mode = FSM_State.STANCE
        self.contact_state = np.ones(4, dtype=bool)
        self.des_torso_pva = np.zeros((3, 6))
        self.des_foot_pva = np.zeros((3, 4, 3))  # 3 physical dimensions(position-velocity-acceleration), 4 feet, 3D position
        self.des_gripper_pva = np.zeros((3, 2, 6))  # 3 physical dimensions(position-velocity-acceleration), 2 grippers, 6D pose
        self.des_gripper_angles = np.zeros(2)

    def reset(self, keep_mode=False):
        if not keep_mode:
            self.operation_mode = FSM_State.STANCE
        self.contact_state[:] = True
        self.des_torso_pva[:] = 0
        self.des_foot_pva[:] = 0
        self.des_gripper_pva[:] = 0
        self.des_gripper_angles[:] = 0

    def copy(self):
        return copy.deepcopy(self)


