from fsm.finite_state_machine import FSM_State, Single_Manipulation_Modes
from robot.base_robot import BaseRobot
from switcher.single_manipulation_switcher import SingleManipSwitcher
from switcher.bi_manipulation_switcher import BimanipulationSwitcher
from switcher.loco_manipulation_switcher import LocoManipulationSwitcher


"""FSMSwitcher generates the switcher for different FSM states."""
class FSMSwitcher:
    def __init__(self, runner, env_ids=0):
        self._runner = runner
        self._robot: BaseRobot = runner._robot
        self._env_ids = env_ids
        self._cfg = self._robot._cfg

        self._single_manipulation_switcher = SingleManipSwitcher(self._robot, self._env_ids)
        self._bi_manipulation_switcher = BimanipulationSwitcher(self._robot, self._env_ids)
        if  self._robot._use_gripper:
            self._stance_to_locomanipulation_switcher = LocoManipulationSwitcher(self._robot, self._env_ids)

    def get_switcher(self):
        # from stance to other modes
        if self._robot._cur_fsm_state == FSM_State.STANCE:
            if self._runner._fsm_state_buffer in Single_Manipulation_Modes:
                return self._single_manipulation_switcher.stance_to_single_manipulation(self._runner._fsm_state_buffer, self._runner._single_leg_idx_buffer)
            elif self._runner._fsm_state_buffer == FSM_State.BIMANIPULATION:
                return self._bi_manipulation_switcher.activate_switcher(stand_up=True)
            elif self._runner._fsm_state_buffer == FSM_State.LOCOMANIPULATION:
                return self._stance_to_locomanipulation_switcher
        # from other modes to stance
        elif self._runner._fsm_state_buffer == FSM_State.STANCE:
            if self._robot._cur_fsm_state in Single_Manipulation_Modes:
                return self._single_manipulation_switcher.single_manipulation_to_stance()
            elif self._robot._cur_fsm_state == FSM_State.BIMANIPULATION:
                return self._bi_manipulation_switcher.activate_switcher(stand_up=False)



