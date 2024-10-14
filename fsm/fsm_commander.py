from robot.base_robot import BaseRobot
from commander.stance_commander import StanceCommander
from commander.sf_manipulation_commander import SingleFootManipCommander
from commander.sg_manipulation_commander import SingleGripperManipCommander
from commander.locomotion_commander import LocomotionCommander
from commander.loco_manipulation_commander import LocoManipCommander
from commander.bi_manipulation_commander import BiManipCommander
from fsm.finite_state_machine import FSM_State


class FSMCommander:
    def __init__(self, runner, env_ids=0):
        self._runner = runner
        self._robot: BaseRobot = runner._robot
        self._env_ids = env_ids
        self._cfg = self._robot._cfg

        self._stance_commander = StanceCommander(self._robot, self._env_ids)
        self._locomotion_commander = LocomotionCommander(self._robot, self._env_ids)
        self._sf_manipulation_commander = SingleFootManipCommander(self._robot, self._env_ids)
        self._sg_manipulation_commander = SingleGripperManipCommander(self._robot, self._env_ids)
        self._bi_manipulation_commander = BiManipCommander(self._robot, self._env_ids)
        self._loco_manipulation_commander = LocoManipCommander(self._robot, self._env_ids)

    def get_robot_commander(self):
        if self._robot._cur_fsm_state == FSM_State.STANCE:
            return self._stance_commander
        elif self._robot._cur_fsm_state == FSM_State.LOCOMOTION:
            return self._locomotion_commander
        elif self._robot._cur_fsm_state == FSM_State.SF_MANIPULATION:
            return self._sf_manipulation_commander
        elif self._robot._cur_fsm_state == FSM_State.SG_MANIPULATION:
            return self._sg_manipulation_commander
        elif self._robot._cur_fsm_state == FSM_State.BIMANIPULATION:
            return self._bi_manipulation_commander
        elif self._robot._cur_fsm_state == FSM_State.LOCOMANIPULATION:
            return self._loco_manipulation_commander


