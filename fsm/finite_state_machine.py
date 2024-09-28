"""
FSM_Command: the command that the FSM receives from the joystick or keyboard
FSM_State: the state of the FSM (include stance and five operation modes described in the paper)
SingleManip_Mode: the mode for single-arm manipulation (left foot, right foot, left eef, right eef)
FSM_Mapping: the mapping from FSM_Command to FSM_State and leg index
FSM_Situation: the situation of the FSM (normal or transition)
"""

import enum

class FSM_Command(enum.Enum):
    STANCE = 0
    LOCOMOTION = 1
    MANIPULATION_LEFT_GRIPPER = 2
    MANIPULATION_RIGHT_GRIPPER = 3
    BIMANIPULATION = 4
    LOCOMANIPULATION = 5
    MANIPULATION_LEFT_FOOT = 6
    MANIPULATION_RIGHT_FOOT = 7

class FSM_State(enum.Enum):
    STANCE = 0
    LOCOMOTION = 1
    SF_MANIPULATION = 2
    SG_MANIPULATION = 3
    BIMANIPULATION = 4
    LOCOMANIPULATION = 5

class SingleLegIndex(enum.Enum):
    RIGHT = 0  # compatible with Go1 leg index
    LEFT = 1

class FSM_Situation(enum.Enum):
    NORMAL = 1
    TRANSITION = 2

Single_Manipulation_Modes = [FSM_State.SF_MANIPULATION, FSM_State.SG_MANIPULATION]
Manipulation_Modes = [FSM_State.SF_MANIPULATION, FSM_State.SG_MANIPULATION, FSM_State.BIMANIPULATION]
Locomotion_Modes = [FSM_State.LOCOMOTION, FSM_State.LOCOMANIPULATION]
Gripper_Manipulation_Modes = [FSM_State.SG_MANIPULATION, FSM_State.BIMANIPULATION]

class FSM_Mapping(enum.Enum):
    STANCE = (FSM_State.STANCE, None)
    LOCOMOTION = (FSM_State.LOCOMOTION, None)
    MANIPULATION_LEFT_GRIPPER = (FSM_State.SG_MANIPULATION, SingleLegIndex.LEFT)
    MANIPULATION_RIGHT_GRIPPER = (FSM_State.SG_MANIPULATION, SingleLegIndex.RIGHT)
    BIMANIPULATION = (FSM_State.BIMANIPULATION, None)
    LOCOMANIPULATION = (FSM_State.LOCOMANIPULATION, None)
    MANIPULATION_LEFT_FOOT = (FSM_State.SF_MANIPULATION, SingleLegIndex.LEFT)
    MANIPULATION_RIGHT_FOOT = (FSM_State.SF_MANIPULATION, SingleLegIndex.RIGHT)

def fsm_command_to_fsm_state_and_leg_index(command, with_gripper=True):
    (fsm_state, leg_index) = FSM_Mapping[FSM_Command(command).name].value
    if not with_gripper and fsm_state == FSM_State.SG_MANIPULATION:
        fsm_state = FSM_State.SF_MANIPULATION
    return fsm_state, leg_index

