from config.config import Cfg
from typing import List
import torch
from robot.motors import MotorCommand
from fsm.fsm_switcher import FSMSwitcher
from switcher.bi_manipulation_switcher import BimanipulationSwitcher
from fsm.fsm_commander import FSMCommander
from wbc.whole_body_controller import WholeBodyController
from wbc.wbc_command import WBCCommand
import rospy
from std_msgs.msg import Int32
from fsm.finite_state_machine import FSM_State, FSM_Command, FSM_Situation, fsm_command_to_fsm_state_and_leg_index, Manipulation_Modes, Locomotion_Modes
from commander.bi_manipulation_commander import BiManipCommander


"""FSMRunner manages the whole pipeline of the robot control system."""
class FSMRunner:
    def __init__(self, cfg: Cfg = None):
        self._cfg = cfg

        # initialize real robot or sim robot
        if self._cfg.sim.use_real_robot:
            from robot.real_robot import RealRobot
            self._robot = RealRobot(self._cfg)
        else:
            self._sim_conf = self._cfg.get_sim_config()
            self._cfg.sim.sim_device = self._sim_conf.sim_device
            self._sim, self._viewer = self._create_sim()
            from robot.sim_robot import SimRobot
            self._robot = SimRobot(self._cfg, self._sim, self._viewer)

        # planner, controller
        self._fsm_switcher = FSMSwitcher(self, 0)
        self._fsm_commander = FSMCommander(self, 0)
        self._wbc_list: List[WholeBodyController] = []
        for i in range(self._robot._num_envs):
            self._wbc_list.append(WholeBodyController(self._robot, i))
        self._robot_commander = None

        # buffers
        self._fsm_state_buffer = self._robot._cur_fsm_state
        self._single_leg_idx_buffer = self._robot._cur_single_leg_idx
        self._fsm_command_sub = rospy.Subscriber(self._cfg.fsm.fsm_command_topic, Int32, self._fsm_command_callback)
        self._contact_state_torch = torch.ones((self._robot._num_envs, 4), dtype=torch.bool, device=self._robot._device, requires_grad=False)

        # general action
        self._contact_state_idx = torch.zeros((self._robot._num_envs, self._robot._num_joints), dtype=torch.bool, device=self._robot._device, requires_grad=False)
        self._desired_joint_pos = torch.zeros((self._robot._num_envs, self._robot._num_joints), device=self._robot._device, requires_grad=False)
        self._desired_joint_vel = torch.zeros((self._robot._num_envs, self._robot._num_joints), device=self._robot._device, requires_grad=False)
        self._desired_joint_torque = torch.zeros((self._robot._num_envs, self._robot._num_joints), device=self._robot._device, requires_grad=False)
        self._kps = self._robot._motors.kps_stance_mani.clone()
        self._kds = self._robot._motors.kds_stance_mani.clone() 
        self._action = MotorCommand(desired_position=torch.zeros((self._robot._num_envs, self._robot._num_joints), device=self._robot._device),
                        kp=self._kps,
                        desired_velocity=torch.zeros((self._robot._num_envs, self._robot._num_joints), device=self._robot._device),
                        kd=self._kds,
                        desired_extra_torque=torch.zeros((self._robot._num_envs, self._robot._num_joints), device=self._robot._device))

        # action for bimanipulation
        self._lock_bimanual = self._cfg.sim.use_real_robot and self._cfg.commander.lock_real_robot_bimanual
        self._bimanual_action = MotorCommand(desired_position=torch.zeros((self._robot._num_envs, self._robot._num_joints), device=self._robot._device),
                        kp=self._kps,
                        desired_velocity=torch.zeros((self._robot._num_envs, self._robot._num_joints), device=self._robot._device),
                        kd=self._kds,
                        desired_extra_torque=torch.zeros((self._robot._num_envs, self._robot._num_joints), device=self._robot._device))
        if self._cfg.sim.use_real_robot:
            self._kp_bimanual_switch = torch.tensor(self._cfg.motor_control.sim.kp_bimanual_switch, device=self._robot._device, dtype=torch.float, requires_grad=False)
            self._kd_bimanual_switch = torch.tensor(self._cfg.motor_control.sim.kd_bimanual_switch, device=self._robot._device, dtype=torch.float, requires_grad=False)
            self._kp_bimanual_command = torch.tensor(self._cfg.motor_control.sim.kp_bimanual_command, device=self._robot._device, dtype=torch.float, requires_grad=False)
            self._kd_bimanual_command = torch.tensor(self._cfg.motor_control.sim.kd_bimanual_command, device=self._robot._device, dtype=torch.float, requires_grad=False)
        else:
            self._kp_bimanual_switch = torch.tensor(self._cfg.motor_control.real.kp_bimanual_switch, device=self._robot._device, dtype=torch.float, requires_grad=False)
            self._kd_bimanual_switch = torch.tensor(self._cfg.motor_control.real.kd_bimanual_switch, device=self._robot._device, dtype=torch.float, requires_grad=False)
            self._kp_bimanual_command = torch.tensor(self._cfg.motor_control.real.kp_bimanual_command, device=self._robot._device, dtype=torch.float, requires_grad=False)
            self._kd_bimanual_command = torch.tensor(self._cfg.motor_control.real.kd_bimanual_command, device=self._robot._device, dtype=torch.float, requires_grad=False)

        # reset robot and command generator
        self._robot.reset()
        self._robot_commander = self._fsm_commander.get_robot_commander()
        self._robot_commander.activate_commander()
        self._robot_commander.reset()

    def _create_sim(self):
        from isaacgym import gymapi, gymutil
        self._gym = gymapi.acquire_gym()
        _, sim_device_id = gymutil.parse_device_str(self._sim_conf.sim_device)
        if self._cfg.sim.show_gui:
            graphics_device_id = sim_device_id
        else:
            graphics_device_id = -1

        sim = self._gym.create_sim(sim_device_id, graphics_device_id,
                            self._sim_conf.physics_engine, self._sim_conf.sim_params)
        if self._cfg.sim.show_gui:
            viewer = self._gym.create_viewer(sim, gymapi.CameraProperties())
            self._gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_ESCAPE, "QUIT")
            self._gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_V, "toggle_viewer_sync")
        else:
            viewer = None
        self._gym.add_ground(sim, self._sim_conf.plane_params)
        return sim, viewer

    def _fsm_command_callback(self, msg: Int32):
        if self._robot._fsm_situation == FSM_Situation.NORMAL:
            if self._lock_bimanual and FSM_Command(msg.data).name == 'BIMANIPULATION':
                msg.data = FSM_Command.MANIPULATION_RIGHT_FOOT.value
            self._fsm_state_buffer, self._single_leg_idx_buffer = fsm_command_to_fsm_state_and_leg_index(msg.data)
            if not self._robot._use_gripper:
                if self._fsm_state_buffer == FSM_State.SG_MANIPULATION:
                    self._fsm_state_buffer = FSM_State.SF_MANIPULATION
                elif self._fsm_state_buffer == FSM_State.LOCOMANIPULATION:
                    self._fsm_state_buffer = FSM_State.LOCOMOTION

    def step(self):
        # Bi-manipulation is a special case since we use the transition trajectory from Unitree's controller and lock the rear joints during manipulation
        if type(self._robot_commander) in [BiManipCommander, BimanipulationSwitcher]:
            if type(self._robot_commander) is BimanipulationSwitcher:
                desired_q, desired_dq, desired_torque = self._robot_commander.compute_motor_command()
                self._bimanual_action.kp[:, self._robot._dog_joint_idx] = self._kp_bimanual_switch
                self._bimanual_action.kd[:, self._robot._dog_joint_idx] = self._kd_bimanual_switch
                self._bimanual_action.desired_position[:, :] = torch.tensor(desired_q, device=self._robot._device, dtype=torch.float, requires_grad=False)
                self._bimanual_action.desired_velocity[:, :] = torch.tensor(desired_dq, device=self._robot._device, dtype=torch.float, requires_grad=False)
                self._bimanual_action.desired_extra_torque[:, :] = torch.tensor(desired_torque, device=self._robot._device, dtype=torch.float, requires_grad=False)
            else:
                wbc_command: WBCCommand = self._robot_commander.compute_command_for_wbc()
                if self._robot._use_gripper:
                    self._robot.set_gripper_angles(torch.tensor(wbc_command.des_gripper_angles, device=self._robot._device, requires_grad=False))
                for i in range(self._robot._num_envs):
                    command_executed, desired_q, desired_dq, desired_torque = self._wbc_list[i].step(wbc_command)
                    desired_q[-6:] = wbc_command.rear_legs_pos
                    desired_dq[-6:] = wbc_command.rear_legs_vel
                    desired_torque[-6:] = wbc_command.rear_legs_torque
                    self._bimanual_action.kp[i, self._robot._dog_joint_idx] = self._kp_bimanual_command
                    self._bimanual_action.kd[i, self._robot._dog_joint_idx] = self._kd_bimanual_command
                    self._robot_commander.feedback_execution(command_executed)
                    self._bimanual_action.desired_position[i, :] = torch.tensor(desired_q, device=self._robot._device, dtype=torch.float, requires_grad=False)
                    self._bimanual_action.desired_velocity[i, :] = torch.tensor(desired_dq, device=self._robot._device, dtype=torch.float, requires_grad=False)
                    self._bimanual_action.desired_extra_torque[i, :] = torch.tensor(desired_torque, device=self._robot._device, dtype=torch.float, requires_grad=False)
            self._robot.step(self._bimanual_action, gripper_cmd=True)
        else:
            wbc_command: WBCCommand = self._robot_commander.compute_command_for_wbc()
            self._update_contact_state_idx(wbc_command.contact_state)
            if self._robot._use_gripper:
                self._robot.set_gripper_angles(torch.tensor(wbc_command.des_gripper_angles, device=self._robot._device, requires_grad=False))
            for i in range(self._robot._num_envs):
                command_executed, desired_q, desired_dq, desired_torque = self._wbc_list[i].step(wbc_command)
                self._desired_joint_pos[i] = torch.tensor(desired_q, device=self._robot._device, requires_grad=False)
                self._desired_joint_vel[i] = torch.tensor(desired_dq, device=self._robot._device, requires_grad=False)
                self._desired_joint_torque[i] = torch.tensor(desired_torque, device=self._robot._device, requires_grad=False)
            self._robot_commander.feedback_execution(command_executed)
            self._construct_and_apply_action()
        self._check_switching()


    def _update_contact_state_idx(self, contact_state):
        self._contact_state_torch[:] = torch.tensor(contact_state, dtype=torch.bool, device=self._robot._device, requires_grad=False)
        self._robot.set_desired_foot_contact(self._contact_state_torch)
        self._contact_state_idx[:, :6] = self._contact_state_torch[:, 0:1].repeat(1, 6)
        self._contact_state_idx[:, 6:12] = self._contact_state_torch[:, 1:2].repeat(1, 6)
        if self._robot._use_gripper:
            self._contact_state_idx[:, 12:15] = self._contact_state_torch[:, 2:3].repeat(1, 3)
            self._contact_state_idx[:, 15:18] = self._contact_state_torch[:, 3:].repeat(1, 3)

    def _construct_and_apply_action(self):
        # update the kp and kd based on whether the foot is in contact
        self._kps[self._contact_state_idx] = self._robot._motors.kps_stance_loco[self._contact_state_idx] if self._robot._cur_fsm_state in Locomotion_Modes else self._robot._motors.kps_stance_mani[self._contact_state_idx]
        self._kps[~self._contact_state_idx] = self._robot._motors.kps_swing_loco[~self._contact_state_idx] if self._robot._cur_fsm_state in Locomotion_Modes else self._robot._motors.kps_swing_mani[~self._contact_state_idx]
        self._kds[self._contact_state_idx] = self._robot._motors.kds_stance_loco[self._contact_state_idx] if self._robot._cur_fsm_state in Locomotion_Modes else self._robot._motors.kds_stance_mani[self._contact_state_idx]
        self._kds[~self._contact_state_idx] = self._robot._motors.kds_swing_loco[~self._contact_state_idx] if self._robot._cur_fsm_state in Locomotion_Modes else self._robot._motors.kds_swing_mani[~self._contact_state_idx]

        # update the action
        self._action.desired_position[:] = self._desired_joint_pos
        self._action.desired_velocity[:] = self._desired_joint_vel
        self._action.desired_extra_torque[:] = self._desired_joint_torque
        self._action.kp[:] = self._kps
        self._action.kd[:] = self._kds

        self._robot.step(self._action, gripper_cmd=True)

    def _check_switching(self):
        if self._robot._fsm_situation == FSM_Situation.TRANSITION:
            if self._robot_commander.check_finished():
                print('-------------Finish transition to {} mode-------------'.format(self._fsm_state_buffer))
                self._switch_to_normal_mode()
        elif self._fsm_state_buffer != self._robot._cur_fsm_state or self._single_leg_idx_buffer != self._robot._cur_single_leg_idx:
            print('-------------In transition to {} mode-------------'.format(self._fsm_state_buffer))
            self._switch_to_transition_mode()

    def _switch_to_normal_mode(self):
        self._robot._cur_fsm_state = self._fsm_state_buffer
        self._robot._cur_single_leg_idx = self._single_leg_idx_buffer
        self._robot_commander = self._fsm_commander.get_robot_commander()
        self._robot_commander.activate_commander()
        self._robot_commander.reset()
        self._robot._fsm_situation = FSM_Situation.NORMAL

    def _switch_to_transition_mode(self):
        self._robot._fsm_situation = FSM_Situation.TRANSITION
        # ensure tha at least one state is stance
        if FSM_State.STANCE in [self._robot._cur_fsm_state, self._fsm_state_buffer]:
            non_stance_state = self._robot._cur_fsm_state if self._fsm_state_buffer == FSM_State.STANCE else self._fsm_state_buffer
            # switch between stance and manipulation, and from stance to loco-manipulation
            if non_stance_state in Manipulation_Modes or self._fsm_state_buffer == FSM_State.LOCOMANIPULATION:
                self._robot_commander.deactivate_commander()
                self._robot_commander = self._fsm_switcher.get_switcher()
                self._robot_commander.reset()
            # from locomotion or loco-manipulation to stance, first slow down and then stand when all feet are in contact
            elif self._robot._cur_fsm_state in Locomotion_Modes:
                self._robot_commander.prepare_to_stand()
        else:
            print('Attention: both states are not stance!')
            self._fsm_state_buffer = FSM_State.STANCE
            self._robot._fsm_situation = FSM_Situation.NORMAL
        self._robot._nex_fsm_state = self._fsm_state_buffer



