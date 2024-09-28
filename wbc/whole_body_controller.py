import numpy as np
import pinocchio as pin
import scipy
from robot.base_robot import BaseRobot
from utilities.orientation_utils_numpy import rot_mat_to_rpy, rpy_to_rot_mat
import quadprog
from fsm.finite_state_machine import FSM_State, FSM_Situation, Manipulation_Modes, Locomotion_Modes, Gripper_Manipulation_Modes
from wbc.wbc_command import WBCCommand


"""
The whole-body controller (WBC) implemented in this script extends the method from https://arxiv.org/abs/1909.06586 without using MPC.
Code contributors: Changyi Lin and Yuxiang Yang
"""
class WholeBodyController:
    def __init__(self, robot: BaseRobot, env_ids):
        self._robot = robot
        self._env_ids = env_ids
        self._use_gripper = self._robot._use_gripper
        self._num_joints = self._robot._num_joints
        self._robot_model = pin.RobotWrapper(pin.buildModelFromUrdf(self._robot._cfg.asset.wbc_urdf_path, pin.JointModelFreeFlyer()))
        self._torso_frame_id = self._robot_model.index('root_joint')
        self._foot_frame_ids = []
        for foot_name in self._robot._cfg.asset.foot_names:
            self._foot_frame_ids.append(self._robot_model.model.getFrameId(foot_name))

        # Gains
        use_real_robot = self._robot._cfg.sim.use_real_robot
        self._torso_position_kp_loco = self._robot._cfg.wbc.real.torso_position_kp_loco if use_real_robot else self._robot._cfg.wbc.sim.torso_position_kp_loco
        self._torso_position_kd_loco = self._robot._cfg.wbc.real.torso_position_kd_loco if use_real_robot else self._robot._cfg.wbc.sim.torso_position_kd_loco
        self._torso_orientation_kp_loco = self._robot._cfg.wbc.real.torso_orientation_kp_loco if use_real_robot else self._robot._cfg.wbc.sim.torso_orientation_kp_loco
        self._torso_orientation_kd_loco = self._robot._cfg.wbc.real.torso_orientation_kd_loco if use_real_robot else self._robot._cfg.wbc.sim.torso_orientation_kd_loco
        self._torso_position_kp_mani = self._robot._cfg.wbc.real.torso_position_kp_mani if use_real_robot else self._robot._cfg.wbc.sim.torso_position_kp_mani
        self._torso_position_kd_mani = self._robot._cfg.wbc.real.torso_position_kd_mani if use_real_robot else self._robot._cfg.wbc.sim.torso_position_kd_mani
        self._torso_orientation_kp_mani = self._robot._cfg.wbc.real.torso_orientation_kp_mani if use_real_robot else self._robot._cfg.wbc.sim.torso_orientation_kp_mani
        self._torso_orientation_kd_mani = self._robot._cfg.wbc.real.torso_orientation_kd_mani if use_real_robot else self._robot._cfg.wbc.sim.torso_orientation_kd_mani
        self._foot_position_kp = self._robot._cfg.wbc.real.foot_position_kp if use_real_robot else self._robot._cfg.wbc.sim.foot_position_kp
        self._foot_position_kd = self._robot._cfg.wbc.real.foot_position_kd if use_real_robot else self._robot._cfg.wbc.sim.foot_position_kd

        # with grippers
        if self._use_gripper:
            self._gripper_frame_ids = []
            for gripper_name in self._robot._cfg.asset.eef_names:
                self._gripper_frame_ids.append(self._robot_model.model.getFrameId(gripper_name))
            from manipulator.manipulator_kinematics import ManipulatorKinematics
            self._manipulator_kinematics = ManipulatorKinematics(self._robot._cfg)
            self._gripper_position_kp = self._robot._cfg.wbc.real.gripper_position_kp if use_real_robot else self._robot._cfg.wbc.sim.gripper_position_kp
            self._gripper_position_kd = self._robot._cfg.wbc.real.gripper_position_kd if use_real_robot else self._robot._cfg.wbc.sim.gripper_position_kd
            self._gripper_orientation_kp = self._robot._cfg.wbc.real.gripper_orientation_kp if use_real_robot else self._robot._cfg.wbc.sim.gripper_orientation_kp
            self._gripper_orientation_kd = self._robot._cfg.wbc.real.gripper_orientation_kd if use_real_robot else self._robot._cfg.wbc.sim.gripper_orientation_kd
            self._tasks = self._robot._cfg.wbc.tracking_objects_w_gripper
            self._task_dims = self._robot._cfg.wbc.tracking_objects_dims_w_gripper
        else:
            self._tasks = self._robot._cfg.wbc.tracking_objects_wo_gripper
            self._task_dims = self._robot._cfg.wbc.tracking_objects_dims_wo_gripper
        self._task_dims = [3] + self._task_dims

        # buffers for WBC
        self._tasks_num = len(self._tasks)
        self._wbc_command = WBCCommand()

        self._e = [np.zeros(self._task_dims[i]) for i in range(self._tasks_num+1)]
        self._dx = [np.zeros(self._task_dims[i]) for i in range(self._tasks_num+1)]
        self._ddx = [np.zeros(self._task_dims[i]) for i in range(self._tasks_num+1)]
        self._q, self._dq = np.zeros(7+self._num_joints), np.zeros(6+self._num_joints)  # robot state
        self._delta_q, self._dq_cmd, self._ddq_cmd, self._torque = np.zeros(self._num_joints+6), np.zeros(self._num_joints+6), np.zeros(self._num_joints+6), np.zeros(self._num_joints+6)
        self._des_q, self._last_des_q, self._last_dq_cmd, self._last_torque = np.zeros(self._num_joints), np.zeros(self._num_joints), np.zeros(self._num_joints), np.zeros(self._num_joints)
        
        self._torso_acc_limits = self._robot._cfg.wbc.torso_acc_limits
        self._joint_acc_limits = np.stack([np.ones(self._num_joints) * self._robot._cfg.wbc.joint_acc_limits[0],
                                           np.ones(self._num_joints) * self._robot._cfg.wbc.joint_acc_limits[1]], axis=0)
        self._W_torso = np.diag( self._robot._cfg.wbc.torso_qp_weight)
        self._W_foot =  self._robot._cfg.wbc.foot_qp_weight
        friction_coef = self._robot._cfg.wbc.friction_coef
        self._friction_constraints_per_leg = np.array([[0., 0., 1.],
                                                 [1, 0, friction_coef],
                                                 [-1, 0, friction_coef],
                                                 [0, 1, friction_coef],
                                                 [0, -1, friction_coef]])
        

        self._N = [np.eye(self._num_joints+6) for _ in range(self._tasks_num+1)]
        self._N_i_im1 = [np.eye(self._num_joints+6) for _ in range(self._tasks_num+1)]
        self._N_dyn = [np.eye(self._num_joints+6) for _ in range(self._tasks_num+1)]
        self._N_i_im1_dyn = [np.eye(self._num_joints+6) for _ in range(self._tasks_num+1)]

        self._J = [np.zeros((self._task_dims[i], self._num_joints+6)) for i in range(self._tasks_num+1)]
        self._J_i_im1 = [np.zeros((self._task_dims[i], self._num_joints+6)) for i in range(self._tasks_num+1)]
        self._J_pre = [np.zeros((self._task_dims[i], self._num_joints+6)) for i in range(self._tasks_num+1)]
        self._J_pre_inv = [np.zeros((self._num_joints+6, self._task_dims[i])) for i in range(self._tasks_num+1)]
        self._J_i_im1_dyn = [np.zeros((self._task_dims[i], self._num_joints+6)) for i in range(self._tasks_num+1)]
        self._J_pre_dyn = [np.zeros((self._task_dims[i], self._num_joints+6)) for i in range(self._tasks_num+1)]
        self._J_pre_dyn_inv = [np.zeros((self._num_joints+6, self._task_dims[i])) for i in range(self._tasks_num+1)]
        self._dJdq = [np.zeros(self._task_dims[i]) for i in range(self._tasks_num+1)]

        # safety
        self._safe_command = False
        self._first_command = True
        self._min_joint_pos = self._robot._motors.min_positions_np.copy()
        self._max_joint_pos = self._robot._motors.max_positions_np.copy()
        self._mani_max_delta_q = self._robot._cfg.wbc.mani_max_delta_q
        self._mani_max_dq = self._robot._cfg.wbc.mani_max_dq
        self._mani_max_ddq = self._robot._cfg.wbc.mani_max_ddq
        self._loco_max_delta_q = self._robot._cfg.wbc.loco_max_delta_q
        self._loco_max_dq = self._robot._cfg.wbc.loco_max_dq
        self._loco_max_ddq = self._robot._cfg.wbc.loco_max_ddq

    def step(self, wbc_command: WBCCommand):
        self.update_robot_model()
        self.update_wbc_command(wbc_command)
        return self.compute_actions()

    def update_robot_model(self):
        self._q = np.concatenate((self._robot.torso_pos_w_np[self._env_ids],
                               self._robot.torso_quat_w2b_np[self._env_ids],
                               self._robot.joint_pos_np[self._env_ids]))

        # the convention of pinocchio: using dq in the local frame !
        self._dq = np.concatenate((self._robot.torso_lin_vel_b_np[self._env_ids],
                                self._robot.torso_ang_vel_b_np[self._env_ids],
                                self._robot.joint_vel_np[self._env_ids]))
        self._robot_model.forwardKinematics(self._q, self._dq, np.zeros_like(self._dq))
        self._robot_model.computeJointJacobians(self._q)
        
    def update_wbc_command(self, wbc_command: WBCCommand):
        self._wbc_command = wbc_command.copy()
        if self._use_gripper:
            joint_space_input_gripper_idx = []
            for i in range(len(self._gripper_frame_ids)):
                if self._wbc_command.contact_state[i] or self._wbc_command.operation_mode not in Gripper_Manipulation_Modes:
                    if self._robot._nex_fsm_state==FSM_State.LOCOMANIPULATION and self._robot._fsm_situation==FSM_Situation.TRANSITION and i==self._robot._cfg.loco_manipulation.manipulate_leg_idx:
                        continue
                    elif self._robot._cur_fsm_state==FSM_State.LOCOMANIPULATION and self._robot._fsm_situation==FSM_Situation.NORMAL and i==self._robot._cfg.loco_manipulation.manipulate_leg_idx:
                        continue
                    else:  # locomotion, foot manipulation, and switching from loco-manipulation to stance
                        joint_space_input_gripper_idx.append(i)
            if len(joint_space_input_gripper_idx) > 0:
                des_gripper_rot_foot = self._manipulator_kinematics.forward_kinematics(self._wbc_command.des_gripper_pva[0, joint_space_input_gripper_idx, 3:6], joint_space_input_gripper_idx)
                for i, gripper_idx in enumerate(joint_space_input_gripper_idx):
                    foot_rot_world = self._robot_model.framePlacement(None, self._foot_frame_ids[gripper_idx], update_kinematics=False).rotation
                    self._wbc_command.des_gripper_pva[0, gripper_idx, 3:6] = rot_mat_to_rpy(foot_rot_world.dot(des_gripper_rot_foot[i]))
                    self._wbc_command.des_gripper_pva[1, gripper_idx, 3:6] = np.zeros(3)
                    self._wbc_command.des_gripper_pva[2, gripper_idx, 3:6] = np.zeros(3)

    def _compute_stance_foot_position_jacobian(self):
        foot_jacobians = []
        for idx, foot_frame_id in enumerate(self._foot_frame_ids):
            if self._wbc_command.contact_state[idx]:
                foot_jacobians.append(
                    self._robot_model.getFrameJacobian(foot_frame_id, rf_frame=pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)[:3])
        return np.concatenate(foot_jacobians, axis=0)

    def _compute_swing_foot_position_jacobian(self):
        foot_jacobians = []
        for idx, foot_frame_id in enumerate(self._foot_frame_ids):
            # LOCAL_WORLD_ALIGNED: while the origin of this frame moves with the moving part, its orientation remains fixed in alignment with the global reference frame.
            if not self._wbc_command.contact_state[idx]:
                foot_jacobians.append(
                    self._robot_model.getFrameJacobian(foot_frame_id, rf_frame=pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)[:3])
        foot_jacobians = np.concatenate(foot_jacobians, axis=0)
        foot_jacobians[:, :6] = 0  # Not affecting base!
        return foot_jacobians

    def _compute_gripper_jacobian(self, gripper_idx, type='position'):
        idx_begin = 0 if type == 'position' else 3
        idx_end = 3 if type == 'position' else 6
        legged_arm_jacobians = []
        for idx in gripper_idx:
            legged_arm_jacobians.append(
                    self._robot_model.getFrameJacobian(self._gripper_frame_ids[idx], rf_frame=pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)[idx_begin:idx_end])
        legged_arm_jacobians = np.concatenate(legged_arm_jacobians, axis=0)
        legged_arm_jacobians[:, :6] = 0  # Not affecting base!
        return legged_arm_jacobians

    def _compute_orientation_error_from_two_world_rpy(self, des_rpy, actual_rpy):
        des_rot = rpy_to_rot_mat(des_rpy)
        actual_rot = rpy_to_rot_mat(actual_rpy)
        return rot_mat_to_rpy(des_rot.dot(actual_rot.T))

    def _compute_orientation_error_from_world_rpy_and_rot_mat(self, des_rpy, actual_rot):
        des_rot = rpy_to_rot_mat(des_rpy)
        return rot_mat_to_rpy(des_rot.dot(actual_rot.T))
    
    def compute_actions(self):
        self._hierarchical_tracking()
        if self._robot._fsm_situation == FSM_Situation.NORMAL:
            self._check_safety()
        else:
            self._safe_command = True
        if self._safe_command:
            self._qp_optimization()
            self._last_des_q[:], self._last_dq_cmd[:], self._last_torque[:] = self._des_q[-self._num_joints:], self._dq_cmd[-self._num_joints:], self._torque[-self._num_joints:]
            self.log_wbc_info()
            return self._safe_command, self._des_q, self._dq_cmd[-self._num_joints:], self._torque[-self._num_joints:]
        else:
            self.log_wbc_info()
            return self._safe_command, self._last_des_q, self._last_dq_cmd, self._last_torque
        
    def _hierarchical_tracking(self):
        # reset the buffers
        self._delta_q[:] = 0
        self._dq_cmd[:] = 0
        self._ddq_cmd[:] = 0
        wbc_steps = self._tasks_num

        # Torso Jacobian
        torso_jacobian = self._robot_model.getFrameJacobian(self._torso_frame_id, rf_frame=pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)  # 6 x 24, [:, 6:]=0

        # Mass matrix
        mass_matrix = self._robot_model.mass(self._q)  # 24 x 24, [:6, :] & [6:, :6] & [6:, the columns for the same leg] != 0
        mass_matrix_inv = scipy.linalg.inv(mass_matrix)  # 24 x 24, all the elements are non-zero

        # compute the contact leg jacobian
        num_stance_foot = np.sum(self._wbc_command.contact_state)
        if num_stance_foot > 0:
            stance_foot_jacobian = self._compute_stance_foot_position_jacobian()  # 3c x 24, only the columns of the torso and the leg's joints are non-zero
            self._J[0] = stance_foot_jacobian
            jcdqd = []
            for stance_index in np.nonzero(self._wbc_command.contact_state)[0]:
                frame_acc = self._robot_model.frameClassicalAcceleration(None, None, None, self._foot_frame_ids[stance_index], update_kinematics=False).vector[:3]
                jcdqd.extend(frame_acc)
            jcdqd = np.array(jcdqd)  # 3c
            self._ddq_cmd[:] = compute_dynamically_consistent_pseudo_inverse(stance_foot_jacobian, mass_matrix_inv).dot(-jcdqd)  # M^{-1} J^T (J M^{-1} J^T)^{-1} (-J_c \ddot{q}_d) # 24
            # The entire expression calculates what is known as the projection matrix into the null space of the stance foot Jacobian.
            # Physically, this represents a transformation that identifies motions (in the joint space) that do not result in any change in the position or orientation (pose) of the robot's stance foot.
            # In other words, it finds the internal motions of the robot that are "invisible" to the stance foot's pose, allowing the robot to adjust its posture or balance without affecting the foot's contact with the ground.
            self._N[0] = np.eye(self._num_joints+6) - scipy.linalg.pinv(stance_foot_jacobian, rcond=1e-3).dot(stance_foot_jacobian)  # 24 x 24
            self._N_dyn[0] = np.eye(self._num_joints+6) - compute_dynamically_consistent_pseudo_inverse(stance_foot_jacobian, mass_matrix_inv).dot(stance_foot_jacobian)  # 24 x 24
        else:
            self._N[0], self._N_dyn[0] = np.eye(self._num_joints+6), np.eye(self._num_joints+6)

        # Object 1: torso position
        self._e[1] = self._wbc_command.des_torso_pva[0, 0:3] - self._robot.torso_pos_w_np[self._env_ids]
        self._dx[1] = self._wbc_command.des_torso_pva[1, 0:3]
        torso_position_kp = self._torso_position_kp_loco if self._wbc_command.operation_mode in Locomotion_Modes else self._torso_position_kp_mani
        torso_position_kd = self._torso_position_kd_loco if self._wbc_command.operation_mode in Locomotion_Modes else self._torso_position_kd_mani
        self._ddx[1] = self._wbc_command.des_torso_pva[2, 0:3] + torso_position_kp * self._e[1] + torso_position_kd * (self._dx[1] - self._robot.torso_lin_vel_w_np[self._env_ids])
        self._J[1] = torso_jacobian[:3]
        self._dJdq[1] = self._robot_model.frameClassicalAcceleration(None, None, None, self._torso_frame_id, update_kinematics=False).vector[:3]

        # Object 2: torso orientation
        self._e[2] = self._compute_orientation_error_from_two_world_rpy(self._wbc_command.des_torso_pva[0, 3:6], self._robot.torso_rpy_w2b_np[self._env_ids])
        self._dx[2] = self._wbc_command.des_torso_pva[1, 3:6]
        torso_orientation_kp = self._torso_orientation_kp_loco if self._wbc_command.operation_mode in Locomotion_Modes else self._torso_orientation_kp_mani
        torso_orientation_kd = self._torso_orientation_kd_loco if self._wbc_command.operation_mode in Locomotion_Modes else self._torso_orientation_kd_mani
        self._ddx[2] = self._wbc_command.des_torso_pva[2, 3:6] + torso_orientation_kp * self._e[2] + torso_orientation_kd * (self._dx[2] - self._robot.torso_ang_vel_w_np[self._env_ids])
        self._J[2] = torso_jacobian[3:6]
        self._dJdq[2] = self._robot_model.frameClassicalAcceleration(None, None, None, self._torso_frame_id, update_kinematics=False).vector[3:6]

        # Object 3: swing foot or gripper position
        swing_indices = np.nonzero(np.logical_not(self._wbc_command.contact_state))[0]
        if len(swing_indices) > 0:
            num_swing_foot = len(swing_indices)
            if self._use_gripper and self._wbc_command.operation_mode in Gripper_Manipulation_Modes:
                gripper_position_kp = np.concatenate([self._gripper_position_kp] * num_swing_foot, axis=0)
                gripper_position_kd = np.concatenate([self._gripper_position_kd] * num_swing_foot, axis=0)
                manipulate_gripper_lin_vel = []
                j3dqd = []
                for swing_index in swing_indices:
                    manipulate_gripper_lin_vel.append(self._robot_model.frameVelocity(None, None, self._gripper_frame_ids[swing_index], update_kinematics=False, reference_frame=pin.ReferenceFrame.LOCAL_WORLD_ALIGNED).vector[:3])
                    j3dqd.extend(self._robot_model.frameClassicalAcceleration(None, None, None, self._gripper_frame_ids[swing_index], update_kinematics=False).vector[:3])
                manipulate_gripper_lin_vel = np.concatenate(manipulate_gripper_lin_vel)
                self._e[3] = self._wbc_command.des_gripper_pva[0, swing_indices, 0:3].flatten() - self._robot.eef_pos_w_np[self._env_ids][swing_indices].flatten()
                self._dx[3] = self._wbc_command.des_gripper_pva[1, swing_indices, 0:3].flatten()
                self._ddx[3] = self._wbc_command.des_gripper_pva[2, swing_indices, 0:3].flatten() + gripper_position_kp * self._e[3] - gripper_position_kd * (self._dx[3] - manipulate_gripper_lin_vel)
                self._J[3] = self._compute_gripper_jacobian(gripper_idx=swing_indices, type='position')
                self._dJdq[3] = np.array(j3dqd)
            else:
                foot_position_kp = np.concatenate([self._foot_position_kp] * num_swing_foot, axis=0)
                foot_position_kd = np.concatenate([self._foot_position_kd] * num_swing_foot, axis=0)
                swing_foot_lin_vel = []
                j3dqd = []
                for swing_index in swing_indices:
                    swing_foot_lin_vel.append(self._robot_model.frameVelocity(None, None, self._foot_frame_ids[swing_index], update_kinematics=False, reference_frame=pin.ReferenceFrame.LOCAL_WORLD_ALIGNED).vector[:3])
                    j3dqd.extend(self._robot_model.frameClassicalAcceleration(None, None, None, self._foot_frame_ids[swing_index], update_kinematics=False).vector[:3])
                swing_foot_lin_vel = np.concatenate(swing_foot_lin_vel)
                self._e[3] = self._wbc_command.des_foot_pva[0, swing_indices, :].flatten() - self._robot.foot_pos_w_np[self._env_ids][swing_indices].flatten()
                self._dx[3] = self._wbc_command.des_foot_pva[1, swing_indices, :].flatten()
                self._ddx[3] = self._wbc_command.des_foot_pva[2, swing_indices, :].flatten() + foot_position_kp * self._e[3] - foot_position_kd * (self._dx[3] - swing_foot_lin_vel)
                self._J[3] = self._compute_swing_foot_position_jacobian()
                self._dJdq[3] = np.array(j3dqd)
        else:
            wbc_steps = self._tasks_num - 1 # no swing foot or gripper position task

        # Object 4: gripper orientation
        if self._use_gripper:
            obj_idx = 4 + wbc_steps - self._tasks_num # 4 if tracking swing foot position, 3 if not
            for grip_idx in range(2):
                self._e[grip_idx+obj_idx] = self._compute_orientation_error_from_world_rpy_and_rot_mat(self._wbc_command.des_gripper_pva[0, grip_idx, 3:6], self._robot.eef_rot_w_np[self._env_ids][grip_idx])
                self._dx[grip_idx+obj_idx] = self._wbc_command.des_gripper_pva[1, grip_idx, 3:6]
                gripper_ang_vel = self._robot_model.frameVelocity(None, None, self._gripper_frame_ids[grip_idx], update_kinematics=False, reference_frame=pin.ReferenceFrame.LOCAL_WORLD_ALIGNED).vector[3:6]
                self._ddx[grip_idx+obj_idx] = self._wbc_command.des_gripper_pva[2, grip_idx, 3:6] + self._gripper_orientation_kp * self._e[grip_idx+obj_idx] - self._gripper_orientation_kd * (self._dx[grip_idx+obj_idx] - gripper_ang_vel)
                self._J[grip_idx+obj_idx] = self._compute_gripper_jacobian(gripper_idx=[grip_idx], type='orientation')
                self._dJdq[grip_idx+obj_idx] = np.array(self._robot_model.frameClassicalAcceleration(None, None, None, self._gripper_frame_ids[grip_idx], update_kinematics=False).vector[3:6])

        # Null space projection and Jacobian construction
        for i in range(1, wbc_steps+1):
            if i>1:
                self._N[i-1] = self._N[i-2] @ self._N_i_im1[i-1]
                self._N_dyn[i-1] = self._N_dyn[i-2] @ self._N_i_im1_dyn[i-1]
            self._J_pre[i] = self._J[i] @ self._N[i-1]
            self._J_pre_dyn[i] = self._J[i] @ self._N_dyn[i-1]
            self._J_i_im1[i] = self._J[i] @ (np.eye(self._num_joints+6) - scipy.linalg.pinv(self._J[i-1]) @ self._J[i-1])
            self._N_i_im1[i] = np.eye(self._num_joints+6) - scipy.linalg.pinv(self._J_i_im1[i]) @ self._J_i_im1[i]
            self._J_i_im1_dyn[i] = self._J[i] @ (np.eye(self._num_joints+6) - compute_dynamically_consistent_pseudo_inverse(self._J[i-1], mass_matrix_inv) @ self._J[i-1])
            self._N_i_im1_dyn[i] = np.eye(self._num_joints+6) - compute_dynamically_consistent_pseudo_inverse(self._J_i_im1_dyn[i], mass_matrix_inv) @ self._J_i_im1_dyn[i]
            self._J_pre_inv[i] = scipy.linalg.pinv(self._J_pre[i])
            self._J_pre_dyn_inv[i] = compute_dynamically_consistent_pseudo_inverse(self._J_pre_dyn[i], mass_matrix_inv)

            # compute command
            self._delta_q[:] = self._delta_q + self._J_pre_inv[i] @ (self._e[i] - self._J[i] @ self._delta_q)
            self._dq_cmd[:] = self._dq_cmd + self._J_pre_inv[i] @ (self._dx[i] - self._J[i] @ self._dq_cmd)
            self._ddq_cmd[:] = self._ddq_cmd + self._J_pre_dyn_inv[i] @ (self._ddx[i] - self._dJdq[i] - self._J[i] @ self._ddq_cmd)
        
        self._des_q[:] = self._q[-self._num_joints:] + self._delta_q[-self._num_joints:]

    def _qp_optimization(self):
        # Clip acceleration commands
        self._ddq_cmd[:6] = np.clip(self._ddq_cmd[:6], self._torso_acc_limits[0], self._torso_acc_limits[1])
        self._ddq_cmd[6:] = np.clip(self._ddq_cmd[6:], self._joint_acc_limits[0], self._joint_acc_limits[1])

        # Prepare items for QP
        mass_matrix = self._robot_model.mass(self._q)  # 24 x 24
        foot_jacobian = self._compute_stance_foot_position_jacobian()  # 3*num_stance_foot x 24
        coriolis_gravity = self._robot_model.nle(self._q, self._dq)  # 24, coriolis-gravity
        num_stance_foot = np.sum(self._wbc_command.contact_state)
        dim_variables = 6 + 3 * num_stance_foot  # 6 for torso acceleration, 3 for each stance foot's force

        # Objective: 1/2 x^T G x - a^T x
        G = np.zeros((dim_variables, dim_variables))
        G[:6, :6] = self._W_torso
        G[6:, 6:] = np.eye(3 * num_stance_foot) * self._W_foot
        a = np.zeros((dim_variables, ))
        a[:6] = self._ddq_cmd[:6].T.dot(self._W_torso)  # add a bias towards the desired torso accelerations
        a[6:] = np.zeros(3 * num_stance_foot)

        # Equality constraint (robot dynamics): CE * x = ce
        A = mass_matrix[:6, :6]  # 6 x 6
        jc_t = foot_jacobian.T[:6]  # 6 x 3*num_stance_foot
        nle = coriolis_gravity[:6]  # 6
        CE = np.zeros((6, dim_variables))
        CE[:, :6] = A
        CE[:, 6:] = -jc_t
        ce = -nle

        # Inequality constraint (friction cone): CI * x >= 0
        CI = np.zeros((5 * num_stance_foot, dim_variables))
        for idx in range(num_stance_foot):
          CI[idx * 5:idx * 5 + 5, 6 + idx * 3:9 + idx * 3] = self._friction_constraints_per_leg

        # Construct both equality and inequality constraints 
        C = np.concatenate((CE, -CE, CI), axis=0).T
        b = np.concatenate((ce - 1e-4, -ce - 1e-4, np.zeros(5 * num_stance_foot)))

        # Call quadprog to solve QP
        sol = quadprog.solve_qp(G, a, C, b)
        ddq = self._ddq_cmd.copy()
        ddq[:6] = sol[0][:6]
        fr = sol[0][6:]

        # Compute motor torques with multi-body dynamics
        self._torque[:] = mass_matrix.dot(ddq) + coriolis_gravity - foot_jacobian.T.dot(fr)

    def _check_safety(self):
        self._safe_command = True
        # make sure to use the command from the first frame
        if self._first_command:
            self._first_command = False
            return
        # check delta q, dq, ddq
        if self._wbc_command.operation_mode == FSM_State.LOCOMOTION:
            if np.sum(abs(self._delta_q[-self._num_joints:]) > self._loco_max_delta_q) + np.sum(abs(self._dq_cmd[-self._num_joints:]) > self._loco_max_dq) + np.sum(abs(self._ddq_cmd[-self._num_joints:]) > self._loco_max_ddq) > 0:
                self._safe_command = False
                print('-----------Beyond Loco Max-----------')
                print('delta_q:', self._delta_q[-self._num_joints:])
                print('dq_cmd:', self._dq_cmd[-self._num_joints:])
                print('ddq_cmd:', self._ddq_cmd[-self._num_joints:])
                return
        elif self._wbc_command.operation_mode in Manipulation_Modes:
            self._safe_command = np.sum((self._des_q >= self._min_joint_pos) & (self._des_q <= self._max_joint_pos)) == self._des_q.shape[0]
            if not self._safe_command:
                print('\n-----------Beyond Joint Limits-----------')
                print('delta_q:', self._delta_q[-self._num_joints:])
                print('dq_cmd:', self._dq_cmd[-self._num_joints:])
                print('ddq_cmd:', self._ddq_cmd[-self._num_joints:])
                if self._use_gripper:
                    print("des_gripper_xyzrpy: ", self._wbc_command.des_gripper_pva[0])
                    print("cur_gripper_xyzrpy: ", self._robot.eef_pos_w_np[self._env_ids], self._robot.eef_rpy_w_np[self._env_ids])
                return
            if np.sum(abs(self._delta_q[-self._num_joints:]) > self._mani_max_delta_q) + np.sum(abs(self._dq_cmd[-self._num_joints:]) > self._mani_max_dq) + np.sum(abs(self._ddq_cmd[-self._num_joints:]) > self._mani_max_ddq) > 0:
                self._safe_command = False
                print('-----------Beyond Mani Max-----------')
                print('delta_q:', self._delta_q[-self._num_joints:])
                print('dq_cmd:', self._dq_cmd[-self._num_joints:])
                print('ddq_cmd:', self._ddq_cmd[-self._num_joints:])
                return
            
    def log_wbc_info(self):
        if self._robot._log_info_now:
            print("\n----------------- WBC -------------------------")
            print("operation_mode:", self._wbc_command.operation_mode)
            print("contact_state:", self._wbc_command.contact_state)
            print("----- Torso -----")
            print("Des Pos: {: .4f}, {: .4f}, {: .4f}".format(*self._wbc_command.des_torso_pva[0, 0:3]))
            print("Cur Pos: {: .4f}, {: .4f}, {: .4f}".format(*self._robot.torso_pos_w_np[self._env_ids]))
            print("Des RPY: {: .4f}, {: .4f}, {: .4f}".format(*np.rad2deg(self._wbc_command.des_torso_pva[0, 3:6])))
            print("Cur RPY: {: .4f}, {: .4f}, {: .4f}".format(*np.rad2deg(self._robot.torso_rpy_w2b_np[self._env_ids])))
            print("Des Lin Vel: {: .4f}, {: .4f}, {: .4f}".format(*self._wbc_command.des_torso_pva[1, 0:3]))
            print("Cur Lin Vel: {: .4f}, {: .4f}, {: .4f}".format(*self._robot.torso_lin_vel_w_np[self._env_ids]))
            print("Des Ang Vel: {: .4f}, {: .4f}, {: .4f}".format(*np.rad2deg(self._wbc_command.des_torso_pva[1, 3:6])))
            print("Cur Ang Vel: {: .4f}, {: .4f}, {: .4f}".format(*np.rad2deg(self._robot.torso_ang_vel_w_np[self._env_ids])))
            print("----- Foot -----")
            print("Des Pos:\n{}".format(self._wbc_command.des_foot_pva[0, :, :]))
            print("Cur Pos:\n{}".format(self._robot.foot_pos_w_np[self._env_ids]))
            if self._use_gripper:
                print("----- Gripper -----")
                print("Des Pos:\n{}".format(self._wbc_command.des_gripper_pva[0, :, 0:3]))
                print("Cur Pos:\n{}".format(self._robot.eef_pos_w_np[self._env_ids]))
                print("Des RPY:\n{}".format(self._wbc_command.des_gripper_pva[0, :, 3:6]))
                print("Cur RPY:\n{}".format(self._robot.eef_rpy_w_np[self._env_ids]))


def compute_dynamically_consistent_pseudo_inverse(J, M_inv):
    M_inv_J_t = M_inv @ (J.T)  # M^{-1} J^T
    return M_inv_J_t @ (scipy.linalg.inv(J @ M_inv_J_t))  # (M^{-1} J^T) (J M^{-1} J^T)^{-1}





