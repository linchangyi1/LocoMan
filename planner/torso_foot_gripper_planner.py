import torch
from planner.trajectory_planner import TrajectoryPlanner


class TorsoFootGripperPlanner:
    def __init__(self, dt, num_envs, device, torso_action_dim, footgripper_pst_dim, gripper_ori_dim):
        self._dt = dt
        self._num_envs = num_envs
        self._device = device
        self._torso_action_dim = torso_action_dim
        self._footgripper_pst_dim = footgripper_pst_dim
        self._gripper_ori_dim = gripper_ori_dim
        self._init_buffers()

    def _init_buffers(self):
        # general information
        self._all_env_idx = torch.arange(self._num_envs, device=self._device)
        self._action_duration = torch.ones(self._num_envs, device=self._device)
        self._current_phase = torch.zeros(self._num_envs, device=self._device)

        # torso trajectory
        self._init_torso_state = torch.zeros((self._num_envs, self._torso_action_dim), device=self._device)
        self._final_torso_state = torch.zeros((self._num_envs, self._torso_action_dim), device=self._device)
        self._torso_pos_ori_planner = TrajectoryPlanner(num_envs=self._num_envs, action_dim=self._torso_action_dim, device=self._device)
        self._torso_pos_ori_planner.setInitialPosition(self._all_env_idx, self._init_torso_state[self._all_env_idx])
        self._torso_pos_ori_planner.setFinalPosition(self._all_env_idx, self._final_torso_state[self._all_env_idx])
        self._torso_pos_ori_planner.setDuration(self._all_env_idx, self._action_duration[self._all_env_idx])

        # leg or gripper position trajectory (depends on the operation mode)
        self._init_footgripper_pst = torch.zeros((self._num_envs, self._footgripper_pst_dim), device=self._device)
        self._final_footgripper_pst = torch.zeros((self._num_envs, self._footgripper_pst_dim), device=self._device)
        self._footgripper_pst_planner = TrajectoryPlanner(num_envs=self._num_envs, action_dim=self._footgripper_pst_dim, device=self._device)
        self._footgripper_pst_planner.setInitialPosition(self._all_env_idx, self._init_footgripper_pst[self._all_env_idx])
        self._footgripper_pst_planner.setFinalPosition(self._all_env_idx, self._final_footgripper_pst[self._all_env_idx])
        self._footgripper_pst_planner.setDuration(self._all_env_idx, self._action_duration[self._all_env_idx])

        # gripper orientations trajectories (for switching "orientations" are in joint space)
        self._init_gripper_oris = torch.zeros((self._num_envs, 2, self._gripper_ori_dim), device=self._device)
        self._final_gripper_oris = torch.zeros((self._num_envs, 2, self._gripper_ori_dim), device=self._device)
        self._gripper_ori_planners = [TrajectoryPlanner(num_envs=self._num_envs, action_dim=self._gripper_ori_dim, device=self._device) for _ in range(2)]
        for i in range(2):
            self._gripper_ori_planners[i].setInitialPosition(self._all_env_idx, self._init_gripper_oris[self._all_env_idx, i])
            self._gripper_ori_planners[i].setFinalPosition(self._all_env_idx, self._final_gripper_oris[self._all_env_idx, i])
            self._gripper_ori_planners[i].setDuration(self._all_env_idx, self._action_duration[self._all_env_idx])

    def step(self):
        self._current_phase[:] += self._dt / self._action_duration[:]
        self._current_phase[:] = torch.clip(self._current_phase[:], 0.0, 1.0)
        self._torso_pos_ori_planner.update(self._current_phase)
        self._footgripper_pst_planner.update(self._current_phase)
        for i in range(2):
            self._gripper_ori_planners[i].update(self._current_phase)
        return self._current_phase==1.0

    # only update the init_state, will not affect the final state, but will make the current phase to be 0
    def set_init_state(self, torso_init_state, footgripper_init_pos, gripper_init_oris, env_ids):
        self._current_phase[env_ids] = .0
        if torso_init_state is not None:
            self._init_torso_state[env_ids] = torso_init_state[env_ids]
            self._torso_pos_ori_planner.setInitialPosition(env_ids, self._init_torso_state[env_ids])
        if footgripper_init_pos is not None:
            self._init_footgripper_pst[env_ids] = footgripper_init_pos[env_ids]
            self._footgripper_pst_planner.setInitialPosition(env_ids, self._init_footgripper_pst[env_ids])
        if gripper_init_oris is not None:
            for i in range(2):
                self._init_gripper_oris[env_ids, i] = gripper_init_oris[env_ids, i]
                self._gripper_ori_planners[i].setInitialPosition(env_ids, self._init_gripper_oris[env_ids, i])

    def set_final_state(self, torso_final_state, footgripper_final_pos, gripper_final_oris, action_duration, env_ids):
        # update the action duration and current phase
        self._action_duration[env_ids] = action_duration[env_ids]
        self._action_duration[self._action_duration==0] = 1.0
        self._current_phase[env_ids] = .0

        # update the start and end state of the torso state trajectory
        self._init_torso_state[env_ids] = self._torso_pos_ori_planner.getPosition()[env_ids]
        self._final_torso_state[env_ids] = self._init_torso_state[env_ids] if torso_final_state is None else torso_final_state[env_ids]
        self._torso_pos_ori_planner.setInitialPosition(env_ids, self._init_torso_state[env_ids])
        self._torso_pos_ori_planner.setFinalPosition(env_ids, self._final_torso_state[env_ids])
        self._torso_pos_ori_planner.setDuration(env_ids, self._action_duration[env_ids])

        # update the start and end state of the footgripper position trajectory
        self._init_footgripper_pst[env_ids] = self._footgripper_pst_planner.getPosition()[env_ids]
        self._final_footgripper_pst[env_ids] = self._init_footgripper_pst[env_ids] if footgripper_final_pos is None else footgripper_final_pos[env_ids]
        self._footgripper_pst_planner.setInitialPosition(env_ids, self._init_footgripper_pst[env_ids])
        self._footgripper_pst_planner.setFinalPosition(env_ids, self._final_footgripper_pst[env_ids])
        self._footgripper_pst_planner.setDuration(env_ids, self._action_duration[env_ids])

        # update the start and end state of the gripper orientation trajectories
        for i in range(2):
            self._init_gripper_oris[env_ids, i] = self._gripper_ori_planners[i].getPosition()[env_ids]
            self._final_gripper_oris[env_ids, i] = self._init_gripper_oris[env_ids, i] if gripper_final_oris is None else gripper_final_oris[env_ids, 3*i:3*i+3]
            self._gripper_ori_planners[i].setInitialPosition(env_ids, self._init_gripper_oris[env_ids, i])
            self._gripper_ori_planners[i].setFinalPosition(env_ids, self._final_gripper_oris[env_ids, i])
            self._gripper_ori_planners[i].setDuration(env_ids, self._action_duration[env_ids])

    def get_desired_torso_pva(self):
        return torch.cat((self._torso_pos_ori_planner._p, self._torso_pos_ori_planner._v, self._torso_pos_ori_planner._a), dim=1)

    def get_desired_footgripper_pva(self):
        return torch.cat((self._footgripper_pst_planner._p, self._footgripper_pst_planner._v, self._footgripper_pst_planner._a), dim=1)

    def get_desired_gripper_ori_pva(self):
        pva = []
        for i in range(2):
            pva.append(torch.cat((self._gripper_ori_planners[i]._p, self._gripper_ori_planners[i]._v, self._gripper_ori_planners[i]._a), dim=1))
        return torch.stack(pva, dim=1)

    def reset(self, torso_state=None, action_duration=None, env_ids=None, footgripper_pst=None):
        if env_ids is None:
            return
        if action_duration is None:
            action_duration = torch.ones_like(env_ids, device=self._device)
        self._action_duration[env_ids] = action_duration[env_ids]
        self._current_phase[env_ids] = .0

        if torso_state is None:
            torso_state = torch.zeros((self._num_envs, self._torso_action_dim), device=self._device)
        self._init_torso_state[env_ids] = torso_state[env_ids]
        self._final_torso_state[env_ids] = torso_state[env_ids]
        self._torso_pos_ori_planner.setInitialPosition(env_ids, self._init_torso_state[env_ids])
        self._torso_pos_ori_planner.setFinalPosition(env_ids, self._final_torso_state[env_ids])
        self._torso_pos_ori_planner.setDuration(env_ids, self._action_duration[env_ids])

        if footgripper_pst is None:
            footgripper_pst = torch.zeros((self._num_envs, self._footgripper_pst_dim), device=self._device)
        self._init_footgripper_pst[env_ids] = footgripper_pst[env_ids]
        self._final_footgripper_pst[env_ids] = footgripper_pst[env_ids]
        self._footgripper_pst_planner.setInitialPosition(env_ids, self._init_footgripper_pst[env_ids])
        self._footgripper_pst_planner.setFinalPosition(env_ids, self._final_footgripper_pst[env_ids])
        self._footgripper_pst_planner.setDuration(env_ids, self._action_duration[env_ids])


