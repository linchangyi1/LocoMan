from params_proto import PrefixProto
import numpy as np
from fsm.finite_state_machine import FSM_Command, FSM_State


class Cfg(PrefixProto, cli=False):
    class sim(PrefixProto, cli=False):
        use_real_robot = False
        num_envs = 4
        use_gpu = True
        show_gui = True
        sim_device = 'cuda:0'
        use_gripper = True

    def update_parms():
        if Cfg.sim.use_real_robot:
            Cfg.sim.num_envs = 1
            Cfg.sim.use_gpu = False
            Cfg.sim.show_gui = False
            Cfg.sim.sim_device = "cpu"
        else:
            from isaacgym import gymapi  # gymapi should be imported before torch
            Cfg.motor_control.reset_time = 1.5  # less time to reset in simulation

    class logging(PrefixProto, cli=False):
        log_info = False
        log_interval = 100

    class wbc(PrefixProto, cli=False):
        # Tracking objectives and their dimensions
        tracking_objects_w_gripper = ['torso_pos', 'torso_rpy', 'swing_footgripper_pst', 'swing_gripper_rpy', 'stance_gripper_pos']
        tracking_objects_dims_w_gripper = [3, 3, 12, 3, 3]
        tracking_objects_wo_gripper = ['torso_pos', 'torso_rpy', 'swing_foot_pos']
        tracking_objects_dims_wo_gripper = [3, 3, 12]

        # Gains for computing ddx in WBC
        class real(PrefixProto, cli=False):
            # locomotion - torso
            torso_position_kp_loco = np.array([1., 1., 1.]) * 100
            torso_position_kd_loco = np.array([1., 1., 1.]) * 10
            torso_orientation_kp_loco = np.array([1., 1., 1.]) * 100
            torso_orientation_kd_loco = np.array([1., 1., 1.]) * 10

            # manipulation - torso
            torso_position_kp_mani = np.array([1., 1., 1.]) * 100
            torso_position_kd_mani = np.array([1., 1., 1.]) * 1
            torso_orientation_kp_mani = np.array([1., 1., 1.]) * 100
            torso_orientation_kd_mani = np.array([1., 1., 1.]) * 1

            # both - foot, gripper
            foot_position_kp = np.array([1., 1., 1.]) * 100
            foot_position_kd = np.array([1., 1., 1.]) * 10
            gripper_position_kp = np.array([1., 1., 1.]) * 100
            gripper_position_kd = np.array([1., 1., 1.]) * 10
            gripper_orientation_kp = np.array([1., 1., 1.]) * 100
            gripper_orientation_kd = np.array([1., 1., 1.]) * 10

        class sim(PrefixProto, cli=False):
            # locomotion - torso
            torso_position_kp_loco = np.array([1., 1., 1.]) * 100
            torso_position_kd_loco = np.array([1., 1., 1.]) * 10
            torso_orientation_kp_loco = np.array([1., 1., 1.]) * 100
            torso_orientation_kd_loco = np.array([1., 1., 1.]) * 10

            # manipulation - torso
            torso_position_kp_mani = np.array([1., 1., 1.]) * 100
            torso_position_kd_mani = np.array([1., 1., 1.]) * 1
            torso_orientation_kp_mani = np.array([1., 1., 1.]) * 100
            torso_orientation_kd_mani = np.array([1., 1., 1.]) * 1

            # both - foot, gripper
            foot_position_kp = np.array([1., 1., 1.]) * 100
            foot_position_kd = np.array([1., 1., 1.]) * 10
            gripper_position_kp = np.array([1., 1., 1.]) * 100
            gripper_position_kd = np.array([1., 1., 1.]) * 10
            gripper_orientation_kp = np.array([1., 1., 1.]) * 100
            gripper_orientation_kd = np.array([1., 1., 1.]) * 10

        # acceleration limits
        torso_acc_limits = np.array([[-5, -5, -10, -10, -10, -10], [5, 5, 30, 10, 10, 10]])
        joint_acc_limits = np.array([-10, 10])
        torso_qp_weight = [20.0, 20.0, 5.0, 1.0, 1.0, 0.2]
        foot_qp_weight = 1e-5
        friction_coef = 0.6

        # Threadsholds for safety checking
        mani_max_delta_q = 0.5
        mani_max_dq = 100.0
        mani_max_ddq = 60.0
        loco_max_delta_q = 3.0
        loco_max_dq = 1000.0
        loco_max_ddq = 1000.0

    class motor_control(PrefixProto, cli=False):
        dt = 0.0025  # the control loop frequency
        reset_time = 3.0
        power_protect_level = 10

        # kp, kd: [hip, thigh, calf, manipulator_joint_1, manipulator_joint_2, manipulator_joint_3]
        class real(PrefixProto, cli=False):
            kp = np.array([200, 200, 200, 50, 5, 0.05])
            kd = kp * 0.01 - np.array([0., 0., 0., 50, 5, 0.05]) * 0.008

            # for locomotion
            kp_stance_loco = np.array([30, 30, 30, 50, 5, 0.05])
            kd_stance_loco = np.array([1, 1, 1, 0, 0, 0]) + np.array([0., 0., 0., 50, 5, 0.05]) * 0.002
            kp_swing_loco = np.array([30, 30, 30, 50, 5, 0.05])
            kd_swing_loco = np.array([1, 1, 1, 0, 0, 0]) + np.array([0., 0., 0., 50, 5, 0.05]) * 0.002

            # for manipulation
            kp_stance_mani = np.array([60, 60, 60, 50, 5, 0.05])
            kd_stance_mani = np.array([2, 2, 2, 0, 0, 0]) + np.array([0., 0., 0., 50, 5, 0.05]) * 0.002
            kp_swing_mani = np.array([60, 60, 60, 50, 5, 0.05])
            kd_swing_mani = np.array([2, 2, 2, 0, 0, 0]) + np.array([0., 0., 0., 50, 5, 0.05]) * 0.002

            # for bi-manipulation
            kp_bimanual_switch = np.array([30., 30., 30., 30., 30., 30., 80., 80., 80., 80., 80., 80.])
            kd_bimanual_switch = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0])
            kp_bimanual_command = np.array([30., 30., 30., 30., 30., 30., 100., 100., 100., 100., 100., 100.])
            kd_bimanual_command = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5])

        class sim(PrefixProto, cli=False):
            kp = np.array([200, 200, 200, 50, 5, 0.05])
            kd = kp * 0.01 - np.array([0., 0., 0., 50, 5, 0.05]) * 0.008

            # for locomotion
            kp_stance_loco = np.array([30, 30, 30, 50, 5, 0.05])
            kd_stance_loco = np.array([1, 1, 1, 0, 0, 0]) + np.array([0., 0., 0., 50, 5, 0.05]) * 0.002
            kp_swing_loco = np.array([30, 30, 30, 50, 5, 0.05])
            kd_swing_loco = np.array([1, 1, 1, 0, 0, 0]) + np.array([0., 0., 0., 50, 5, 0.05]) * 0.002

            # for manipulation
            kp_stance_mani = np.array([60, 60, 60, 50, 5, 0.05])
            kd_stance_mani = np.array([2, 2, 2, 0, 0, 0]) + np.array([0., 0., 0., 50, 5, 0.05]) * 0.002
            kp_swing_mani = np.array([60, 60, 60, 50, 5, 0.05])
            kd_swing_mani = np.array([2, 2, 2, 0, 0, 0]) + np.array([0., 0., 0., 50, 5, 0.05]) * 0.002

            # for bi-manipulation
            kp_bimanual_switch = np.array([30., 30., 30., 30., 30., 30., 80., 80., 80., 80., 80., 80.])
            kd_bimanual_switch = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0])
            kp_bimanual_command = np.array([30., 30., 30., 30., 30., 30., 100., 120., 100., 100., 120., 100.])
            kd_bimanual_command = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.5, 3.0, 2.5, 2.5, 3.0, 2.5])

    class manipulator(PrefixProto, cli=False):
        vid_pid = "0403:6014"
        baudrate = 1000000

        # totally 8 motors in two manipulators
        motor_names = ['RM_joint_1', 'RM_joint_2', 'RM_joint_3', 'RM_eef', 'LM_joint_1', 'LM_joint_2', 'LM_joint_3', 'LM_eef']
        motor_ids = [1, 2, 3, 4, 5, 6, 7, 8]
        dof_idx = [0, 1, 2, 4, 5, 6]
        gripper_idx = [3, 7]
        manipulator_1_idx = [0, 1, 2, 3]
        manipulator_2_idx = [4, 5, 6, 7]
        min_position = [-3.25, -2.4, -3.15, 0., -3.25, -0.6, -3.15, 0.]
        max_position = [1.74, 0.6, 3.15, -0.5, 1.74, 2.4, 3.15, -0.5]
        min_velocity = [-20.0, -20.0, -20.0, -20.0, -20.0, -20.0, -20.0, -20.0]
        max_velocity = [20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0]
        min_torque = [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0]
        max_torque = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

        reset_pos_sim = np.array([-np.pi, -0.05, 0., 0.02, -np.pi, 0.05, 0., 0.02])
        reset_time = 0.5
        kI = np.zeros(len(motor_ids))
        kP = np.array([8, 6, 6, 6, 8, 6, 6, 6]) * 128
        kD = np.array([65, 40, 40, 100, 65, 40, 40, 100]) * 16
        # Max at current (in unit 1ma)
        curr_lim = 900
        gripper_delta_max = 0.2
        # real_pos = sim_pos * s2r_scale + s2r_offset
        s2r_scale = np.array([-1.0, 1.0, -1.0, 1.0, 1.0, 1.0, -1.0, 1.0])
        s2r_offset = np.array([-np.pi, 0.0, 0.0, 0.0, np.pi, 0.0, 0.0, 0.0])

        des_pos_sim_topic = '/manipulator_des_pos_sim'
        cur_state_sim_topic = '/manipulator_cur_state_sim'
        update_manipulator_freq = 200


    class fsm(PrefixProto, cli=False):
        fsm_command_topic = "/fsm_command"
        fsm_command_mapping = {'stance': FSM_Command.STANCE.value,
                               'locomotion': FSM_Command.LOCOMOTION.value,
                               'lg_mani': FSM_Command.MANIPULATION_LEFT_GRIPPER.value,
                               'rg_mani': FSM_Command.MANIPULATION_RIGHT_GRIPPER.value,
                               'bi_mani': FSM_Command.BIMANIPULATION.value,
                               'loco_mani': FSM_Command.LOCOMANIPULATION.value,
                               'lf_mani': FSM_Command.MANIPULATION_LEFT_FOOT.value,
                               'rf_mani': FSM_Command.MANIPULATION_RIGHT_FOOT.value}

    class commander(PrefixProto, cli=False):
        eef_task_space = 'world'
        # eef_task_space = 'eef'
        human_command_topic = "/human_command"
        reset_manipulator_when_switch = True

        torso_pv_scale = np.array([0.0005, 0.0005, 0.002, 0.01, 0.01, 0.01, 0.02, 0.02, 0.02, 0.01, 0.01, 0.1])
        foot_xyz_scale = 0.001
        gripper_xyz_scale = 0.001
        gripper_rpy_scale = 0.005
        gripper_angle_scale = 0.02

        locomotion_height_range = [0.15, 0.40]
        gripper_angle_range = [0.02, 0.7]

        class sim_limit(PrefixProto, cli=False):
            torso_pv_limit = np.array([0.1, 0.1, 0.2, 0.3, 0.3, 0.4, 0.6, 0.6, 0, 0, 0, 1.0])

        class real_limit(PrefixProto, cli=False):
            torso_pv_limit = np.array([0.06, 0.06, 0.2, 0.3, 0.3, 0.4, 0.2, 0.2, 0, 0, 0, 0.3])

        # bi-manipulation may destroy the robot if the trajectory is not well collected
        # so be careful to unlock it after testing in simulation
        lock_real_robot_bimanual = False

    class switcher(PrefixProto, cli=False):
        min_switch_interval = 3.0 # minimum time interval between two switches
        locomotion_switching_time = 0.5

        class stance_and_single_manipulation(PrefixProto, cli=False):
            torso_movement = np.array([-0.05, 0.03, 0.02, .0, .0, .0])  # for right
            foot_movement = np.array([0., 0., 0.04])
            manipulator_angles = np.array([-np.pi/8, 0., 0.])
            torso_moving_time = 1.0
            foot_moving_time = 1.5
            manipulator_moving_time = 2.0

            # gripper_reset_pos_sim
            grp = np.array([-np.pi, -0.05, 0., 0.02, -np.pi, 0.05, 0., 0.02])
            # 0:6-body state, 6:9-foot_or_gripper_position, 9:15-gripper joint_pos(for switching),
            # 15-operation_mode, 16-reset_signal, 17-contact_state, 18-swing_leg,
            # 19:21gripper_angles, 21-duration, 22:stay time
            # The elements with a value of "-1." will be updated in the constructor
            stance_to_manipulation_trajectory = np.array([
            [-1., -1., -1., -1., -1., -1., 0., 0., 0., grp[0], grp[1], grp[2], grp[4], grp[5], grp[6], FSM_State.STANCE.value, 1, 1, -1, grp[3], grp[7], torso_moving_time, .1],
            [0., 0., 0., .0, .0, .0, -1., -1., -1., grp[0], grp[1], grp[2], grp[4], grp[5], grp[6], FSM_State.SF_MANIPULATION.value, 0, 0, -1, grp[3], grp[7], manipulator_moving_time, .1],
                                                    ])
            manipulation_to_stance_trajectory = np.array([
            [-1., -1., -1., -1., -1., -1., -1., -1., -1., grp[0], grp[1], grp[2], grp[4], grp[5], grp[6], FSM_State.SF_MANIPULATION.value, 0, 0, -1, grp[3], grp[7], manipulator_moving_time, .1],
            [0., 0., 0., .0, .0, .0, -1., -1., -1., grp[0], grp[1], grp[2], grp[4], grp[5], grp[6], FSM_State.SF_MANIPULATION.value, 1, 0, -1, grp[3], grp[7], foot_moving_time, .1],
            [-1, -1, 0., 0., 0., 0., 0., 0., 0., grp[0], grp[1], grp[2], grp[4], grp[5], grp[6], FSM_State.STANCE.value, 0, 1, -1, grp[3], grp[7], torso_moving_time, .1],
                                                    ])

        class stance_and_locomanipulation(PrefixProto, cli=False):            
            transition_time = 1.5
            stablization_time = 0.2


    class locomotion(PrefixProto, cli=False):
        early_touchdown_phase_threshold = 0.5
        lose_contact_phase_threshold = 0.1
        gait_params = [2, np.pi, np.pi, 0., 0.45]
        desired_pose = np.array([0., 0., 0.28, 0.0, 0., 0.])
        desired_velocity = np.array([0., 0., 0.0, 0.0, 0.0, 0.])
        foot_landing_clearance_sim = 0.0
        foot_landing_clearance_real = 0.01
        foot_height = 0.08


    class manipulation(PrefixProto, cli=False):
        torso_action_dim = 6
        footgripper_pst_dim = 3  # foot or gripper position dim
        gripper_ori_dim = 3  # gripper orientation dim


    class loco_manipulation(PrefixProto, cli=False):
        manipulate_leg_idx = 0  # 0:FR, 1:FL
        desired_eef_rpy_w = np.array([0., -np.pi, 0.]) if manipulate_leg_idx == 0 else np.array([0., -np.pi, 0.])


    class bimanual_trajectory(PrefixProto, cli=False):
        recording_fps_mul_motor_control_fps = 2
        move_gripper_foot_time = 1.5

        class with_gripper(PrefixProto, cli=False):
            trajectory_path = 'bimanual_trajectory/bimanual_w_gripper_data.pkl'
            stand_up_start_time = 0.1
            stand_up_end_time = 2.5
            stabilize_time = 0.5
            sit_down_start_time = 5.4
            sit_down_end_time = 7.2

        class without_gripper(PrefixProto, cli=False):
            trajectory_path = 'bimanual_trajectory/bimanual_wo_gripper_data.pkl'
            stand_up_start_time = 0.4
            stand_up_end_time = 2.2
            stabilize_time = 0.5
            sit_down_start_time = 5.6
            sit_down_end_time = 7.4


    class teleoperation(PrefixProto, cli=False):
        robot_state_topic = "/robot_state"

        class joystick(PrefixProto, cli=False):
            command_topic = "/joystick_command"
            feature_names = ['DualSense', 'Xbox 360']
            # button order: [action_down, action_up, action_left, action_right, l1, r1, l3, r3, ps_button]
            button_ids = [[0, 2, 3, 1, 4, 5, 11, 12, 10],
                          [0, 3, 2, 1, 4, 5, 9, 10, 8]]
            # hat order: [dpad_down, dpad_up, dpad_left, dpad_right]
            hat_values = [[(0, -1), (0, 1), (-1, 0), (1, 0)],
                       [(0, -1), (0, 1), (-1, 0), (1, 0)]]
            # axis order: [left_up_down, left_left_right, right_up_down, right_left_right, l2, r2]
            axis_ids = [[1, 0, 4, 3, 2, 5],
                        [1, 0, 4, 3, 2, 5]]
            torso_incremental_control = True  # True: incremental control, False: proprtional control
            gripper_task_space_world = True  # True: world frame, False: gripper frame

        class human_teleoperator(PrefixProto, cli=False):
            receive_action_topic = "/receive_action"
            SERVER_HOST = '189.176.158.13'
            SERVER_PORT = 12345
            mode = 3  # 0:stance, 1:right-arm manipulation, 2: left-arm manipulation 3: bi-manual manipulation, 4:None

            # thresholds are used to filter out the human shaking and noise
            torso_xyz_threshold = 0.003
            torso_rpy_threshold = 0.003
            eef_xyz_threshold = 0.001
            eef_rpy_threshold = 0.005
            gripper_angle_threshold = 0.01

            torso_xyz_scale = 0.5
            torso_rpy_scale = 1.0
            eef_xyz_scale = 1.0
            eef_rpy_scale = 1.0
            gripper_angle_scale = 1.0

            if mode == 1 or mode == 2:
                torso_xyz_scale = 0.0
                torso_rpy_scale = 1.0
            
            torso_xyz_max_step = 0.001
            torso_rpy_max_step = 0.002
            eef_xyz_max_step = 0.001
            eef_rpy_max_step = 0.002
            gripper_angle_max_step = 0.03


    class asset(PrefixProto, cli=False):
        urdf_path = 'asset/locoman/urdf/go1.urdf'

    class init_state(PrefixProto, cli=False):
        pos = [0.0, 0.0, 1.]  # x,y,z [m]
        rot = [0.0, 0.0, 0.0, 1.0]  # x,y,z,w [quat]
        lin_vel = [0.0, 0.0, 0.0]  # x,y,z [m/s]
        ang_vel = [0.0, 0.0, 0.0]  # x,y,z [rad/s]


    def get_sim_config(use_penetrating_contact=True):
        from isaacgym import gymapi
        from ml_collections import ConfigDict
        use_gpu = Cfg.sim.use_gpu
        # dt = 0.0005
        dt = 0.0025  # for faster simulation
        # simulation parameters
        sim_params = gymapi.SimParams()
        sim_params.use_gpu_pipeline = use_gpu
        sim_params.dt = dt
        sim_params.substeps = 1
        sim_params.up_axis = gymapi.UpAxis(gymapi.UP_AXIS_Z)
        sim_params.gravity = gymapi.Vec3(0., 0., -9.81)
        sim_params.physx.use_gpu = use_gpu
        sim_params.physx.num_subscenes = 0  #default_args.subscenes
        sim_params.physx.num_threads = 10
        sim_params.physx.solver_type = 1 # TGS
        sim_params.physx.num_position_iterations = 6 #4 improve solver convergence
        sim_params.physx.num_velocity_iterations = 1 # keep default
        if use_penetrating_contact:
            sim_params.physx.contact_offset = 0.
            sim_params.physx.rest_offset = -0.004
        else:
            sim_params.physx.contact_offset = 0.01
            sim_params.physx.rest_offset = 0.
        sim_params.physx.bounce_threshold_velocity = 0.2  #0.5 [m/s]
        sim_params.physx.max_depenetration_velocity = 100.0
        sim_params.physx.max_gpu_contact_pairs = 2**23  #2**24 needed for 8000+ envs
        sim_params.physx.default_buffer_size_multiplier = 5
        sim_params.physx.contact_collection = gymapi.ContactCollection(2)  # 0: never, 1: last sub-step, 2: all sub-steps (default=2)
        # plane parameters
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = 1.0
        plane_params.dynamic_friction = 1.0
        plane_params.restitution = 0.
        # create config
        config = ConfigDict()
        config.sim_device = 'cuda:0' if use_gpu else 'cpu'
        config.physics_engine = gymapi.SIM_PHYSX
        config.sim_params = sim_params
        config.plane_params = plane_params
        config.action_repeat = int(Cfg.motor_control.dt / sim_params.dt)
        config.dt = dt
        config.env_spacing = 2.0
        return config
    

    def get_asset_config():
        from isaacgym import gymapi
        from ml_collections import ConfigDict
        config = ConfigDict()
        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = 3
        asset_options.collapse_fixed_joints = True
        asset_options.replace_cylinder_with_capsule = True
        # asset_options.flip_visual_attachments = True  # for dae
        asset_options.flip_visual_attachments = False  # for stl
        asset_options.fix_base_link = False
        asset_options.density = 0.001
        asset_options.angular_damping = 0.
        asset_options.linear_damping = 0.
        asset_options.max_angular_velocity = 1000.
        asset_options.max_linear_velocity = 1000.
        asset_options.armature = 0.
        asset_options.thickness = 0.01
        asset_options.disable_gravity = False
        config.asset_options = asset_options
        config.self_collisions = 0  # 1 to disable, 0 to enable...bitwise filter
        return config


